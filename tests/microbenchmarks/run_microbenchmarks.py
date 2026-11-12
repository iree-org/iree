#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compiles and benchmarks the compiler microbenchmark cases.

For every (case, config) pair this script:
  1. compiles the case to the flow phase and records its dispatch structure,
  2. compiles the case to a .vmfb,
  3. benchmarks every public function with iree-benchmark-module.

With --base it runs everything for two iree-compile binaries and reports
structural differences and time ratios. See README.md for details.

Only the Python standard library is used.
"""

import argparse
import concurrent.futures
import datetime
import fnmatch
import hashlib
import json
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 1

# Splat value used to synthesize each input element type.
_FLOAT_TYPES = ("f16", "bf16", "f32", "f64")
_INT_TYPES = ("i1", "i8", "i16", "i32", "i64")


###############################################################################
# Manifest and cases
###############################################################################


def load_manifest(path):
    with open(path) as f:
        manifest = json.load(f)
    for key in ("targets", "configs"):
        if key not in manifest:
            raise ValueError(f"{path}: missing '{key}'")
    return manifest


def discover_cases(cases_dir):
    """Returns {case_id: path} for every .mlir file under `cases_dir`."""
    cases = {}
    for path in sorted(Path(cases_dir).rglob("*.mlir")):
        case_id = path.relative_to(cases_dir).with_suffix("").as_posix()
        cases[case_id] = path
    return cases


def _matches(case_id, patterns):
    return any(fnmatch.fnmatchcase(case_id, p) for p in patterns)


def config_applies(manifest, config_name, case_id):
    config = manifest["configs"][config_name]
    if not _matches(case_id, config.get("include", ["*"])):
        return False
    if _matches(case_id, config.get("exclude", [])):
        return False
    for pattern, override in manifest.get("case_overrides", {}).items():
        if not fnmatch.fnmatchcase(case_id, pattern):
            continue
        only = override.get("only_configs")
        if only is not None and config_name not in only:
            return False
        if config_name in override.get("exclude_configs", []):
            return False
    return True


def benchmark_allowed(manifest, case_id):
    for pattern, override in manifest.get("case_overrides", {}).items():
        if fnmatch.fnmatchcase(case_id, pattern) and override.get("skip_benchmark"):
            return False
    return True


_FUNC_RE = re.compile(r"func\.func\s+(?:(public|private|nested)\s+)?@([\w$.-]+)\s*\(")


def _split_top_level(text, sep=","):
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def parse_entry_functions(source):
    """Returns [(name, [arg_type, ...])] for public func.func ops."""
    funcs = []
    for m in _FUNC_RE.finditer(source):
        if m.group(1) in ("private", "nested"):
            continue
        depth, i = 1, m.end()
        while depth:
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
            i += 1
        args = source[m.end() : i - 1]
        types = []
        for arg in _split_top_level(args):
            # `%name: type {attrs}` -> type
            arg_type = arg.split(":", 1)[1].strip()
            arg_type = re.sub(r"\s*\{.*\}\s*$", "", arg_type, flags=re.S)
            types.append(arg_type)
        funcs.append((m.group(2), types))
    return funcs


def synthesize_input(arg_type):
    """Returns an --input= value for a static tensor or scalar type."""
    m = re.fullmatch(r"tensor<((?:\d+x)*)([a-z]\w*)>", arg_type)
    if m:
        shape, elem = m.group(1).rstrip("x"), m.group(2)
    elif re.fullmatch(r"\w+", arg_type):
        shape, elem = "", arg_type
    else:
        raise ValueError(f"cannot synthesize an input for '{arg_type}'")
    if elem in _FLOAT_TYPES:
        value = "0.5"
    elif elem in _INT_TYPES:
        value = "1"
    else:
        raise ValueError(f"cannot synthesize an input for '{arg_type}'")
    return f"{shape}x{elem}={value}" if shape else f"{elem}={value}"


###############################################################################
# Structure extraction
###############################################################################

_EXECUTABLE_RE = re.compile(r"^\s*flow\.executable\s+(?:private\s+|public\s+)?@")
_EXPORT_RE = re.compile(r"flow\.executable\.export\s+(?:public\s+)?@([\w$.-]+)")
_DISPATCH_PREFIX_RE = re.compile(r"^.*?_dispatch_\d+_")
_ENC_IN_RE = re.compile(r"dispatch\.tensor<readonly:tensor<[^>]*#iree_encoding\.")
_ENC_OUT_RE = re.compile(
    r"dispatch\.tensor<(?:writeonly|readwrite):tensor<[^>]*#iree_encoding\."
)
_MMT4D_RE = re.compile(r"\blinalg\.(?:batch_)?mmt4d\b")
_PACK_RE = re.compile(r"\blinalg\.pack\b")
_UNPACK_RE = re.compile(r"\blinalg\.unpack\b")
_OP_RE = re.compile(r"\b((?:linalg|iree_linalg_ext|tensor)\.[a-z_0-9]+)\b")
_IGNORED_OPS = {
    "linalg.yield",
    "linalg.index",
    "linalg.fill",
    "iree_linalg_ext.yield",
    "tensor.empty",
    "tensor.yield",
    "tensor.dim",
}
_TENSOR_ENCODE_RE = re.compile(r"\bflow\.tensor\.encode\b")


def _executable_blocks(flow_ir):
    """Yields the text of each top-level flow.executable op."""
    lines = flow_ir.splitlines()
    i = 0
    while i < len(lines):
        if not _EXECUTABLE_RE.match(lines[i]):
            i += 1
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        j = i + 1
        while j < len(lines):
            line = lines[j]
            if line.strip() == "}" and len(line) - len(line.lstrip()) == indent:
                break
            j += 1
        yield "\n".join(lines[i : j + 1]), (i, j)
        i = j + 1


def extract_structure(flow_ir):
    """Summarizes dispatches of a module compiled with --compile-to=flow.

    Data tiling is detected from the dispatch bodies:
      enc_in   a dispatch reads a tensor with an #iree_encoding (the root op of
               the dispatch consumes data-tiled operands; materialized to
               mmt4d by codegen),
      enc_out  a dispatch writes an encoded tensor (a fused set_encoding
               producer),
      mmt4d/pack/unpack  layouts already materialized before dispatch creation
               (e.g. early materialization).
    A dispatch counts as data-tiled when it has enc_in or mmt4d.
    """
    dispatches = []
    host_lines = flow_ir.splitlines()
    exec_ranges = []
    for block, line_range in _executable_blocks(flow_ir):
        exec_ranges.append(line_range)
        tags = []
        if _ENC_IN_RE.search(block):
            tags.append("enc_in")
        if _ENC_OUT_RE.search(block):
            tags.append("enc_out")
        if _MMT4D_RE.search(block):
            tags.append("mmt4d")
        if _PACK_RE.search(block):
            tags.append("pack")
        if _UNPACK_RE.search(block):
            tags.append("unpack")
        ops = sorted(set(_OP_RE.findall(block)) - _IGNORED_OPS)
        for export in _EXPORT_RE.findall(block):
            dispatches.append(
                {
                    "export": export,
                    "name": _DISPATCH_PREFIX_RE.sub("", export),
                    "tags": tags,
                    "data_tiled": "enc_in" in tags or "mmt4d" in tags,
                    "ops": ops,
                }
            )
    host = [
        line
        for k, line in enumerate(host_lines)
        if not any(a <= k <= b for a, b in exec_ranges)
    ]
    return {
        "dispatches": dispatches,
        "num_dispatches": len(dispatches),
        "data_tiled": any(d["data_tiled"] for d in dispatches),
        "host_tensor_encodes": sum(
            len(_TENSOR_ENCODE_RE.findall(line)) for line in host
        ),
    }


def structure_signature(structure):
    """The part of a structure that A/B comparisons check for equality."""
    if structure is None:
        return None
    return [
        f"{d['name']}{' [' + ','.join(d['tags']) + ']' if d['tags'] else ''}"
        for d in structure["dispatches"]
    ]


###############################################################################
# Codegen configuration extraction
###############################################################################

_CONFIG_FUNC_RE = re.compile(r"^\s*func\.func\s+(?:private\s+|public\s+)?@([\w$.-]+)\(")
_CONFIG_OP_RE = re.compile(r"(?:=\s*)?\b([a-z_][a-z_0-9]*(?:\.[a-z_0-9]+)+)\b")


def _balanced_attr(text, start):
    """Returns the `#dialect.attr<...>` text starting at `start`."""
    open_at = text.find("<", start)
    if open_at < 0:
        return text[start:].split()[0]
    depth = 0
    for i in range(open_at, len(text)):
        ch = text[i]
        if ch == "<":
            depth += 1
        elif ch == ">" and text[i - 1] != "-":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _attr_value(line, name):
    marker = f"{name} = "
    at = line.find(marker)
    return _balanced_attr(line, at + len(marker)) if at >= 0 else None


def extract_codegen_configs(config_ir):
    """Summarizes a module compiled with --compile-to=executable-configurations.

    For every dispatch function this records the `translation_info` (the
    codegen pipeline and its options) and, in order, each op that carries a
    `lowering_config` (tile sizes and similar per-op decisions). This is what
    strategy selection decided for the dispatch, before any lowering runs. It
    is target-specific, unlike the flow-phase structure.
    """
    functions = []
    current = None
    for line in config_ir.splitlines():
        m = _CONFIG_FUNC_RE.match(line)
        if m:
            current = {
                "function": m.group(1),
                "name": _DISPATCH_PREFIX_RE.sub("", m.group(1)),
                "translation_info": _attr_value(line, "translation_info"),
                "ops": [],
            }
            functions.append(current)
            continue
        if current is None or "lowering_config = " not in line:
            continue
        stripped = line.strip()
        # `%r = op.name ...` or `op.name ...` for ops without results.
        rhs = stripped.split(" = ", 1)[1] if stripped.startswith("%") else stripped
        op = _CONFIG_OP_RE.match(rhs)
        current["ops"].append(
            {
                "op": op.group(1) if op else "?",
                "lowering_config": _attr_value(line, "lowering_config"),
            }
        )
    return functions


def codegen_signature(codegen):
    """The part of the codegen configs that A/B comparisons check."""
    if codegen is None:
        return None
    lines = []
    for f in codegen:
        lines.append(f"{f['name']}: {f['translation_info'] or '(no translation_info)'}")
        lines += [f"  {o['op']}: {o['lowering_config']}" for o in f["ops"]]
    return lines


###############################################################################
# Tool invocation
###############################################################################


def _run(cmd, timeout):
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = -1, e.stdout or "", f"timeout after {timeout}s"
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
    return rc, out, err, time.monotonic() - start


def _first_error(log):
    for line in log.splitlines():
        if "error:" in line:
            return line.strip()[:300]
    lines = [l for l in log.strip().splitlines() if l.strip()]
    return lines[-1][:300] if lines else "unknown error"


def compile_case(job):
    """Runs the flow and vmfb compiles for one (side, case, config) job."""
    out_dir = job["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / job["case_id"].replace("/", "__")
    flags = job["flags"]
    result = {"status": "ok", "error": None, "structure": None}

    flow_path = Path(f"{stem}.flow.mlir")
    cmd = [job["compile"], str(job["case_path"]), *flags, "--compile-to=flow"]
    cmd += ["-o", str(flow_path)]
    rc, out, err, secs = _run(cmd, job["timeout"])
    result["flow_compile_cmd"] = shlex.join(cmd)
    if rc != 0:
        Path(f"{stem}.flow.log").write_text(out + err)
        error = _first_error(out + err).replace(str(job["case_path"]), job["case_id"])
        result.update(status="compile_error", error=error)
        return result
    result["structure"] = extract_structure(flow_path.read_text())
    if not job["keep_flow"]:
        flow_path.unlink()

    if job["codegen_configs"]:
        config_path = Path(f"{stem}.configs.mlir")
        cmd = [job["compile"], str(job["case_path"]), *flags]
        cmd += ["--compile-to=executable-configurations", "-o", str(config_path)]
        rc, out, err, secs = _run(cmd, job["timeout"])
        result["configs_compile_cmd"] = shlex.join(cmd)
        if rc != 0:
            Path(f"{stem}.configs.log").write_text(out + err)
            error = _first_error(out + err).replace(
                str(job["case_path"]), job["case_id"]
            )
            result.update(status="compile_error", error=error)
            return result
        result["codegen"] = extract_codegen_configs(config_path.read_text())
        if not job["keep_flow"]:
            config_path.unlink()

    if job["structure_only"]:
        return result

    vmfb_path = Path(f"{stem}.vmfb")
    cmd = [job["compile"], str(job["case_path"]), *flags, "-o", str(vmfb_path)]
    rc, out, err, secs = _run(cmd, job["timeout"])
    result["vmfb_compile_cmd"] = shlex.join(cmd)
    result["vmfb_compile_seconds"] = round(secs, 3)
    if rc != 0:
        Path(f"{stem}.vmfb.log").write_text(out + err)
        error = _first_error(out + err).replace(str(job["case_path"]), job["case_id"])
        result.update(status="compile_error", error=error)
        return result
    result["vmfb"] = str(vmfb_path)
    result["vmfb_sha256"] = hashlib.sha256(vmfb_path.read_bytes()).hexdigest()
    return result


def _remote_cmd(remote, shell_command):
    """Command line that runs `shell_command` on the remote benchmark host."""
    return [*remote["ssh"], remote["host"], shell_command]


def _remote_copy(remote, local_path, remote_path):
    """Copies a file to the remote host through the ssh command itself."""
    cmd = _remote_cmd(remote, f"cat > {remote_path}")
    with open(local_path, "rb") as f:
        proc = subprocess.run(cmd, stdin=f, capture_output=True, check=False)
    return proc.returncode == 0, proc.stderr.decode(errors="replace")


def _to_ms(value, unit):
    return value * {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}[unit]


def benchmark_function(args, side, vmfb, run_flags, func, arg_types, log_stem):
    try:
        inputs = [f"--input={synthesize_input(t)}" for t in arg_types]
    except ValueError as e:
        return {"status": "input_error", "error": str(e)}
    out_json = Path(f"{log_stem}.{func}.bench.json")
    remote = args.remote
    if remote:
        # The module is copied by benchmark_entries(); the benchmark JSON is
        # written on the remote host and fetched afterwards.
        module = f"{remote['dir']}/{side}.vmfb"
        bench_out = f"{remote['dir']}/{side}.bench.json"
    else:
        module, bench_out = vmfb, out_json
    cmd = [
        remote["iree_benchmark_module"] if remote else args.benchmark_tools[side],
        f"--module={module}",
        f"--function={func}",
        *run_flags,
        *inputs,
        f"--benchmark_repetitions={args.repetitions}",
        f"--benchmark_min_time={args.min_time}",
        f"--benchmark_out={bench_out}",
        "--benchmark_out_format=json",
    ]
    if args.threads:
        cmd.append(f"--task_topology_group_count={args.threads}")
    if args.dispatch_stats:
        cmd.append("--dispatch_statistics")
    if remote:
        out_json.unlink(missing_ok=True)
        cmd = _remote_cmd(remote, f"rm -f {bench_out}; {shlex.join(cmd)}")
    result = {"cmd": shlex.join(cmd)}
    rc, out, err, _ = _run(cmd, args.bench_timeout)
    if rc == -1:
        # Rare benchmark hangs have been seen that did not reproduce; retry once.
        result["timeouts"] = 1
        rc, out, err, _ = _run(cmd, args.bench_timeout)
    if remote and rc == 0:
        frc, fout, ferr, _ = _run(_remote_cmd(remote, f"cat {bench_out}"), 60)
        if frc == 0:
            out_json.write_text(fout)
        else:
            err += ferr
    if rc != 0 or not out_json.exists():
        Path(f"{log_stem}.{func}.bench.log").write_text(out + err)
        result.update(status="run_error", error=_first_error(out + err))
        return result
    data = json.loads(out_json.read_text())
    samples, dispatch_rows = [], {}
    for row in data.get("benchmarks", []):
        if row.get("run_type") != "iteration":
            continue
        ms = _to_ms(row["real_time"], row["time_unit"])
        if "process_time" in row["name"]:
            samples.append(ms)
        elif "calls" in row:
            name = row["name"].split("/", 1)[-1]
            dispatch_rows.setdefault(name, []).append(ms)
    if not samples:
        result.update(status="run_error", error="no benchmark samples in output")
        return result
    median = statistics.median(samples)
    result.update(
        status="ok",
        samples_ms=samples,
        median_ms=median,
        min_ms=min(samples),
        max_ms=max(samples),
        cv=(
            (statistics.stdev(samples) / statistics.mean(samples))
            if len(samples) > 1
            else 0.0
        ),
        context={
            k: data.get("context", {}).get(k)
            for k in ("num_cpus", "mhz_per_cpu", "cpu_scaling_enabled", "load_avg")
        },
    )
    if dispatch_rows:
        result["dispatch_median_ms"] = {
            k: statistics.median(v) for k, v in dispatch_rows.items()
        }
    return result


###############################################################################
# Metadata
###############################################################################


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# iree-compile is a thin launcher; the compiler itself is this shared library.
_COMPILER_LIBS = ("libIREECompiler.dylib", "libIREECompiler.so", "IREECompiler.dll")


def tool_metadata(path):
    path = Path(path).resolve()
    meta = {"path": str(path)}
    try:
        meta["sha256"] = _file_hash(path)
        meta["mtime"] = datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(
            timespec="seconds"
        )
        for lib_dir in (path.parent.parent / "lib", path.parent):
            for name in _COMPILER_LIBS:
                lib = lib_dir / name
                if lib.exists():
                    meta["compiler_lib"] = str(lib)
                    meta["compiler_lib_sha256"] = _file_hash(lib)
                    # Identify the build by the library when there is one.
                    meta["sha256"] = meta["compiler_lib_sha256"]
                    break
            if "compiler_lib" in meta:
                break
    except OSError as e:
        meta["error"] = str(e)
    rc, out, err, _ = _run([str(path), "--version"], 60)
    meta["version"] = (out + err).strip() if rc == 0 else None
    return meta


def host_metadata():
    meta = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if sys.platform == "darwin":
        rc, out, _, _ = _run(["sysctl", "-n", "machdep.cpu.brand_string"], 10)
        if rc == 0:
            meta["cpu"] = out.strip()
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    meta["cpu"] = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return meta


###############################################################################
# Comparison and reporting
###############################################################################


def _fmt_ms(v):
    if v is None:
        return "-"
    return f"{v:.3f}" if v < 100 else f"{v:.1f}"


def _dispatch_cell(entry):
    structure = entry.get("structure")
    if structure is None:
        return "-"
    sig = structure_signature(structure)
    return "<br>".join(f"`{s}`" for s in sig) if sig else "(none)"


def _func_times(entry):
    """{function: median_ms} for the functions that ran."""
    return {
        f: r["median_ms"]
        for f, r in entry.get("benchmarks", {}).items()
        if r.get("status") == "ok"
    }


def _status(entry):
    if entry is None:
        return "missing"
    if entry["status"] != "ok":
        return entry["status"]
    bad = [r["status"] for r in entry.get("benchmarks", {}).values()]
    bad = [s for s in bad if s != "ok"]
    return bad[0] if bad else "ok"


def compare(base, head, threshold):
    """Compares two results dicts. Returns a list of per-entry diffs."""
    rows = []
    keys = sorted(set(base["entries"]) | set(head["entries"]))
    for key in keys:
        b, h = base["entries"].get(key), head["entries"].get(key)
        row = {
            "key": key,
            "base_status": _status(b),
            "head_status": _status(h),
            "structural_change": False,
            "codegen_change": False,
            "time_change": None,
            "functions": {},
        }
        bs = structure_signature(b.get("structure")) if b else None
        hs = structure_signature(h.get("structure")) if h else None
        if row["base_status"] != row["head_status"] or bs != hs:
            row["structural_change"] = True
            row["removed"] = [s for s in (bs or []) if s not in (hs or [])]
            row["added"] = [s for s in (hs or []) if s not in (bs or [])]
            row["base_dispatches"], row["head_dispatches"] = bs, hs
        bh, hh = (b or {}).get("vmfb_sha256"), (h or {}).get("vmfb_sha256")
        row["identical_modules"] = bh is not None and bh == hh
        bc = codegen_signature(b.get("codegen")) if b else None
        hc = codegen_signature(h.get("codegen")) if h else None
        if bc is not None and hc is not None and bc != hc:
            row["codegen_change"] = True
            row["base_codegen"] = [s for s in bc if s not in hc]
            row["head_codegen"] = [s for s in hc if s not in bc]
        bt, ht = (_func_times(b) if b else {}), (_func_times(h) if h else {})
        for func in sorted(set(bt) & set(ht)):
            ratio = ht[func] / bt[func] if bt[func] > 0 else None
            br, hr = b["benchmarks"][func], h["benchmarks"][func]
            # Sample ranges that overlap mean the difference is within noise.
            overlap = br["min_ms"] <= hr["max_ms"] and hr["min_ms"] <= br["max_ms"]
            row["functions"][func] = {
                "base_ms": bt[func],
                "head_ms": ht[func],
                "ratio": ratio,
                "ranges_overlap": overlap,
            }
            if ratio is None or abs(ratio - 1.0) <= threshold:
                continue
            if row["identical_modules"]:
                # Only reachable with --bench-identical: same code, so noise.
                row["time_change"] = row["time_change"] or "noise"
                continue
            if overlap:
                row["time_change"] = row["time_change"] or "noise"
                continue
            change = "slower" if ratio > 1.0 else "faster"
            # Report the worst direction if functions disagree.
            if row["time_change"] != "slower":
                row["time_change"] = change
        rows.append(row)
    return rows


def _md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _metadata_md(results, label):
    meta = results["metadata"]
    tool = meta["iree_compile"]
    label = f"{label} " if label else ""
    lines = [
        f"- {label}iree-compile: `{tool['path']}`"
        f" (sha256 {str(tool.get('sha256'))[:12]}, mtime {tool.get('mtime')})"
    ]
    if meta.get("extra_compile_flags"):
        flags = " ".join(meta["extra_compile_flags"])
        lines.append(f"- {label}extra flags: `{flags}`")
    return lines


def render_single(results):
    meta = results["metadata"]
    out = ["# Compiler microbenchmarks", ""]
    out += _metadata_md(results, "")
    out += [
        f"- host: {meta['host'].get('cpu', meta['host']['machine'])},"
        f" {meta['host']['platform']}",
        f"- target: `{meta['target']}`, mode: {meta['mode']},"
        f" repetitions: {meta['repetitions']}, min time: {meta['min_time']},"
        f" threads: {meta['threads'] or 'default'}",
        "",
        "Dispatch tags: `enc_in` = consumes data-tiled operands, `enc_out` ="
        " produces an encoded tensor, `mmt4d`/`pack`/`unpack` = materialized"
        " layouts.",
        "",
    ]
    rows = []
    for key in sorted(results["entries"]):
        entry = results["entries"][key]
        case_id, config = key.split("@")
        times = _func_times(entry)
        status = _status(entry)
        time_cell = "<br>".join(
            f"{_fmt_ms(t)}" + (f" ({f})" if len(times) > 1 else "")
            for f, t in times.items()
        )
        cv_cell = "<br>".join(
            f"{100 * r['cv']:.1f}%"
            for r in entry.get("benchmarks", {}).values()
            if r.get("status") == "ok"
        )
        if status != "ok":
            err = entry.get("error") or next(
                (
                    r.get("error")
                    for r in entry.get("benchmarks", {}).values()
                    if r.get("error")
                ),
                "",
            )
            status = f"{status}: {err}".replace("|", "\\|")
        rows.append(
            [
                case_id,
                config,
                _dispatch_cell(entry),
                "yes" if (entry.get("structure") or {}).get("data_tiled") else "no",
                time_cell or "-",
                cv_cell or "-",
                status,
            ]
        )
    out.append(
        _md_table(
            ["case", "config", "dispatches", "DT", "median ms", "cv", "status"], rows
        )
    )
    return "\n".join(out) + "\n"


def render_compare(base, head, rows, threshold):
    out = ["# Compiler microbenchmarks: A/B", ""]
    out += _metadata_md(base, "base")
    out += _metadata_md(head, "head")
    meta = head["metadata"]
    out += [
        f"- host: {meta['host'].get('cpu', meta['host']['machine'])},"
        f" {meta['host']['platform']}",
        f"- target: `{meta['target']}`, mode: {meta['mode']},"
        f" repetitions: {meta['repetitions']}, min time: {meta['min_time']},"
        f" threads: {meta['threads'] or 'default'}, threshold:"
        f" {100 * threshold:.0f}%",
        "",
    ]
    structural = [r for r in rows if r["structural_change"]]
    timed = [r for r in rows if r["time_change"] in ("slower", "faster")]
    if any(r["identical_modules"] for r in rows):
        identical = sum(r["identical_modules"] for r in rows)
        out += [
            f"{identical} of {sum('vmfb_sha256' in e for e in head['entries'].values())}"
            " compiled modules are byte-identical between base and head; those"
            " entries are not timed (use `--bench-identical` to time them"
            " anyway).",
            "",
        ]
    out += [f"## Structural changes ({len(structural)})", ""]
    if structural:
        srows = []
        for r in structural:
            case_id, config = r["key"].split("@")
            status = (
                f"{r['base_status']} -> {r['head_status']}"
                if r["base_status"] != r["head_status"]
                else r["head_status"]
            )
            srows.append(
                [
                    case_id,
                    config,
                    status,
                    "<br>".join(f"`{s}`" for s in r.get("removed", [])) or "-",
                    "<br>".join(f"`{s}`" for s in r.get("added", [])) or "-",
                ]
            )
        header = ["case", "config", "status", "base only", "head only"]
        out.append(_md_table(header, srows))
    else:
        out.append("None.")
    if any(e.get("codegen") is not None for e in head["entries"].values()):
        codegen = [r for r in rows if r["codegen_change"]]
        out += ["", f"## Codegen configuration changes ({len(codegen)})", ""]
        if codegen:
            crows = []
            for r in codegen:
                case_id, config = r["key"].split("@")
                crows.append(
                    [
                        case_id,
                        config,
                        "<br>".join(f"`{s.strip()}`" for s in r["base_codegen"]) or "-",
                        "<br>".join(f"`{s.strip()}`" for s in r["head_codegen"]) or "-",
                    ]
                )
            out.append(_md_table(["case", "config", "base only", "head only"], crows))
        else:
            out.append("None.")
    out += [
        "",
        f"## Time changes over {100 * threshold:.0f}% outside noise ({len(timed)})",
        "",
    ]
    if timed:
        trows = []
        for r in timed:
            case_id, config = r["key"].split("@")
            for func, f in r["functions"].items():
                trows.append(
                    [
                        case_id,
                        config,
                        func,
                        _fmt_ms(f["base_ms"]),
                        _fmt_ms(f["head_ms"]),
                        f"{f['ratio']:.2f}",
                    ]
                )
        header = ["case", "config", "function", "base ms", "head ms", "ratio"]
        out.append(_md_table(header, trows))
    else:
        out.append("None.")
    out += ["", "## All results", ""]
    arows = []
    for r in rows:
        case_id, config = r["key"].split("@")
        funcs = r["functions"]
        arows.append(
            [
                case_id,
                config,
                "<br>".join(_fmt_ms(f["base_ms"]) for f in funcs.values()) or "-",
                "<br>".join(_fmt_ms(f["head_ms"]) for f in funcs.values()) or "-",
                "<br>".join(
                    f"{f['ratio']:.2f}" if f["ratio"] is not None else "-"
                    for f in funcs.values()
                )
                or "-",
                ("CHANGED" if r["structural_change"] else "same"),
                ("CHANGED" if r["codegen_change"] else "same"),
                ("identical" if r["identical_modules"] else "differ"),
                r["time_change"] or "-",
            ]
        )
    out.append(
        _md_table(
            [
                "case",
                "config",
                "base ms",
                "head ms",
                "ratio",
                "structure",
                "codegen",
                "module",
                "time",
            ],
            arows,
        )
    )
    return "\n".join(out) + "\n"


def structure_only_view(results):
    """Deterministic subset of results (no times, paths or host data)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "target": results["metadata"]["target"],
        "iree_compile_version": results["metadata"]["iree_compile"].get("version"),
        "entries": {
            key: {
                "status": e["status"],
                "error": e.get("error"),
                "dispatches": structure_signature(e.get("structure")),
                "data_tiled": (e.get("structure") or {}).get("data_tiled"),
                **(
                    {"codegen": codegen_signature(e["codegen"])}
                    if e.get("codegen") is not None
                    else {}
                ),
            }
            for key, e in sorted(results["entries"].items())
        },
    }


###############################################################################
# Main
###############################################################################


def parse_args(argv):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--head",
        "--iree-compile",
        dest="head",
        help="iree-compile under test (required unless --compare-results).",
    )
    p.add_argument("--base", help="Baseline iree-compile; enables A/B mode.")
    p.add_argument(
        "--iree-benchmark-module",
        help="iree-benchmark-module used for both sides (default: next to --head).",
    )
    p.add_argument(
        "--base-iree-benchmark-module",
        help="iree-benchmark-module for the base side (default: same as head).",
    )
    p.add_argument(
        "--head-flags",
        default="",
        help="Extra compile flags for the head side (one shell-quoted string).",
    )
    p.add_argument(
        "--base-flags",
        default=None,
        help="Extra compile flags for the base side. With --base omitted and"
        " this set, the head binary is used for both sides (flag A/B).",
    )
    p.add_argument("--manifest", default=str(SCRIPT_DIR / "manifest.json"))
    p.add_argument("--cases-dir", default=str(SCRIPT_DIR / "cases"))
    p.add_argument("--target", help="Target from the manifest (default: its default).")
    p.add_argument(
        "--case",
        action="append",
        default=[],
        help="Glob over case ids, e.g. 'conv/*' or '*shortcut*'. Repeatable.",
    )
    p.add_argument(
        "--exclude-case", action="append", default=[], help="Glob to skip. Repeatable."
    )
    p.add_argument(
        "--config", action="append", default=[], help="Config name. Repeatable."
    )
    p.add_argument(
        "--structure-only",
        action="store_true",
        help="Only compile to flow and record dispatch structure; no timing.",
    )
    p.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile the modules and compare them between base and head, but"
        " run no benchmarks. Works for cross-compiled targets and shows"
        " whether a change affects the generated code at all.",
    )
    p.add_argument(
        "--codegen-configs",
        action="store_true",
        help="Also record what strategy selection chose per dispatch: the"
        " codegen pipeline and each op's lowering_config"
        " (--compile-to=executable-configurations).",
    )
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument(
        "--min-time",
        default="0.2s",
        help="--benchmark_min_time per repetition (e.g. 0.2s or 100x).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Task workers (--task_topology_group_count); 0 = runtime default.",
    )
    p.add_argument(
        "--dispatch-stats",
        action="store_true",
        help="Also record per-dispatch times (--dispatch_statistics).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Relative time change flagged in A/B mode (default 0.05 = 5%%).",
    )
    p.add_argument(
        "--bench-identical",
        action="store_true",
        help="In A/B mode, also time entries whose base and head modules are"
        " byte-identical (skipped by default).",
    )
    p.add_argument(
        "--remote-host",
        help="Run the benchmarks on this host over ssh (e.g. user@host),"
        " for targets that are cross-compiled. Overrides the target's"
        " 'remote.host' in the manifest.",
    )
    p.add_argument(
        "--remote-dir",
        help="Scratch directory on the remote host (default: the target's"
        " 'remote.dir' or /tmp/iree-microbench).",
    )
    p.add_argument(
        "--remote-iree-benchmark-module",
        help="Path of iree-benchmark-module on the remote host.",
    )
    p.add_argument(
        "--ssh-command",
        help="ssh command line used to reach the remote host (default: 'ssh').",
    )
    p.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--timeout", type=int, default=600, help="Per compile, seconds.")
    p.add_argument(
        "--bench-timeout",
        type=int,
        default=120,
        help="Per benchmark call, seconds; a timed-out call is retried once.",
    )
    p.add_argument("-o", "--output-dir", help="Default: ./microbench-<timestamp>.")
    p.add_argument(
        "--keep-flow",
        action="store_true",
        help="Keep the flow-phase (and codegen configuration) .mlir files.",
    )
    p.add_argument("--list", action="store_true", help="List case x config and exit.")
    p.add_argument(
        "--compare-results",
        nargs=2,
        metavar=("BASE_JSON", "HEAD_JSON"),
        help="Compare two existing results.json files and exit.",
    )
    p.add_argument(
        "--fail-on-structural-change",
        action="store_true",
        help="Exit with 1 if A/B finds a structural change.",
    )
    p.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with 1 if any compile or run fails.",
    )
    args = p.parse_args(argv)
    if not args.compare_results and not args.head and not args.list:
        p.error("--head (or --iree-compile) is required")
    return args


def select_work(args, manifest):
    cases = discover_cases(args.cases_dir)
    configs = args.config or list(manifest["configs"])
    for c in configs:
        if c not in manifest["configs"]:
            raise SystemExit(f"unknown config '{c}'")
    work = []
    for case_id, path in cases.items():
        if args.case and not _matches(case_id, args.case):
            continue
        if _matches(case_id, args.exclude_case):
            continue
        for config in configs:
            if config_applies(manifest, config, case_id):
                work.append((case_id, path, config))
    return work


def run_side(args, manifest, side, compile_tool, extra_flags, work, out_root):
    target = manifest["targets"][args.target]
    jobs = []
    for case_id, path, config in work:
        flags = target["compile_flags"] + manifest["configs"][config]["compile_flags"]
        jobs.append(
            {
                "key": f"{case_id}@{config}",
                "case_id": case_id,
                "case_path": path,
                "config": config,
                "compile": compile_tool,
                "flags": flags + extra_flags,
                "out_dir": out_root / side / config,
                "structure_only": args.structure_only,
                "keep_flow": args.keep_flow,
                "codegen_configs": args.codegen_configs,
                "timeout": args.timeout,
            }
        )
    entries = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(compile_case, j): j for j in jobs}
        for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            job = futures[fut]
            res = fut.result()
            res.update(case=job["case_id"], config=job["config"])
            entries[job["key"]] = res
            mark = "ok" if res["status"] == "ok" else f"FAIL ({res['error']})"
            print(f"[{side} {n}/{len(jobs)}] compile {job['key']}: {mark}", flush=True)
    return entries


def benchmark_entries(args, manifest, sides, results, work):
    """Benchmarks all compiled entries, alternating sides per entry."""
    run_flags = manifest["targets"][args.target].get("run_flags", [])
    total = len(work)
    for n, (case_id, path, config) in enumerate(work, 1):
        key = f"{case_id}@{config}"
        if not benchmark_allowed(manifest, case_id):
            continue
        funcs = parse_entry_functions(Path(path).read_text())
        hashes = {results[s]["entries"][key].get("vmfb_sha256") for s in sides}
        if len(sides) > 1 and len(hashes) == 1 and None not in hashes:
            # Both compilers produced the same module: any time difference
            # would be noise, so there is nothing to measure.
            for side in sides:
                results[side]["entries"][key]["benchmarks"] = {}
            if not args.bench_identical:
                print(f"[{n}/{total}] skip {key}: identical modules", flush=True)
                continue
        for side in sides:
            entry = results[side]["entries"][key]
            entry["benchmarks"] = {}
            if entry["status"] != "ok":
                continue
            stem = Path(entry["vmfb"]).with_suffix("")
            if args.remote:
                ok, err = _remote_copy(
                    args.remote, entry["vmfb"], f"{args.remote['dir']}/{side}.vmfb"
                )
                if not ok:
                    for func, _ in funcs:
                        entry["benchmarks"][func] = {
                            "status": "run_error",
                            "error": f"copy to remote failed: {err.strip()[:200]}",
                        }
                    continue
            for func, arg_types in funcs:
                r = benchmark_function(
                    args, side, entry["vmfb"], run_flags, func, arg_types, stem
                )
                entry["benchmarks"][func] = r
                msg = (
                    f"{_fmt_ms(r['median_ms'])} ms (cv {100 * r['cv']:.1f}%)"
                    if r["status"] == "ok"
                    else f"FAIL ({r['error']})"
                )
                print(f"[{side} {n}/{total}] run {key} {func}: {msg}", flush=True)


def main(argv):
    args = parse_args(argv)

    if args.compare_results:
        base = json.loads(Path(args.compare_results[0]).read_text())
        head = json.loads(Path(args.compare_results[1]).read_text())
        rows = compare(base, head, args.threshold)
        print(render_compare(base, head, rows, args.threshold))
        changed = any(r["structural_change"] or r["codegen_change"] for r in rows)
        return 1 if changed and args.fail_on_structural_change else 0

    manifest = load_manifest(args.manifest)
    args.target = args.target or manifest.get("default_target")
    if args.target not in manifest["targets"]:
        raise SystemExit(f"unknown target '{args.target}'")
    work = select_work(args, manifest)
    if args.list:
        for case_id, _, config in work:
            print(f"{case_id}@{config}")
        return 0
    if not work:
        raise SystemExit("no (case, config) pairs selected")

    head_flags = shlex.split(args.head_flags)
    base_flags = shlex.split(args.base_flags or "")
    sides = {"head": (args.head, head_flags)}
    if args.base or args.base_flags is not None:
        sides = {"base": (args.base or args.head, base_flags), **sides}

    bench_tool = args.iree_benchmark_module or str(
        Path(args.head).parent / "iree-benchmark-module"
    )
    args.benchmark_tools = {
        "head": bench_tool,
        "base": args.base_iree_benchmark_module or bench_tool,
    }

    remote_cfg = dict(manifest["targets"][args.target].get("remote", {}))
    if args.remote_host:
        remote_cfg["host"] = args.remote_host
    if args.remote_dir:
        remote_cfg["dir"] = args.remote_dir
    if args.remote_iree_benchmark_module:
        remote_cfg["iree_benchmark_module"] = args.remote_iree_benchmark_module
    args.remote = None
    no_runs = args.structure_only or args.compile_only
    if remote_cfg.get("host") and not no_runs:
        remote_cfg["ssh"] = shlex.split(
            args.ssh_command or remote_cfg.get("ssh", "ssh")
        )
        remote_cfg.setdefault("dir", "/tmp/iree-microbench")
        if not remote_cfg.get("iree_benchmark_module"):
            raise SystemExit(
                "remote runs need --remote-iree-benchmark-module (or"
                " 'remote.iree_benchmark_module' in the manifest target)"
            )
        rc, out, err, _ = _run(
            _remote_cmd(remote_cfg, f"mkdir -p {remote_cfg['dir']} && uname -srm"), 60
        )
        if rc != 0:
            raise SystemExit(f"cannot reach remote host: {err.strip()[:300]}")
        remote_cfg["uname"] = out.strip()
        args.remote = remote_cfg
    elif manifest["targets"][args.target].get("cross_compiled") and not no_runs:
        raise SystemExit(
            f"target '{args.target}' is cross-compiled: use --structure-only,"
            " --compile-only, or give --remote-host to run the benchmarks on a"
            " matching machine"
        )

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.output_dir or f"microbench-{stamp}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    host = host_metadata()
    results = {}
    for side, (tool, extra) in sides.items():
        results[side] = {
            "schema_version": SCHEMA_VERSION,
            "metadata": {
                "side": side,
                "date": datetime.datetime.now().isoformat(timespec="seconds"),
                "iree_compile": tool_metadata(tool),
                "iree_benchmark_module": (
                    None
                    if args.structure_only or args.compile_only or args.remote
                    else tool_metadata(args.benchmark_tools[side])
                ),
                "remote": (
                    {k: v for k, v in args.remote.items() if k != "ssh"}
                    if args.remote
                    else None
                ),
                "extra_compile_flags": extra,
                "target": args.target,
                "target_flags": manifest["targets"][args.target],
                "configs": {
                    c: manifest["configs"][c] for c in sorted({w[2] for w in work})
                },
                "mode": (
                    "structure-only"
                    if args.structure_only
                    else "compile-only" if args.compile_only else "timing"
                ),
                "repetitions": args.repetitions,
                "min_time": args.min_time,
                "threads": args.threads,
                "host": host,
            },
            "entries": run_side(args, manifest, side, tool, extra, work, out_root),
        }

    if not args.structure_only and not args.compile_only:
        benchmark_entries(args, manifest, list(sides), results, work)

    for side, res in results.items():
        (out_root / f"results.{side}.json").write_text(
            json.dumps(res, indent=2, sort_keys=True) + "\n"
        )
        (out_root / f"structure.{side}.json").write_text(
            json.dumps(structure_only_view(res), indent=2, sort_keys=True) + "\n"
        )

    if "base" in results:
        rows = compare(results["base"], results["head"], args.threshold)
        report = render_compare(results["base"], results["head"], rows, args.threshold)
        (out_root / "comparison.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n"
        )
    else:
        rows = []
        report = render_single(results["head"])
    (out_root / "report.md").write_text(report)
    print()
    print(report)
    print(f"Results written to {out_root}")

    failed = any(
        _status(e) != "ok" for r in results.values() for e in r["entries"].values()
    )
    if failed and args.fail_on_error:
        return 1
    if args.fail_on_structural_change and any(
        r["structural_change"] or r["codegen_change"] for r in rows
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
