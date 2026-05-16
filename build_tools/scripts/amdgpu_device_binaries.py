#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Builds precompiled AMDGPU HAL builtin device code objects.

This script intentionally runs outside the normal runtime build. It preserves
the existing in-tree/out-of-tree LLVM tool flows while allowing the runtime
build to consume checked-in code objects without depending on LLVM.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence

import amdgpu_target_map


TARGET_TRIPLE = "amdgcn-amd-amdhsa"
DEFAULT_TARGET_SELECTIONS = amdgpu_target_map.DEFAULT_TARGET_SELECTIONS
DEVICE_BINARY_TARGET_ENV = "IREE_HAL_AMDGPU_DEVICE_BINARY_TARGETS"
DEVICE_PACKAGE_PATH = Path("runtime/src/iree/hal/drivers/amdgpu/device")
DEVICE_SOURCE_LIST_PATH = DEVICE_PACKAGE_PATH / "device_bitcode_sources.bzl"
DEVICE_SOURCE_LIST_VARIABLE = "IREE_HAL_AMDGPU_DEVICE_BITCODE_SRCS"


class Toolchain:
    def __init__(
        self,
        *,
        clang: Path,
        llvm_link: Path,
        lld: Path,
        llvm_objcopy: Path,
        clang_resource_include: Path,
    ):
        self.clang = clang
        self.llvm_link = llvm_link
        self.lld = lld
        self.llvm_objcopy = llvm_objcopy
        self.clang_resource_include = clang_resource_include


def eprint(*args):
    print(*args, file=sys.stderr)


def repo_root_from_script() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "runtime" / "src" / "iree").is_dir():
            return parent
    raise RuntimeError("could not find repository root from script location")


def literal_string_list_from_ast(
    node: ast.AST, *, path: Path, variable_name: str
) -> list[str]:
    try:
        values = ast.literal_eval(node)
    except (SyntaxError, ValueError) as exc:
        raise RuntimeError(
            f"{path}: {variable_name} must be a literal string list"
        ) from exc
    if not isinstance(values, list):
        raise RuntimeError(f"{path}: {variable_name} must be a literal string list")
    if not all(isinstance(value, str) for value in values):
        raise RuntimeError(f"{path}: {variable_name} must contain only strings")
    return list(values)


def load_device_source_paths(repo_root: Path) -> list[Path]:
    source_list_path = repo_root / DEVICE_SOURCE_LIST_PATH
    package_path = repo_root / DEVICE_PACKAGE_PATH
    module = ast.parse(source_list_path.read_text(), filename=str(source_list_path))
    source_names = None
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == DEVICE_SOURCE_LIST_VARIABLE
            for target in node.targets
        ):
            source_names = literal_string_list_from_ast(
                node.value,
                path=source_list_path,
                variable_name=DEVICE_SOURCE_LIST_VARIABLE,
            )
            break
    if source_names is None:
        raise RuntimeError(
            f"{source_list_path}: missing {DEVICE_SOURCE_LIST_VARIABLE} assignment"
        )

    source_paths: list[Path] = []
    for source_name in source_names:
        source_path = Path(source_name)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise RuntimeError(
                f"{source_list_path}: {DEVICE_SOURCE_LIST_VARIABLE} entry must be "
                f"package-relative: {source_name}"
            )
        if source_path.suffix != ".c":
            raise RuntimeError(
                f"{source_list_path}: {DEVICE_SOURCE_LIST_VARIABLE} entry is "
                f"not a C source file: {source_name}"
            )
        absolute_source_path = package_path / source_path
        if not absolute_source_path.is_file():
            raise RuntimeError(
                f"{source_list_path}: {DEVICE_SOURCE_LIST_VARIABLE} entry "
                f"does not exist: {source_name}"
            )
        source_paths.append(absolute_source_path)
    return source_paths


def split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in re.split(r"[,;:\s]+", value) if part]


def append_unique(values: list[Path], new_values: Iterable[Path]):
    seen = {value.resolve() if value.exists() else value for value in values}
    for value in new_values:
        key = value.resolve() if value.exists() else value
        if key not in seen:
            values.append(value)
            seen.add(key)


def executable_path(path: Path) -> Path | None:
    if path.is_file() and os.access(path, os.X_OK):
        return path
    return None


def candidate_tool_dirs(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []

    explicit_tool_dirs = []
    for value in args.tool_dir or []:
        explicit_tool_dirs.extend(split_env_list(value))
    append_unique(dirs, [Path(value) for value in explicit_tool_dirs])

    explicit_rocm_roots = []
    for value in args.rocm_path or []:
        explicit_rocm_roots.extend(Path(part) for part in split_env_list(value))
    for root in explicit_rocm_roots:
        append_unique(dirs, rocm_tool_dirs(root))

    tool_dir_envs = (
        "IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_TOOLS_DIR",
        "IREE_HOST_BIN_DIR",
        "IREE_HOST_TOOLS",
        "IREE_HOST_TOOLS_DIR",
        "IREE_BINARY_DIR",
        "IREE_LLVM_BINARY_DIR",
        "IREE_LLVM_TOOLS_DIR",
        "IREE_LLVM_TOOL_DIR",
        "LLVM_TOOLS_BINARY_DIR",
        "LLVM_BINARY_DIR",
    )
    for env_var in tool_dir_envs:
        for value in split_env_list(os.environ.get(env_var)):
            path = Path(value)
            append_unique(dirs, [path.parent if path.is_file() else path])

    expanded_dirs = []
    for directory in dirs:
        expanded_dirs.append(directory)
        expanded_dirs.append(directory / "bin")
        expanded_dirs.append(directory / "llvm-project" / "bin")
        expanded_dirs.append(directory / "llvm" / "bin")
        expanded_dirs.append(directory / "lib" / "llvm" / "bin")
    dirs = []
    append_unique(dirs, expanded_dirs)

    for root in candidate_rocm_roots(args, include_explicit=False):
        append_unique(dirs, rocm_tool_dirs(root))

    path_dirs = [Path(value) for value in os.environ.get("PATH", "").split(os.pathsep)]
    append_unique(dirs, path_dirs)
    return dirs


def candidate_rocm_roots(
    args: argparse.Namespace, *, include_explicit: bool = True
) -> list[Path]:
    roots: list[Path] = []
    if include_explicit:
        for value in args.rocm_path or []:
            roots.extend(Path(part) for part in split_env_list(value))
    for env_var in (
        "IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH",
        "IREE_ROCM_PATH",
        "ROCM_PATH",
        "ROCM_ROOT",
        "ROCM_HOME",
        "HIP_PATH",
    ):
        for value in split_env_list(os.environ.get(env_var)):
            path = Path(value)
            roots.append(path)
            if env_var == "HIP_PATH" and path.name == "hip":
                roots.append(path.parent)
    if hipcc_path := shutil.which("hipcc"):
        roots.append(Path(hipcc_path).resolve().parent.parent)
    if Path("/opt/rocm").exists():
        roots.append(Path("/opt/rocm"))
    result: list[Path] = []
    append_unique(result, roots)
    return result


def rocm_tool_dirs(root: Path) -> list[Path]:
    return [
        root / "llvm" / "bin",
        root / "lib" / "llvm" / "bin",
        root / "bin",
    ]


def find_tool(
    *,
    explicit_path: str | None,
    env_vars: Sequence[str],
    names: Sequence[str],
    tool_dirs: Sequence[Path],
    additional_candidates: Sequence[Path] = (),
    description: str,
) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    for env_var in env_vars:
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))
    candidates.extend(additional_candidates)
    for directory in tool_dirs:
        for name in names:
            candidates.append(directory / name)
    for name in names:
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    for candidate in candidates:
        if executable := executable_path(candidate):
            return executable.resolve()

    env_list = ", ".join(env_vars)
    name_list = ", ".join(names)
    raise RuntimeError(
        f"could not find {description}; tried explicit flags, env vars "
        f"[{env_list}], tool dirs, and PATH names [{name_list}]"
    )


def run_capture(args: Sequence[str]) -> str:
    result = subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip()


def clang_tool_candidates(clang: Path, names: Sequence[str]) -> list[Path]:
    candidates: list[Path] = []
    for name in names:
        try:
            printed_name = run_capture([str(clang), f"--print-prog-name={name}"])
        except subprocess.CalledProcessError:
            continue
        if not printed_name:
            continue
        path = Path(printed_name)
        if path.is_absolute():
            append_unique(candidates, [path])
        else:
            append_unique(candidates, [clang.parent / path, path])
    return candidates


def detect_clang_resource_include(clang: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
    elif os.environ.get("IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_RESOURCE_INCLUDE"):
        path = Path(
            os.environ["IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_RESOURCE_INCLUDE"]
        )
    elif os.environ.get("IREE_CLANG_BUILTIN_HEADERS_PATH"):
        path = Path(os.environ["IREE_CLANG_BUILTIN_HEADERS_PATH"])
    else:
        resource_dir = Path(run_capture([str(clang), "-print-resource-dir"]))
        path = resource_dir / "include"
    if not path.is_dir():
        raise RuntimeError(f"clang resource include directory not found: {path}")
    return path.resolve()


def detect_toolchain(args: argparse.Namespace) -> Toolchain:
    dirs = candidate_tool_dirs(args)
    clang = find_tool(
        explicit_path=args.clang,
        env_vars=(
            "IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_BINARY",
            "IREE_CLANG_BINARY",
            "CLANG",
        ),
        names=("clang", "amdclang", "clang-22"),
        tool_dirs=dirs,
        description="clang with AMDGPU support",
    )
    llvm_link = find_tool(
        explicit_path=args.llvm_link,
        env_vars=(
            "IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_LINK_BINARY",
            "IREE_LLVM_LINK_BINARY",
            "LLVM_LINK",
        ),
        names=("llvm-link",),
        tool_dirs=dirs,
        additional_candidates=clang_tool_candidates(clang, ("llvm-link",)),
        description="llvm-link",
    )
    lld = find_tool(
        explicit_path=args.lld,
        env_vars=(
            "IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLD_BINARY",
            "IREE_LLD_BINARY",
            "LLD",
            "LD_LLD",
        ),
        names=("lld", "ld.lld"),
        tool_dirs=dirs,
        additional_candidates=clang_tool_candidates(clang, ("lld", "ld.lld")),
        description="lld",
    )
    llvm_objcopy = find_tool(
        explicit_path=args.llvm_objcopy,
        env_vars=(
            "IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_OBJCOPY_BINARY",
            "IREE_LLVM_OBJCOPY_BINARY",
            "LLVM_OBJCOPY",
        ),
        names=("llvm-objcopy",),
        tool_dirs=dirs,
        additional_candidates=clang_tool_candidates(clang, ("llvm-objcopy",)),
        description="llvm-objcopy",
    )
    return Toolchain(
        clang=clang,
        llvm_link=llvm_link,
        lld=lld,
        llvm_objcopy=llvm_objcopy,
        clang_resource_include=detect_clang_resource_include(
            clang, args.clang_resource_include
        ),
    )


def expand_target_selections(selections: Sequence[str]) -> list[str]:
    code_object_targets = []
    all_exact_targets = set(amdgpu_target_map.exact_targets())
    all_code_object_targets = set(amdgpu_target_map.code_object_targets())
    exact_to_code_object = dict(amdgpu_target_map.EXACT_TARGET_CODE_OBJECTS)
    family_map = {
        family: amdgpu_target_map.family_targets(targets)
        for family, targets in amdgpu_target_map.TARGET_FAMILIES
    }

    def append_code_object(target: str):
        if target not in code_object_targets:
            code_object_targets.append(target)

    for selection in selections:
        if selection in all_code_object_targets:
            append_code_object(selection)
        elif selection in all_exact_targets:
            append_code_object(exact_to_code_object[selection])
        elif selection in family_map:
            for exact_target in family_map[selection]:
                append_code_object(exact_to_code_object[exact_target])
        else:
            available = sorted(
                all_code_object_targets | all_exact_targets | set(family_map)
            )
            raise RuntimeError(
                f"unknown AMDGPU target selector '{selection}'. "
                f"Available selectors include: {', '.join(available)}"
            )
    return code_object_targets


def default_target_selections() -> tuple[str, ...]:
    env_value = os.environ.get(DEVICE_BINARY_TARGET_ENV)
    if env_value:
        return tuple(split_env_list(env_value))
    return DEFAULT_TARGET_SELECTIONS


def run_command(command: Sequence[str], *, verbose: bool, dry_run: bool):
    if verbose or dry_run:
        print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def compile_source(
    *,
    source_path: Path,
    output_path: Path,
    arch: str,
    repo_root: Path,
    binary_root: Path | None,
    toolchain: Toolchain,
    extra_copts: Sequence[str],
    verbose: bool,
    dry_run: bool,
):
    include_args = [
        "-isystem",
        str(toolchain.clang_resource_include),
        "-I",
        str(repo_root / "runtime/src"),
    ]
    if binary_root:
        include_args.extend(["-I", str(binary_root / "runtime/src")])

    command = [
        str(toolchain.clang),
        "-x",
        "c",
        "-std=c23",
        "-Xclang",
        "-finclude-default-header",
        "-nogpulib",
        "-fno-short-wchar",
        "-target",
        TARGET_TRIPLE,
        f"-march={arch}",
        "-fgpu-rdc",
        *include_args,
        "-Wno-gnu-pointer-arith",
        "-fno-ident",
        "-fvisibility=hidden",
        "-O3",
        *extra_copts,
        "-c",
        "-emit-llvm",
        "-o",
        str(output_path),
        str(source_path),
    ]
    run_command(command, verbose=verbose, dry_run=dry_run)


def link_bitcode(
    *,
    input_paths: Sequence[Path],
    output_path: Path,
    toolchain: Toolchain,
    verbose: bool,
    dry_run: bool,
):
    command = [str(toolchain.llvm_link), *map(str, input_paths), "-o", str(output_path)]
    run_command(command, verbose=verbose, dry_run=dry_run)


def internalize_bitcode(
    *,
    input_path: Path,
    output_path: Path,
    toolchain: Toolchain,
    verbose: bool,
    dry_run: bool,
):
    command = [
        str(toolchain.llvm_link),
        "-internalize",
        "-only-needed",
        str(input_path),
        "-o",
        str(output_path),
    ]
    run_command(command, verbose=verbose, dry_run=dry_run)


def write_local_all_version_script(path: Path):
    path.write_text("{\n  local:\n    *;\n};\n")


def link_code_object(
    *,
    input_path: Path,
    output_path: Path,
    arch: str,
    version_script: Path | None,
    toolchain: Toolchain,
    linkopts: Sequence[str],
    verbose: bool,
    dry_run: bool,
):
    command = [
        str(toolchain.lld),
        "-flavor",
        "gnu",
        "-m",
        "elf64_amdgpu",
        "--build-id=none",
        "--no-undefined",
        "-shared",
        f"-plugin-opt=mcpu={arch}",
        "-plugin-opt=O3",
        "--lto-CGO3",
        "--no-whole-archive",
        "--gc-sections",
        "--strip-debug",
        "--discard-all",
        "--discard-locals",
        *linkopts,
    ]
    if version_script:
        command.append(f"--version-script={version_script}")
    command.extend([str(input_path), "-o", str(output_path)])
    run_command(command, verbose=verbose, dry_run=dry_run)


def minimize_code_object(
    *,
    input_path: Path,
    output_path: Path,
    toolchain: Toolchain,
    verbose: bool,
    dry_run: bool,
):
    command = [
        str(toolchain.llvm_objcopy),
        "-R",
        ".comment",
        "-R",
        ".AMDGPU.gpr_maximums",
        "--discard-all",
        "-N",
        "_DYNAMIC",
    ]
    command.extend([str(input_path), str(output_path)])
    run_command(command, verbose=verbose, dry_run=dry_run)


def build_target(
    *,
    arch: str,
    source_paths: Sequence[Path],
    repo_root: Path,
    binary_root: Path | None,
    output_dir: Path,
    toolchain: Toolchain,
    minimize: bool,
    keep_intermediates: bool,
    extra_copts: Sequence[str],
    linkopts: Sequence[str],
    verbose: bool,
    dry_run: bool,
) -> dict:
    output_name = f"{TARGET_TRIPLE}--{arch}.so"
    output_path = output_dir / output_name
    if dry_run:
        work_dir = output_dir / f"{output_name}.work"
        work_dir.mkdir(parents=True, exist_ok=True)
        temp_dir_context = None
    elif keep_intermediates:
        work_dir = output_dir / f"{output_name}.work"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        temp_dir_context = None
    else:
        temp_dir_context = tempfile.TemporaryDirectory(
            prefix=f"{output_name}.", dir=output_dir
        )
        work_dir = Path(temp_dir_context.name)

    try:
        object_paths = []
        for index, source_path in enumerate(source_paths):
            object_path = work_dir / f"{index}_{source_path.stem}.bc"
            compile_source(
                source_path=source_path,
                output_path=object_path,
                arch=arch,
                repo_root=repo_root,
                binary_root=binary_root,
                toolchain=toolchain,
                extra_copts=extra_copts,
                verbose=verbose,
                dry_run=dry_run,
            )
            object_paths.append(object_path)

        archive_path = work_dir / f"{arch}.archive.bc"
        link_bitcode(
            input_paths=object_paths,
            output_path=archive_path,
            toolchain=toolchain,
            verbose=verbose,
            dry_run=dry_run,
        )

        linked_bitcode_path = work_dir / f"{arch}.linked.bc"
        internalize_bitcode(
            input_path=archive_path,
            output_path=linked_bitcode_path,
            toolchain=toolchain,
            verbose=verbose,
            dry_run=dry_run,
        )

        version_script = None
        linked_code_object_path = output_path
        if minimize:
            version_script = work_dir / "local-all.version"
            if not dry_run:
                write_local_all_version_script(version_script)
            linked_code_object_path = work_dir / output_name

        link_code_object(
            input_path=linked_bitcode_path,
            output_path=linked_code_object_path,
            arch=arch,
            version_script=version_script,
            toolchain=toolchain,
            linkopts=linkopts,
            verbose=verbose,
            dry_run=dry_run,
        )

        if minimize:
            minimize_code_object(
                input_path=linked_code_object_path,
                output_path=output_path,
                toolchain=toolchain,
                verbose=verbose,
                dry_run=dry_run,
            )

        if output_path.exists():
            output_path.chmod(0o644)
        size = output_path.stat().st_size if output_path.exists() else 0
        return {
            "target": arch,
            "path": output_name,
            "size": size,
        }
    finally:
        if temp_dir_context is not None:
            temp_dir_context.cleanup()


def supported_amdgpu_processors(clang: Path) -> set[str]:
    try:
        output = run_capture(
            [str(clang), "-target", TARGET_TRIPLE, "--print-supported-cpus"]
        )
    except subprocess.CalledProcessError:
        return set()
    processors = set()
    for line in output.splitlines():
        candidate = line.strip().split(" ", 1)[0]
        if candidate.startswith("gfx") or candidate in ("generic", "generic-hsa"):
            processors.add(candidate)
    return processors


def validate_toolchain_targets(
    toolchain: Toolchain, code_object_targets: Sequence[str]
):
    supported_processors = supported_amdgpu_processors(toolchain.clang)
    if not supported_processors:
        return
    unsupported_targets = [
        target for target in code_object_targets if target not in supported_processors
    ]
    if unsupported_targets:
        raise RuntimeError(
            "selected AMDGPU code-object target(s) are not reported by clang: "
            f"{', '.join(unsupported_targets)}. Point --rocm-path/--tool-dir at "
            "a newer LLVM/ROCm toolchain or choose a smaller --targets set."
        )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build precompiled AMDGPU HAL builtin device binaries."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root_from_script(),
        help="IREE repository root containing runtime/src. Defaults to auto-detection.",
    )
    default_binary_root = (
        Path(os.environ["IREE_BINARY_DIR"])
        if os.environ.get("IREE_BINARY_DIR")
        else None
    )
    parser.add_argument(
        "--binary-root",
        type=Path,
        default=default_binary_root,
        help="Optional build tree root to add as a generated include root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to receive generated code objects.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(default_target_selections()),
        help=(
            "Comma/space separated exact targets, code-object targets, or families. "
            f"Defaults to ${DEVICE_BINARY_TARGET_ENV} or "
            f"{','.join(DEFAULT_TARGET_SELECTIONS)}."
        ),
    )
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help="Build every known code-object target.",
    )
    parser.add_argument(
        "--tool-dir",
        action="append",
        help=(
            "Directory containing clang, llvm-link, lld, and llvm-objcopy. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--rocm-path",
        action="append",
        help=(
            "ROCm installation root. May be repeated. If omitted, the script "
            "checks IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH, IREE_ROCM_PATH, "
            "ROCM_PATH, ROCM_ROOT, ROCM_HOME, HIP_PATH, hipcc on PATH, and "
            "/opt/rocm."
        ),
    )
    parser.add_argument("--clang", help="Explicit clang/amdclang executable.")
    parser.add_argument("--llvm-link", help="Explicit llvm-link executable.")
    parser.add_argument("--lld", help="Explicit lld or ld.lld executable.")
    parser.add_argument("--llvm-objcopy", help="Explicit llvm-objcopy executable.")
    parser.add_argument(
        "--clang-resource-include",
        help=(
            "Explicit clang resource include directory. Defaults to "
            "IREE_CLANG_BUILTIN_HEADERS_PATH or clang -print-resource-dir/include."
        ),
    )
    parser.add_argument(
        "--minimize",
        dest="minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep only the exported .kd symbol surface in regular symbol tables "
            "after linking. Enabled by default."
        ),
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep per-target bitcode and intermediate code objects next to outputs.",
    )
    parser.add_argument(
        "--copt",
        action="append",
        default=[],
        help="Additional clang compile option. May be repeated.",
    )
    parser.add_argument(
        "--linkopt",
        action="append",
        default=[],
        help="Additional lld link option. May be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print commands as they execute.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = (
        amdgpu_target_map.code_object_targets()
        if args.all_targets
        else split_env_list(args.targets)
    )
    code_object_targets = expand_target_selections(selections)
    source_paths = load_device_source_paths(repo_root)
    toolchain = detect_toolchain(args)
    validate_toolchain_targets(toolchain, code_object_targets)

    eprint("AMDGPU device binary generator")
    eprint(f"  repo root: {repo_root}")
    eprint(f"  output dir: {output_dir}")
    eprint(f"  targets: {', '.join(code_object_targets)}")
    eprint(
        "  sources: "
        + ", ".join(str(path.relative_to(repo_root)) for path in source_paths)
    )
    eprint(f"  clang: {toolchain.clang}")
    eprint(f"  llvm-link: {toolchain.llvm_link}")
    eprint(f"  lld: {toolchain.lld}")
    eprint(f"  llvm-objcopy: {toolchain.llvm_objcopy}")
    eprint(f"  clang resource include: {toolchain.clang_resource_include}")

    outputs = []
    for arch in code_object_targets:
        eprint(f"Building {arch}...")
        outputs.append(
            build_target(
                arch=arch,
                source_paths=source_paths,
                repo_root=repo_root,
                binary_root=args.binary_root.resolve() if args.binary_root else None,
                output_dir=output_dir,
                toolchain=toolchain,
                minimize=args.minimize,
                keep_intermediates=args.keep_intermediates,
                extra_copts=args.copt,
                linkopts=args.linkopt,
                verbose=args.verbose,
                dry_run=args.dry_run,
            )
        )

    if not args.dry_run:
        total_size = sum(output["size"] for output in outputs)
        for output in outputs:
            eprint(f"  wrote {output['path']}: {output['size']} bytes")
        eprint(f"Total code-object size: {total_size} bytes")
    return 0


def run_cli(argv: Sequence[str]) -> int:
    try:
        return main(argv)
    except RuntimeError as exc:
        eprint(f"error: {exc}")
        return 1
    except subprocess.CalledProcessError as exc:
        if exc.returncode < 0:
            detail = f"signal {-exc.returncode}"
        else:
            detail = f"exit code {exc.returncode}"
        eprint(f"error: command failed with {detail}: {' '.join(exc.cmd)}")
        return 1


if __name__ == "__main__":
    sys.exit(run_cli(sys.argv[1:]))
