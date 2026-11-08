# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Prepare reference files and build/run the SmolLM2 WASI deployment."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote, urlparse

import numpy as np

_COMMON_DEPS = [
    "@//runtime/src:runtime_defines",
    "@//runtime/src/iree/async",
    "@//runtime/src/iree/async/util:proactor_pool",
    "@//runtime/src/iree/base",
    "@//runtime/src/iree/base/tooling:flags",
    "@//runtime/src/iree/hal",
    "@//runtime/src/iree/hal/drivers/local_sync:sync_driver",
    "@//runtime/src/iree/io:file_handle",
    "@//runtime/src/iree/modules/hal",
    "@//runtime/src/iree/modules/hal:types",
    "@//runtime/src/iree/tooling:function_io",
    "@//runtime/src/iree/tooling:parameter_util",
    "@//runtime/src/iree/vm",
    "@//runtime/src/iree/vm/bytecode:module",
]


def write_reference_case(directory, metadata, reference, *, full_prefix_checked):
    """Serialise a sequence and its producer's reference-validation evidence."""
    shapes = metadata["state_shapes"]
    if not 1 <= len(shapes) <= 64 or not reference:
        raise ValueError("WASI requires 1-64 cache tensors and a nonempty sequence")
    for step in reference:
        if len(step["state"]) != len(shapes):
            raise ValueError("reference cache count differs from the export")
        for value, shape in zip(step["state"], shapes, strict=True):
            if value.dtype != np.float32 or list(value.shape) != shape:
                raise ValueError("reference cache type/shape differs from the export")
            if not np.isfinite(value).all():
                raise ValueError("reference cache must be finite")
        logits = step["logits"]
        if (
            logits.dtype != np.float32
            or logits.ndim != 3
            or logits.shape[0] != 1
            or min(logits.shape[1:]) < 1
            or not np.isfinite(logits).all()
        ):
            raise ValueError("reference logits must be finite batch-one f32 arrays")
        if (
            step["tokens"].dtype != np.int64
            or step["tokens"].ndim != 2
            or step["tokens"].shape[0] != 1
            or step["tokens"].shape[1] != logits.shape[1]
            or not 1 <= step["length"] <= metadata["capacity"]
        ):
            raise ValueError("invalid token/position reference inputs")

    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "case.json"
    manifest.unlink(missing_ok=True)

    def save_arrays(name, values):
        with (directory / name).open("wb") as stream:
            for value in values:
                # WebAssembly is little-endian, independently of the host.
                np.save(
                    stream,
                    value.astype(value.dtype.newbyteorder("<"), copy=False),
                    allow_pickle=False,
                )

    save_arrays("initial.npy", [np.zeros(shape, np.float32) for shape in shapes])
    for i, step in enumerate(reference):
        save_arrays(
            f"step{i}-inputs.npy",
            [step["tokens"], np.arange(step["length"], dtype=np.int64)],
        )
        save_arrays(f"step{i}-expected.npy", [*step["state"], step["logits"]])
    manifest.write_text(
        json.dumps(
            {
                "export": metadata,
                "state_count": len(shapes),
                "steps": len(reference),
                "retained_state_checked": not metadata["state_donated"],
                "reference_checked_against_full_prefix": full_prefix_checked,
                "atol": 1e-4,
                "rtol": 1e-4,
            },
            indent=2,
        )
        + "\n"
    )


def prepare_module(module_text, mode, phase):
    """Give each static executable a unique namespace across VM contexts."""
    from iree.compiler import ir

    with ir.Context():
        module = ir.Module.parse(module_text)
        module.operation.attributes["sym_name"] = ir.StringAttr.get(
            f"smollm2_{mode}_{phase}"
        )
        return str(module)


def compiler_arguments(args, mode, phase):
    flags = (
        ["--iree-llvmcpu-target-cpu=generic"]
        if args.target_backend == "llvm-cpu"
        else []
    ) + args.compiler_arg
    if not args.wasi or args.target_backend != "llvm-cpu":
        return flags
    required = {
        "--iree-llvmcpu-target-triple": "wasm32-unknown-unknown",
        "--iree-llvmcpu-link-embedded": "false",
        "--iree-llvmcpu-link-static": "true",
        "--iree-llvmcpu-static-library-output-path": str(
            args.output_dir / "wasi_model" / f"{mode}-{phase}.o"
        ),
        "--iree-vm-target-index-bits": "32",
    }
    if any(flag.partition("=")[0] in required for flag in args.compiler_arg):
        raise ValueError(
            "--wasi controls the target triple, link mode and VM index width"
        )
    return [*flags, *(f"{key}={value}" for key, value in required.items())]


def write_model_repository(directory, backend):
    """Declare generated objects in an external Bazel repository."""
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        Path(__file__).with_name("wasi") / "smollm2_wasi.c",
        directory / "smollm2_wasi.c",
    )
    sources = ["smollm2_wasi.c"]
    defines = []
    deps = list(_COMMON_DEPS)
    if backend == "llvm-cpu":
        includes, queries = [], []
        for mode in ("specialized", "fallback"):
            for phase in ("prefill", "decode"):
                stem = f"{mode}-{phase}"
                header = directory / f"{stem}.h"
                queries_in_header = re.findall(
                    r"\b([A-Za-z_]\w*_library_query)\s*\(", header.read_text()
                )
                if len(set(queries_in_header)) != 1:
                    raise ValueError(f"expected one executable query in {header}")
                query = queries_in_header[0]
                if query in queries:
                    raise ValueError(
                        "model static libraries must have unique query names"
                    )
                queries.append(query)
                includes.append(f'#include "{stem}.h"')
                sources.extend([f"{stem}.o", f"{stem}.h"])
        (directory / "static_libraries.h").write_text(
            "\n".join(includes)
            + "\nstatic const iree_hal_executable_library_query_fn_t "
            "iree_torchax_model_libraries[] = {\n" + ",\n".join(queries) + "\n};\n"
        )
        sources.append("static_libraries.h")
        defines.append("IREE_TORCHAX_STATIC_MODEL=1")
        deps.append("@//runtime/src/iree/hal/local/loaders:static_library_loader")
    elif backend == "vmvx":
        deps.append("@//runtime/src/iree/hal/local/loaders:vmvx_module_loader")
    else:
        raise ValueError("WASI supports llvm-cpu static objects and VMVX modules")
    (directory / "MODULE.bazel").write_text('module(name = "torchax_smollm2_model")\n')
    (directory / "BUILD.bazel").write_text(
        'load("@//build_tools/bazel:iree_wasm_library.bzl", "iree_wasm_cc_binary")\n'
        "iree_wasm_cc_binary(\n"
        '    name = "runner",\n'
        '    main = "@//build_tools/wasm:wasm_test_main.mjs",\n'
        f"    srcs = {json.dumps(sources)},\n"
        f"    defines = {json.dumps(defines)},\n"
        f"    deps = {json.dumps(deps)},\n"
        ")\n"
    )
    # The normal CMake runtime-plugin hook can consume the same generated
    # repository; no model objects or build configuration enter the checkout.
    cmake_sources = "\n".join(
        f'    "${{CMAKE_CURRENT_LIST_DIR}}/{source}"' for source in sources
    )
    cmake_deps = []
    for dep in deps:
        if dep == "@//runtime/src:runtime_defines":
            continue
        package, _, target = dep.removeprefix("@//runtime/src/").partition(":")
        name = package.replace("/", "::")
        if target:
            name += "::" + target
        cmake_deps.append("    " + name)
    (directory / "iree_runtime_plugin.cmake").write_text(
        'if(NOT CMAKE_SYSTEM_NAME STREQUAL "WASI")\n'
        '  message(FATAL_ERROR "The Torchax WASI runner requires the WASI toolchain")\n'
        "endif()\n"
        + (
            # Runtime plugins are included in a function/subdirectory scope.
            # The cache setting reaches the subsequent runtime target setup,
            # including toolchains that disable all default loaders first.
            "set(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE ON CACHE BOOL\n"
            '    "Required by the selected Torchax VMVX runner" FORCE)\n'
            if backend == "vmvx"
            else ""
        )
        + "iree_wasm_cc_binary(\n"
        "  PACKAGE iree::torchax\n"
        "  NAME smollm2_wasi_runner\n"
        '  MAIN "${IREE_SOURCE_DIR}/build_tools/wasm/wasm_test_main.mjs"\n'
        "  SRCS\n" + cmake_sources + "\n"
        "  DEFINES\n    " + " ".join(defines) + "\n"
        "  DEPS\n" + "\n".join(cmake_deps) + "\n)\n"
    )


def find_bundle(build_events):
    bundles = set()
    for line in build_events.read_text().splitlines():
        event = json.loads(line)
        for file in event.get("namedSetOfFiles", {}).get("files", []):
            uri = urlparse(file.get("uri", ""))
            if uri.scheme == "file" and uri.path.endswith("/runner_bundle.mjs"):
                bundles.add(Path(unquote(uri.path)))
    if len(bundles) != 1:
        raise ValueError("Bazel must produce exactly one local runner bundle")
    bundle = bundles.pop()
    if not bundle.is_file():
        raise ValueError(f"Bazel runner output is missing: {bundle}")
    return bundle


def build_runner(args):
    directory = args.output_dir / "wasi_model"
    root = Path(__file__).resolve().parents[3]
    events = directory / "build_events.jsonl"
    command = [
        args.bazel,
        *args.bazel_startup_arg,
        "build",
        "--jobs=4",
        *args.bazel_build_arg,
        "--config=wasm32_wasi",
        "--copt=-DIREE_FILE_IO_ENABLE=1",
        "--remote_download_outputs=all",
        f"--inject_repository=torchax_smollm2_model={directory}",
        f"--build_event_json_file={events}",
        "@torchax_smollm2_model//:runner_bundle",
        "@torchax_smollm2_model//:runner_wasm",
    ]
    subprocess.run(command, cwd=root, check=True)
    built_bundle = find_bundle(events)
    built_wasm = built_bundle.with_name("runner_wasm")
    if not built_wasm.is_file():
        raise ValueError(f"Bazel WASM output is missing: {built_wasm}")
    # Preserve the matched pair before another build can replace Bazel outputs.
    # The bundle refers to runner_wasm beside itself, as declared by our macro.
    bundle = directory / "runner_bundle.mjs"
    shutil.copyfile(built_bundle, bundle)
    shutil.copyfile(built_wasm, directory / "runner_wasm")
    with bundle.open("rb") as stream:
        bundle_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with (directory / "runner_wasm").open("rb") as stream:
        wasm_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return bundle, {
        "command": command,
        "runner_bundle": str(bundle),
        "runner_bundle_sha256": bundle_hash,
        "runner_wasm_sha256": wasm_hash,
        "node_version": subprocess.check_output(
            [args.node, "--version"], text=True
        ).strip(),
    }


def runner_command(args, metadata, bundle, mode, steps):
    return [
        args.node,
        str(bundle),
        f"--prefill={args.output_dir / (mode + '-prefill.vmfb')}",
        f"--decode={args.output_dir / (mode + '-decode.vmfb')}",
        f"--parameters=model={args.output_dir / 'weights.irpa'}",
        "--parameter_mode=preload",
        f"--case_directory={args.output_dir / 'wasi_case'}",
        f"--state_count={len(metadata['state_shapes'])}",
        f"--steps={steps}",
        f"--retain_state={'false' if metadata['state_donated'] else 'true'}",
    ]


def run_programs(args, metadata, reference, bundle, mode):
    command = runner_command(args, metadata, bundle, mode, len(reference))
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        env={**os.environ, "TEST_TMPDIR": str(args.output_dir)},
    )
    records = [json.loads(line) for line in completed.stdout.splitlines() if line]
    if len(records) != len(reference):
        raise ValueError("WASI runner did not check every invocation")
    results = []
    for i, (record, expected) in enumerate(zip(records, reference, strict=True)):
        if (
            record["step"] != i
            or record["next_token"] != int(expected["logits"][0, -1].argmax())
            or record["retained_state_checked"] != args.retain_state
        ):
            raise ValueError("WASI runner returned inconsistent sequence results")
        record.update(
            stage="prefill" if i == 0 else "decode",
            prefix_length=expected["length"],
        )
        print(mode, record, flush=True)
        results.append(record)
    return results
