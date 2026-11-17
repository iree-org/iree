# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise the actual WASI checker with small stateful VMVX/LLVM modules."""

import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from iree.compiler import tools
import iree.runtime as runtime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import smollm2_wasm as wasm


_MODULE = """
module {
  func.func @main(%a: tensor<1xf32>, %b: tensor<1xf32>,
                  %tokens: tensor<1x?xi64>, %positions: tensor<?xi64>)
      -> (tensor<1xf32>, tensor<1xf32>, tensor<1x1x4xf32>) {
    %one = arith.constant dense<1.0> : tensor<1xf32>
    %two = arith.constant dense<2.0> : tensor<1xf32>
    %next_a = arith.addf %a, %one : tensor<1xf32>
    %next_b = arith.addf %b, %two : tensor<1xf32>
    %logits = arith.constant dense<[[[1.0, 1.00005, 0.0, -1.0]]]>
        : tensor<1x1x4xf32>
    return %next_a, %next_b, %logits
        : tensor<1xf32>, tensor<1xf32>, tensor<1x1x4xf32>
  }
}
"""


def make_reference():
    return [
        {
            "tokens": np.array([[7]], np.int64),
            "length": i + 2,
            "state": [
                np.array([i + 1], np.float32),
                np.array([2 * (i + 1)], np.float32),
            ],
            "logits": np.array([[[1, 1.00005, 0, -1]]], np.float32),
        }
        for i in range(3)
    ]


def check_failure(args, metadata, reference, bundle, expected_error):
    result = subprocess.run(
        wasm.runner_command(args, metadata, bundle, "specialized", len(reference)),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "TEST_TMPDIR": str(args.output_dir)},
    )
    if result.returncode == 0 or expected_error not in result.stderr:
        raise AssertionError(
            f"expected a failing WASI process reporting {expected_error!r}; "
            f"exit={result.returncode}\n{result.stdout}\n{result.stderr}"
        )


def test_backend(args):
    args.output_dir.mkdir(parents=True)
    model = args.output_dir / "wasi_model"
    model.mkdir()
    runtime.ParameterIndex().create_archive_file(str(args.output_dir / "weights.irpa"))
    for mode in ("specialized", "fallback"):
        for phase in ("prefill", "decode"):
            module = wasm.prepare_module(_MODULE, mode, phase)
            binary = tools.compile_str(
                module,
                input_type="none",
                target_backends=[args.target_backend],
                extra_args=wasm.compiler_arguments(args, mode, phase),
            )
            (args.output_dir / f"{mode}-{phase}.vmfb").write_bytes(binary)
    wasm.write_model_repository(model, args.target_backend)
    bundle, _ = wasm.build_runner(args)
    metadata = {"state_shapes": [[1], [1]], "state_donated": False, "capacity": 8}
    reference = make_reference()
    for retained in (True, False):
        args.retain_state = retained
        metadata["state_donated"] = not retained
        wasm.write_reference_case(
            args.output_dir / "wasi_case",
            metadata,
            reference,
            full_prefix_checked=False,
        )
        for mode in ("specialized", "fallback"):
            wasm.run_programs(args, metadata, reference, bundle, mode)

    bad_cache = copy.deepcopy(reference)
    bad_cache[1]["state"][1][0] += 100
    wasm.write_reference_case(
        args.output_dir / "wasi_case", metadata, bad_cache, full_prefix_checked=False
    )
    check_failure(args, metadata, reference, bundle, "step 1 output 1")

    # Every element still meets the numerical tolerance, but the expected
    # final-row argmax changes. Token equality is an independent requirement.
    bad_token = copy.deepcopy(reference)
    bad_token[1]["logits"][0, -1, 0] += 8e-5
    wasm.write_reference_case(
        args.output_dir / "wasi_case", metadata, bad_token, full_prefix_checked=False
    )
    check_failure(args, metadata, reference, bundle, "predicted token mismatch")

    wasm.write_reference_case(
        args.output_dir / "wasi_case", metadata, reference, full_prefix_checked=False
    )
    expected_file = args.output_dir / "wasi_case/step1-expected.npy"
    # Bypass preparation's finite-value validation to exercise the checker on
    # corrupted on-disk input, rather than merely retesting the Python guard.
    with expected_file.open("wb") as stream:
        for value in reference[1]["state"]:
            np.save(stream, value, allow_pickle=False)
        np.save(stream, np.full((1, 1, 4), np.nan, np.float32), allow_pickle=False)
    check_failure(args, metadata, reference, bundle, "step 1 output 2")

    wasm.write_reference_case(
        args.output_dir / "wasi_case", metadata, reference, full_prefix_checked=False
    )
    for missing in (args.output_dir / "missing.vmfb", args.output_dir / "invalid.vmfb"):
        if missing.name == "invalid.vmfb":
            missing.write_bytes(b"invalid bytecode")
        original = args.output_dir / "specialized-prefill.vmfb"
        saved = original.read_bytes()
        try:
            if missing.exists():
                original.write_bytes(missing.read_bytes())
            else:
                original.unlink()
            check_failure(
                args,
                metadata,
                reference,
                bundle,
                "INVALID_ARGUMENT" if missing.exists() else "unable to open file",
            )
        finally:
            original.write_bytes(saved)
    print(f"{args.target_backend}: state carry-forward and negative WASI checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bazel", default="bazel")
    parser.add_argument("--bazel-startup-arg", action="append", default=[])
    parser.add_argument("--bazel-build-arg", action="append", default=[])
    parser.add_argument("--node", default=os.environ.get("IREE_WASM_NODE", "node"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = (args.output_dir or Path(directory)).resolve()
        for backend in ("vmvx", "llvm-cpu"):
            test_args = argparse.Namespace(
                **vars(args),
                target_backend=backend,
                compiler_arg=[],
                wasi=True,
                retain_state=True,
            )
            test_args.output_dir = root / backend
            test_backend(test_args)


if __name__ == "__main__":
    main()
