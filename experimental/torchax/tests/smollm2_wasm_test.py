# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

_SOURCE = Path(__file__).resolve().parents[1] / "examples/smollm2_wasm.py"
_SPEC = importlib.util.spec_from_file_location("smollm2_wasm", _SOURCE)
wasm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wasm)


class SmolLM2WasmTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.metadata = {
            "state_shapes": [[2], [1]],
            "state_donated": False,
            "capacity": 8,
        }
        self.reference = [
            {
                "tokens": np.array([[7]], np.int64),
                "length": 3,
                "state": [np.array([1, 2], np.float32), np.array([3], np.float32)],
                "logits": np.array([[[0, 1, 2, 3]]], np.float32),
            }
        ]

    def test_reference_roundtrip(self):
        case = self.directory / "case"
        wasm.write_reference_case(
            case, self.metadata, self.reference, full_prefix_checked=False
        )
        with (case / "initial.npy").open("rb") as stream:
            np.testing.assert_array_equal(np.load(stream), [0, 0])
            np.testing.assert_array_equal(np.load(stream), [0])
            self.assertEqual(stream.read(), b"")
        with (case / "step0-inputs.npy").open("rb") as stream:
            np.testing.assert_array_equal(np.load(stream), [[7]])
            np.testing.assert_array_equal(np.load(stream), [0, 1, 2])
            self.assertEqual(stream.read(), b"")
        with (case / "step0-expected.npy").open("rb") as stream:
            for want in [*self.reference[0]["state"], self.reference[0]["logits"]]:
                actual = np.load(stream, allow_pickle=False)
                self.assertEqual(actual.dtype, np.dtype("<f4"))
                np.testing.assert_array_equal(actual, want)
            self.assertEqual(stream.read(), b"")
        manifest = json.loads((case / "case.json").read_text())
        self.assertEqual(manifest["state_count"], 2)
        self.assertTrue(manifest["retained_state_checked"])
        self.assertFalse(manifest["reference_checked_against_full_prefix"])

    def test_invalid_reference_does_not_write_a_case(self):
        invalid = [
            {**self.reference[0], "state": [np.ones(2, np.float32)]},
            {
                **self.reference[0],
                "state": [np.ones(2, np.float64), np.ones(1, np.float32)],
            },
            {
                **self.reference[0],
                "state": [np.ones(2, np.float32), np.array([np.nan], np.float32)],
            },
            {**self.reference[0], "logits": np.zeros((1, 0, 4), np.float32)},
            {**self.reference[0], "length": 9},
        ]
        for reference in invalid:
            with self.subTest(reference=reference):
                case = self.directory / "invalid"
                with self.assertRaises(ValueError):
                    wasm.write_reference_case(
                        case, self.metadata, [reference], full_prefix_checked=False
                    )
                self.assertFalse(case.exists())

    def test_wasi_flags_and_conflicting_configuration(self):
        args = SimpleNamespace(
            output_dir=self.directory,
            target_backend="llvm-cpu",
            wasi=True,
            compiler_arg=["--iree-llvmcpu-target-cpu-features=+simd128"],
        )
        flags = wasm.compiler_arguments(args, "fallback", "decode")
        self.assertIn("--iree-llvmcpu-target-triple=wasm32-unknown-unknown", flags)
        self.assertIn("--iree-vm-target-index-bits=32", flags)
        self.assertIn(args.compiler_arg[0], flags)
        self.assertIn(
            "--iree-llvmcpu-static-library-output-path="
            + str(self.directory / "wasi_model/fallback-decode.o"),
            flags,
        )
        args.compiler_arg = ["--iree-llvmcpu-target-triple=x86_64-none-elf"]
        with self.assertRaises(ValueError):
            wasm.compiler_arguments(args, "specialized", "prefill")
        args.target_backend = "vmvx"
        args.compiler_arg = []
        self.assertEqual(wasm.compiler_arguments(args, "specialized", "prefill"), [])

    def test_duplicate_static_queries_are_rejected(self):
        for mode in ("specialized", "fallback"):
            for phase in ("prefill", "decode"):
                (self.directory / f"{mode}-{phase}.h").write_text(
                    "void same_library_query(int version);\n"
                )
        with self.assertRaisesRegex(ValueError, "unique query"):
            wasm.write_model_repository(self.directory, "llvm-cpu")

    def test_local_bundle_uri_with_spaces(self):
        bundle = self.directory / "path with spaces/runner_bundle.mjs"
        bundle.parent.mkdir()
        bundle.touch()
        events = self.directory / "events.jsonl"
        event = {"namedSetOfFiles": {"files": [{"uri": bundle.as_uri()}]}}
        events.write_text(json.dumps(event) + "\n" + json.dumps(event) + "\n")
        self.assertEqual(wasm.find_bundle(events), bundle)
        bundle.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            wasm.find_bundle(events)
        events.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            wasm.find_bundle(events)


if __name__ == "__main__":
    unittest.main()
