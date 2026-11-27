# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for iree.importers.torch.pt2."""

import tempfile
import unittest
from pathlib import Path

import torch

from iree.importers.torch import register_import_decomposition
from iree.importers.torch.import_decompositions import _CUSTOM_DECOMPOSITIONS
from iree.importers.torch.params import ParameterArchive
from iree.importers.torch.pt2 import (
    externalize_program_parameters,
    import_programs,
    load_pt2,
    share_program_tensors,
)
from iree.importers.torch.support.tensor_traits import ExternalTensorTrait


class Prefill(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(64, 64)
        self.register_buffer("cache", torch.zeros(1, 64))

    def forward(self, x):
        y = self.projection(x)
        self.cache.copy_(y)
        return y


class Decode(Prefill):
    def forward(self, x):
        return self.projection(x) + self.cache


def _export(module: torch.nn.Module) -> torch.export.ExportedProgram:
    return torch.export.export(module, (torch.zeros(1, 64),))


def _save_pair(directory: Path) -> tuple[Path, Path]:
    """Save a prefill and a decode program that have the same state."""
    prefill = Prefill()
    decode = Decode()
    decode.load_state_dict(prefill.state_dict())
    prefill_path = directory / "prefill.pt2"
    decode_path = directory / "decode.pt2"
    torch.export.save(_export(prefill), prefill_path)
    torch.export.save(_export(decode), decode_path)
    return prefill_path, decode_path


class LoadPt2Test(unittest.TestCase):
    def test_loads_a_saved_program(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.pt2"
            torch.export.save(_export(Prefill()), path)

            program = load_pt2(path)

            self.assertEqual(
                sorted(program.state_dict),
                ["cache", "projection.bias", "projection.weight"],
            )


class ShareProgramTensorsTest(unittest.TestCase):
    def test_shares_equal_tensors_of_the_same_name(self):
        with tempfile.TemporaryDirectory() as td:
            prefill_path, decode_path = _save_pair(Path(td))
            prefill = load_pt2(prefill_path)
            decode = load_pt2(decode_path)

            replaced = share_program_tensors({"prefill": prefill, "decode": decode})

            self.assertEqual(replaced, 3)
            for name in ("cache", "projection.bias", "projection.weight"):
                self.assertIs(decode.state_dict[name], prefill.state_dict[name])

    def test_keeps_tensors_of_the_same_name_that_differ(self):
        prefill = _export(Prefill())
        decode = _export(Decode())
        decode_weight = decode.state_dict["projection.weight"]

        replaced = share_program_tensors({"prefill": prefill, "decode": decode})

        # Only the zero-initialized cache is equal in the two modules.
        self.assertEqual(replaced, 1)
        self.assertIs(decode.state_dict["projection.weight"], decode_weight)

    def test_does_not_merge_equal_tensors_within_a_program(self):
        class TwoBuffers(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.zeros(1, 64))
                self.register_buffer("b", torch.zeros(1, 64))

            def forward(self, x):
                return x + self.a + self.b

        program = _export(TwoBuffers())

        replaced = share_program_tensors({"main": program})

        self.assertEqual(replaced, 0)
        self.assertIsNot(program.state_dict["a"], program.state_dict["b"])


class ExternalizeProgramParametersTest(unittest.TestCase):
    def test_marks_tensors_with_their_name(self):
        program = _export(Prefill())

        marked = externalize_program_parameters({"main": program})

        self.assertEqual(
            sorted(marked), ["cache", "projection.bias", "projection.weight"]
        )
        trait = ExternalTensorTrait.get(program.state_dict["projection.weight"])
        self.assertEqual(trait.external_name, "projection.weight")

    def test_prefixes_the_program_name_when_names_collide(self):
        prefill = _export(Prefill())
        decode = _export(Decode())

        marked = externalize_program_parameters({"prefill": prefill, "decode": decode})

        self.assertEqual(
            sorted(marked),
            [
                "cache",
                "decode.cache",
                "decode.projection.bias",
                "decode.projection.weight",
                "projection.bias",
                "projection.weight",
            ],
        )


class RegisterImportDecompositionTest(unittest.TestCase):
    def test_later_imports_use_the_registered_decomposition(self):
        class Relu(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        self.addCleanup(_CUSTOM_DECOMPOSITIONS.pop, torch.ops.aten.relu.default, None)

        register_import_decomposition(
            torch.ops.aten.relu.default, lambda x: torch.clamp(x, min=0.0)
        )
        output = import_programs(
            {"main": torch.export.export(Relu(), (torch.zeros(1, 64),))}
        )

        text = str(output.mlir_module)
        self.assertIn("torch.aten.clamp", text)
        self.assertNotIn("torch.aten.relu", text)


class ImportProgramsTest(unittest.TestCase):
    def test_rejects_no_programs(self):
        with self.assertRaises(ValueError):
            import_programs({})

    def test_rejects_saving_parameters_that_stay_in_the_module(self):
        with self.assertRaises(ValueError):
            import_programs(
                {"main": _export(Prefill())}, save_parameters_to="weights.irpa"
            )

    def test_runs_the_given_decompositions(self):
        class Relu(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        output = import_programs(
            {"main": torch.export.export(Relu(), (torch.zeros(1, 64),))},
            decompositions={
                torch.ops.aten.relu.default: lambda x: torch.clamp(x, min=0.0)
            },
        )

        text = str(output.mlir_module)
        self.assertIn("torch.aten.clamp", text)
        self.assertNotIn("torch.aten.relu", text)

    def test_given_decompositions_take_precedence_over_the_import_ones(self):
        class Hardtanh(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.hardtanh(x)

        def hardtanh_as_relus(x, min_val=-1.0, max_val=1.0):
            return max_val - torch.relu(max_val - min_val - torch.relu(x - min_val))

        output = import_programs(
            {"main": torch.export.export(Hardtanh(), (torch.zeros(1, 64),))},
            decompositions={torch.ops.aten.hardtanh.default: hardtanh_as_relus},
        )

        text = str(output.mlir_module)
        self.assertIn("torch.aten.relu", text)
        self.assertNotIn("torch.aten.clamp", text)
        self.assertNotIn("torch.aten.hardtanh", text)

    def test_runs_only_the_given_decompositions_without_the_import_ones(self):
        class LinearRelu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 8)

            def forward(self, x):
                return torch.relu(self.linear(x))

        output = import_programs(
            {"main": torch.export.export(LinearRelu(), (torch.zeros(1, 64),))},
            decompose=False,
            decompositions={
                torch.ops.aten.relu.default: lambda x: torch.clamp(x, min=0.0)
            },
        )

        text = str(output.mlir_module)
        self.assertIn("torch.aten.linear", text)
        self.assertIn("torch.aten.clamp", text)
        self.assertNotIn("torch.aten.relu", text)

    def test_embeds_parameters_by_default(self):
        output = import_programs({"main": _export(Prefill())})

        text = str(output.mlir_module)
        self.assertIn("func.func @main(", text)
        self.assertNotIn("#flow.parameter.named", text)

    def test_imports_programs_from_files_as_functions_with_common_state(self):
        with tempfile.TemporaryDirectory() as td:
            prefill_path, decode_path = _save_pair(Path(td))
            archive_path = Path(td) / "weights.irpa"

            output = import_programs(
                {"prefill": load_pt2(prefill_path), "decode": load_pt2(decode_path)},
                externalize_parameters=True,
                save_parameters_to=archive_path,
            )

            text = str(output.mlir_module)
            self.assertIn("func.func @prefill(", text)
            self.assertIn("func.func @decode(", text)
            self.assertEqual(text.count("util.global private"), 3)
            self.assertIn(
                'util.global private mutable @__auto.cache = #flow.parameter.named<"model"::"cache">',
                text,
            )
            self.assertIn("util.global.store", text)
            archive = ParameterArchive(archive_path, mmap=False)
            self.assertEqual(
                sorted(name for name, _ in archive.items()),
                ["cache", "projection.bias", "projection.weight"],
            )

    def test_keeps_separate_state_when_tensors_are_not_shared(self):
        with tempfile.TemporaryDirectory() as td:
            prefill_path, decode_path = _save_pair(Path(td))

            output = import_programs(
                {"prefill": load_pt2(prefill_path), "decode": load_pt2(decode_path)},
                share_tensors=False,
                externalize_parameters=True,
            )

            self.assertEqual(str(output.mlir_module).count("util.global private"), 6)

    def test_lowers_to_iree_input(self):
        output = import_programs({"main": _export(Prefill())})

        output.lower_to_iree()

        text = str(output.mlir_module)
        self.assertIn("util.func public @main$async(", text)
        self.assertIn("linalg.matmul", text)
        self.assertNotIn("torch.aten", text)

    def test_lowers_without_async_entry_points(self):
        output = import_programs({"main": _export(Prefill())})

        output.lower_to_iree(emit_async_entry_points=False)

        text = str(output.mlir_module)
        self.assertIn("@main(", text)
        self.assertNotIn("$async", text)


if __name__ == "__main__":
    unittest.main()
