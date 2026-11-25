# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the iree-import-torch console tool."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch

from iree.importers.torch.__main__ import main as run_import
from iree.importers.torch.__main__ import parse_arguments


class Add(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.ones(64, 64))

    def forward(self, x):
        return x + self.offset


def main(argv: list[str]) -> int:
    return run_import(parse_arguments(argv))


def _save(path: Path) -> Path:
    torch.export.save(torch.export.export(Add(), (torch.zeros(64, 64),)), path)
    return path


class MainTest(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.dir = Path(temp_dir.name)
        self.model = _save(self.dir / "model.pt2")

    def test_writes_iree_input_for_an_unnamed_program(self):
        output = self.dir / "model.mlir"

        self.assertEqual(main([str(self.model), "-o", str(output)]), 0)

        text = output.read_text()
        self.assertIn("util.func public @main$async(", text)
        self.assertNotIn("torch.aten", text)

    def test_writes_the_torch_dialect(self):
        output = self.dir / "model.mlir"

        main([str(self.model), "-o", str(output), "--output-type=torch"])

        self.assertIn("torch.aten.add.Tensor", output.read_text())

    def test_names_functions_after_the_inputs(self):
        output = self.dir / "model.mlir"

        main(
            [
                f"first={self.model}",
                f"second={self.model}",
                "-o",
                str(output),
                "--output-type=torch",
            ]
        )

        text = output.read_text()
        self.assertIn("func.func @first(", text)
        self.assertIn("func.func @second(", text)

    def test_writes_bytecode_for_mlirbc(self):
        output = self.dir / "model.mlirbc"

        main([str(self.model), "-o", str(output)])

        self.assertEqual(output.read_bytes()[:4], b"ML\xefR")

    def test_writes_the_parameter_archive(self):
        output = self.dir / "model.mlir"
        archive = self.dir / "model.irpa"

        main(
            [
                str(self.model),
                "-o",
                str(output),
                "--externalize-params",
                "--save-params-to",
                str(archive),
            ]
        )

        self.assertIn('#flow.parameter.named<"model"::"offset">', output.read_text())
        self.assertGreater(archive.stat().st_size, 64 * 64 * 4)

    def test_rejects_a_function_name_used_twice(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main([str(self.model), str(self.model), "-o", str(self.dir / "o.mlir")])

    def test_rejects_an_archive_without_externalized_parameters(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(
                [
                    str(self.model),
                    "-o",
                    str(self.dir / "o.mlir"),
                    "--save-params-to",
                    str(self.dir / "o.irpa"),
                ]
            )


if __name__ == "__main__":
    unittest.main()
