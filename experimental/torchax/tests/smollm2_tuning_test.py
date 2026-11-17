# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import subprocess
import unittest

from iree.compiler import ir, tools
from iree.compiler.tools.binaries import find_tool


_PROFILE = Path(__file__).resolve().parents[1] / "examples/smollm2_spirv_f32.mlir"
_SERIAL_CONFIG = "tile_sizes = [[1, 32], [1, 1], [0, 0, 1]]"


def _make_projection(
    *, m="?", n=64, k=576, dtype="f32", transposed=True, buffers=False, attrs=""
):
    container = "memref" if buffers else "tensor"
    lhs = f"{container}<{m}x{k}x{dtype}>"
    rhs_shape = f"{n}x{k}" if transposed else f"{k}x{n}"
    rhs = f"{container}<{rhs_shape}x{dtype}>"
    output = f"{container}<{m}x{n}x{dtype}>"
    rhs_map = "(n, k)" if transposed else "(k, n)"
    suffix = "i" if dtype.startswith("i") else "f"
    result_type = "" if buffers else f" -> {output}"
    assignment = "" if buffers else "%result = "
    returned = "" if buffers else f" %result : {output}"
    annotations = "" if not attrs else f" attrs = {{{attrs}}}"
    return f"""
func.func @projection(%lhs: {lhs}, %rhs: {rhs}, %init: {output}){result_type} {{
  {assignment}linalg.generic {{
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> {rhs_map}>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  }} ins(%lhs, %rhs : {lhs}, {rhs}) outs(%init : {output}){annotations} {{
  ^bb0(%a: {dtype}, %b: {dtype}, %acc: {dtype}):
    %product = arith.mul{suffix} %a, %b : {dtype}
    %sum = arith.add{suffix} %acc, %product : {dtype}
    linalg.yield %sum : {dtype}
  }}{result_type}
  return{returned}
}}
"""


class SmolLM2TuningTest(unittest.TestCase):
    def _configure(self, payload):
        result = subprocess.run(
            [
                find_tool("iree-opt"),
                f"--iree-codegen-transform-dialect-library={_PROFILE}",
                "--iree-codegen-materialize-user-configs",
            ],
            input=payload,
            text=True,
            capture_output=True,
            check=True,
        )
        with ir.Context():
            module = ir.Module.parse(result.stdout)
            function = module.body.operations[0].operation
            projection = next(
                op.operation
                for op in function.regions[0].blocks[0].operations
                if op.operation.name == "linalg.generic"
            )
            return {
                "lowering": (
                    str(projection.attributes["lowering_config"])
                    if "lowering_config" in projection.attributes
                    else None
                ),
                "translation": (
                    str(function.attributes["translation_info"])
                    if "translation_info" in function.attributes
                    else None
                ),
            }

    def test_projection_widths_and_dynamic_rows(self):
        for shape in (
            {"m": "?", "n": 192, "k": 576},
            {"m": "1", "n": 576, "k": 1536},
            {"m": "5", "n": 1536, "k": 576},
            {"m": "?", "n": 49152, "k": 576},
        ):
            with self.subTest(**shape):
                config = self._configure(_make_projection(**shape))
                self.assertIn(_SERIAL_CONFIG, config["lowering"])
                self.assertIn("spirv_pipeline<BaseVectorize>", config["translation"])
                self.assertIn("workgroup_size = [32, 1, 1]", config["translation"])

    def test_unsupported_projections_are_unchanged(self):
        for shape in (
            {"n": 63},
            {"n": "?"},
            {"k": 128},
            {"k": "?"},
            {"dtype": "f16"},
            {"dtype": "f64"},
            {"dtype": "i32"},
            {"transposed": False},
            {"buffers": True},
        ):
            with self.subTest(**shape):
                config = self._configure(_make_projection(**shape))
                self.assertIsNone(config["lowering"])
                self.assertIsNone(config["translation"])

    def test_existing_lowering_config_is_preserved(self):
        original = (
            "#iree_codegen.lowering_config<tile_sizes = [[1, 16], [1, 1], [0, 0, 4]]>"
        )
        config = self._configure(
            _make_projection(attrs=f"lowering_config = {original}")
        )
        self.assertEqual(config["lowering"], original)
        self.assertIsNone(config["translation"])

    def test_batched_attention_contractions_are_unchanged(self):
        payload = _make_projection()
        payload = payload.replace("tensor<?x576xf32>", "tensor<1x?x576xf32>")
        payload = payload.replace("tensor<64x576xf32>", "tensor<1x64x576xf32>")
        payload = payload.replace("tensor<?x64xf32>", "tensor<1x?x64xf32>")
        for indices in ("m, k", "n, k", "m, n"):
            payload = payload.replace(
                f"affine_map<(m, n, k) -> ({indices})>",
                f"affine_map<(b, m, n, k) -> (b, {indices})>",
            )
        payload = payload.replace(
            '["parallel", "parallel", "reduction"]',
            '["parallel", "parallel", "parallel", "reduction"]',
        )
        config = self._configure(payload)
        self.assertIsNone(config["lowering"])
        self.assertIsNone(config["translation"])

    def test_existing_compilation_info_is_preserved(self):
        original = (
            "#iree_codegen.lowering_config<tile_sizes = [[1, 16], [1, 1], [0, 0, 4]]>"
        )
        compilation = (
            f"#iree_codegen.compilation_info<lowering_config = {original}, "
            "translation_info = #iree_codegen.translation_info<"
            "pipeline = #iree_gpu.spirv_pipeline<BaseVectorize> "
            "workgroup_size = [16, 1, 1]>>"
        )
        config = self._configure(
            _make_projection(attrs=f"compilation_info = {compilation}")
        )
        self.assertEqual(config["lowering"], original)
        self.assertIn("workgroup_size = [16, 1, 1]", config["translation"])

    def test_serial_fma_survives_configuration_checkpoint(self):
        payload = _make_projection()
        for backend, flags in (
            ("metal-spirv", ["--iree-metal-compile-to-metallib=false"]),
            ("vulkan-spirv", []),
            ("vulkan-spirv", ["--iree-vulkan-dispatch-abi=bda"]),
        ):
            with self.subTest(backend=backend, flags=flags):
                checkpoint = tools.compile_str(
                    payload,
                    target_backends=[backend],
                    extra_args=[
                        *flags,
                        f"--iree-codegen-transform-dialect-library={_PROFILE}",
                        "--compile-to=executable-configurations",
                    ],
                )
                # The checkpoint contains the selected configuration, so
                # resumption must not need the profile file or its option again.
                lowered = tools.compile_str(
                    checkpoint,
                    target_backends=[backend],
                    extra_args=[
                        *flags,
                        "--compile-from=executable-configurations",
                        "--compile-to=executable-targets",
                    ],
                ).decode()
                self.assertRegex(lowered, r"spirv.GL.Fma .* : f32")
                self.assertNotIn("spirv.GroupNonUniformFAdd", lowered)
                self.assertNotIn("spirv.Dot", lowered)


if __name__ == "__main__":
    unittest.main()
