# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 The StableHLO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for StableHLO Python APIs."""

# pylint: disable=wildcard-import,undefined-variable

import io
import re
from iree.compiler import ir, passmanager as pm
from iree.compiler.dialects import stablehlo

import numpy as np


def run(f):
  with ir.Context() as context:
    stablehlo.register_dialect(context)
    f()
  return f


@run
def test_channel_handle():
  attr = stablehlo.ChannelHandle.get(handle=1, type=2)
  assert attr is not None
  assert attr.handle == 1
  assert attr.channel_type == 2


@run
def test_comparison_direction_attr():
  attr = stablehlo.ComparisonDirectionAttr.get("EQ")
  assert attr is not None
  assert str(attr) == ("#stablehlo<comparison_direction EQ>")
  assert attr.value == "EQ"


@run
def test_comparison_type_attr():
  attr = stablehlo.ComparisonTypeAttr.get("FLOAT")
  assert attr is not None
  assert str(attr) == ("#stablehlo<comparison_type FLOAT>")
  assert attr.value == "FLOAT"


@run
def test_conv_dimension_numbers():
  attr = stablehlo.ConvDimensionNumbers.get(
      input_batch_dimension=0,
      input_feature_dimension=1,
      input_spatial_dimensions=[2, 3, 4],
      kernel_input_feature_dimension=0,
      kernel_output_feature_dimension=1,
      kernel_spatial_dimensions=[2, 3],
      output_batch_dimension=0,
      output_feature_dimension=1,
      output_spatial_dimensions=[2, 3])
  assert str(attr) == ("#stablehlo.conv<[b, f, 0, 1, 2]x[i, o, 0, 1]->"
                       "[b, f, 0, 1]>")
  assert attr is not None
  assert attr.input_batch_dimension == 0
  assert attr.input_feature_dimension == 1
  assert attr.input_spatial_dimensions == [2, 3, 4]
  assert attr.kernel_input_feature_dimension == 0
  assert attr.kernel_output_feature_dimension == 1
  assert attr.kernel_spatial_dimensions == [2, 3]
  assert attr.output_batch_dimension == 0
  assert attr.output_feature_dimension == 1
  assert attr.output_spatial_dimensions == [2, 3]


@run
def test_dot_algorithm():
  # BF16_BF16_F32_X3
  attr = stablehlo.DotAlgorithm.get(
      lhs_precision_type=ir.BF16Type.get(),
      rhs_precision_type=ir.BF16Type.get(),
      accumulation_type=ir.F32Type.get(),
      lhs_component_count=1,
      rhs_component_count=1,
      num_primitive_operations=3,
      allow_imprecise_accumulation=False)
  assert attr is not None
  assert str(attr) == ("#stablehlo.dot_algorithm<lhs_precision_type = bf16, "
                       "rhs_precision_type = bf16, accumulation_type = f32, "
                       "lhs_component_count = 1, rhs_component_count = 1, "
                       "num_primitive_operations = 3, "
                       "allow_imprecise_accumulation = false>")
  assert isinstance(attr.lhs_precision_type, ir.BF16Type)
  assert isinstance(attr.rhs_precision_type, ir.BF16Type)
  assert isinstance(attr.accumulation_type, ir.F32Type)
  assert attr.lhs_component_count == 1
  assert attr.rhs_component_count == 1
  assert attr.num_primitive_operations == 3
  assert attr.allow_imprecise_accumulation == False


@run
def test_dot_dimension_numbers():
  attr = stablehlo.DotDimensionNumbers.get(
      lhs_batching_dimensions=[0, 1],
      rhs_batching_dimensions=[2, 3],
      lhs_contracting_dimensions=[4, 5],
      rhs_contracting_dimensions=[6, 7])
  assert attr is not None
  assert str(attr) == ("#stablehlo.dot<lhs_batching_dimensions = [0, 1], "
                       "rhs_batching_dimensions = [2, 3], "
                       "lhs_contracting_dimensions = [4, 5], "
                       "rhs_contracting_dimensions = [6, 7]>")
  assert attr.lhs_batching_dimensions == [0, 1]
  assert attr.rhs_batching_dimensions == [2, 3]
  assert attr.lhs_contracting_dimensions == [4, 5]
  assert attr.rhs_contracting_dimensions == [6, 7]


@run
def test_fft_type_attr():
  attr = stablehlo.FftTypeAttr.get("FFT")
  assert attr is not None
  assert str(attr) == ("#stablehlo<fft_type FFT>")
  assert attr.value == "FFT"


@run
def test_gather_dimension_numbers():
  attr = stablehlo.GatherDimensionNumbers.get(
      offset_dims=[1, 2],
      collapsed_slice_dims=[3, 4, 5],
      operand_batching_dims=[6, 7],
      start_indices_batching_dims=[8, 9],
      start_index_map=[10],
      index_vector_dim=11,
  )
  assert attr is not None
  assert str(attr) == (
      "#stablehlo.gather<offset_dims = [1, 2], "
      "collapsed_slice_dims = [3, 4, 5], "
      "operand_batching_dims = [6, 7], "
      "start_indices_batching_dims = [8, 9], "
      "start_index_map = [10], "
      "index_vector_dim = 11>"
  )
  assert attr.offset_dims == [1, 2]
  assert attr.collapsed_slice_dims == [3, 4, 5]
  assert attr.operand_batching_dims == [6, 7]
  assert attr.start_indices_batching_dims == [8, 9]
  assert attr.start_index_map == [10]
  assert attr.index_vector_dim == 11


@run
def test_output_operand_alias():
  attr = stablehlo.OutputOperandAlias.get(
      output_tuple_indices=[0],
      operand_index=0,
      operand_tuple_indices=[1])
  assert attr is not None
  assert str(attr) == ("#stablehlo.output_operand_alias<output_tuple_indices = [0], "
                       "operand_index = 0, "
                       "operand_tuple_indices = [1]>")
  assert attr.output_tuple_indices == [0]
  assert attr.operand_index == 0
  assert attr.operand_tuple_indices == [1]


@run
def test_precision_attr():
  attr = stablehlo.PrecisionAttr.get("DEFAULT")
  assert attr is not None
  assert str(attr) == ("#stablehlo<precision DEFAULT>")
  assert attr.value == "DEFAULT"


@run
def test_rng_algorithm_attr():
  attr = stablehlo.RngAlgorithmAttr.get("DEFAULT")
  assert attr is not None
  assert str(attr) == ("#stablehlo<rng_algorithm DEFAULT>")
  assert attr.value == "DEFAULT"


@run
def test_rng_distribution_attr():
  attr = stablehlo.RngDistributionAttr.get("UNIFORM")
  assert attr is not None
  assert str(attr) == ("#stablehlo<rng_distribution UNIFORM>")
  assert attr.value == "UNIFORM"


@run
def test_scatter_dimension_numbers():
  attr = stablehlo.ScatterDimensionNumbers.get(
      update_window_dims=[1, 2, 3],
      inserted_window_dims=[4, 5],
      input_batching_dims=[6, 7],
      scatter_indices_batching_dims=[8, 9],
      scattered_dims_to_operand_dims=[10, 11],
      index_vector_dim=12,
  )
  assert attr is not None
  assert str(attr) == (
      "#stablehlo.scatter<update_window_dims = [1, 2, 3], "
      "inserted_window_dims = [4, 5], "
      "input_batching_dims = [6, 7], "
      "scatter_indices_batching_dims = [8, 9], "
      "scatter_dims_to_operand_dims = [10, 11], "
      "index_vector_dim = 12>"
  )
  assert attr.update_window_dims == [1, 2, 3]
  assert attr.inserted_window_dims == [4, 5]
  assert attr.input_batching_dims == [6, 7]
  assert attr.scatter_indices_batching_dims == [8, 9]
  assert attr.scattered_dims_to_operand_dims == [10, 11]
  assert attr.index_vector_dim == 12


@run
def test_transpose_attr():
  attr = stablehlo.TransposeAttr.get("TRANSPOSE")
  assert attr is not None
  assert str(attr) == ("#stablehlo<transpose TRANSPOSE>")
  assert attr.value == "TRANSPOSE"


@run
def test_token_type():
  type = stablehlo.TokenType.get()
  assert type is not None
  assert str(type) == "!stablehlo.token"


@run
def test_type_extensions():
  dyn_size = ir.ShapedType.get_dynamic_size()
  attr = stablehlo.TypeExtensions.get(bounds=[128, dyn_size])
  assert attr is not None
  assert attr.bounds == [128, dyn_size]


@run
def test_api_version():
  api_version = stablehlo.get_api_version()
  assert type(api_version) == int
  assert api_version > 0


def is_semver_format(version_str):
  return re.match("^\d+\.\d+\.\d+$", version_str)


@run
def test_current_version():
  curr_version = stablehlo.get_current_version()
  assert is_semver_format(curr_version)


@run
def test_minimum_version():
  curr_version = stablehlo.get_minimum_version()
  assert is_semver_format(curr_version)


@run
def test_version_requirements():
  for req in (
      stablehlo.StablehloCompatibilityRequirement.NONE,
      stablehlo.StablehloCompatibilityRequirement.WEEK_4,
      stablehlo.StablehloCompatibilityRequirement.WEEK_12,
      stablehlo.StablehloCompatibilityRequirement.MAX,
  ):
    assert is_semver_format(
        stablehlo.get_version_from_compatibility_requirement(req)
    )


ASM_FORMAT = """
func.func @test(%arg0: tensor<{0}>) -> tensor<{0}> {{
  %0 = stablehlo.add %arg0, %arg0 : (tensor<{0}>, tensor<{0}>) -> tensor<{0}>
  func.return %0 : tensor<{0}>
}}
"""


# @run
# def test_reference_api():
#   # Formatted as (tensor_type, np_value)
#   # Program runs arg + arg, which is used for expected value
#   tests = [
#     # No numpy types for f8 - skipping fp8 tests
#     ("f16", np.asarray(1, np.float16)),
#     ("f32", np.asarray(2, np.float32)),
#     ("f64", np.asarray(3, np.double)),
#     ("1xi8", np.asarray([4], np.int8)),
#     ("1xi16", np.asarray([5], np.int16)),
#     ("1xi32", np.asarray([-6], np.int32)),
#     # Numpy's uint treated as int by DenseElementsAttr, skipping np.uint tests
#     ("2x2xf16", np.asarray([1, 2, 3, 4], np.float16).reshape(2,2)),
#     ("2x1x2xf16", np.asarray([1, 2, 3, 4], np.float16).reshape(2,1,2)),
#     ("?x?xf16", np.asarray([1, 2, 3, 4], np.float16).reshape(2,2)),
#     ("?x2xf16", np.asarray([1, 2, 3, 4], np.float16).reshape(2,2)),
#   ]
#   for test in tests:
#     tensor_type, arg = test
#     with ir.Context() as context:
#       stablehlo.register_dialect(context)
#       m = ir.Module.parse(ASM_FORMAT.format(tensor_type))
#       args = [ir.DenseIntElementsAttr.get(arg)]
#
#     actual = np.array(stablehlo.eval_module(m, args)[0])
#     expected = arg + arg
#     assert (actual == expected).all()
#

@run
def test_get_smaller_version():
  curr_version = stablehlo.get_current_version()
  min_version = stablehlo.get_minimum_version()
  assert stablehlo.get_smaller_version(curr_version, min_version) == min_version


@run
def test_serialization_apis():
  curr_version = stablehlo.get_current_version()

  with ir.Context() as context:
    stablehlo.register_dialect(context)
    m = ir.Module.parse(ASM_FORMAT.format("2xf32"))
    assert m is not None
    module_str = str(m)
    serialized = stablehlo.serialize_portable_artifact(m, curr_version)
    deserialized = stablehlo.deserialize_portable_artifact(context, serialized)
    assert module_str == str(deserialized)


@run
def test_str_serialization_apis():
  curr_version = stablehlo.get_current_version()

  def module_to_bytecode(module: ir.Module) -> bytes:
    output = io.BytesIO()
    module.operation.write_bytecode(file=output)
    return output.getvalue()

  with ir.Context() as context:
    stablehlo.register_dialect(context)
    m = ir.Module.parse(ASM_FORMAT.format("2xf32"))
    assert m is not None
    module_str = str(m)
    bytecode = module_to_bytecode(m)
    serialized = stablehlo.serialize_portable_artifact_str(
        bytecode, curr_version
    )
    deserialized = stablehlo.deserialize_portable_artifact_str(serialized)
    deserialized_module = ir.Module.parse(deserialized)
    assert module_str == str(deserialized_module)


@run
def test_register_passes():
  """Tests pass registration."""
  with ir.Context() as context:
    stablehlo.register_dialect(context)
    module = ir.Module.parse(ASM_FORMAT.format("2xf32"))
    assert module is not None

    stablehlo.register_stablehlo_passes()
    pipeline = [
        "stablehlo-legalize-to-vhlo",
        "vhlo-legalize-to-stablehlo",
    ]
    pipeline = pm.PassManager.parse(f"builtin.module({','.join(pipeline)})")

    cloned_module = module.operation.clone()
    pipeline.run(cloned_module.operation)
    assert str(module) == str(cloned_module)
