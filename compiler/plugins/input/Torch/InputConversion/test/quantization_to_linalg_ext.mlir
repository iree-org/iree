// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(torch-iree-torch-quantization-to-linalg-ext))" %s | FileCheck %s

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> ()>
// CHECK-LABEL: func.func @per_tensor(
// CHECK: arith.truncf %{{.+}} : f64 to f32
// CHECK: iree_linalg_ext.quantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map1, #map]
// CHECK-SAME: quant_min = 0
// CHECK-SAME: quant_max = 255
// CHECK-SAME: storage_unsigned
// CHECK-SAME: ins({{.*}} : tensor<4x8xf32>, f32, i64)
// CHECK-SAME: outs({{.*}} : tensor<4x8xi8>)
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map1, #map]
// CHECK-SAME: input_unsigned
// CHECK-SAME: ins({{.*}} : tensor<4x8xi8>, f32, i64)
func.func @per_tensor(%input: !torch.vtensor<[4,8],f32>)
    -> !torch.vtensor<[4,8],f32> {
  %scale = torch.constant.float 3.000000e-01
  %zero_point = torch.constant.int 10
  %minimum = torch.constant.int 0
  %maximum = torch.constant.int 255
  %dtype = torch.constant.int 0
  %none = torch.constant.none
  %out_dtype = torch.derefine %none : !torch.none to !torch.optional<int>
  %quantized = torch.quantized_decomposed.quantize_per_tensor
      %input, %scale, %zero_point, %minimum, %maximum, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,8],ui8>
  %result = torch.quantized_decomposed.dequantize_per_tensor
      %quantized, %scale, %zero_point, %minimum, %maximum, %dtype, %out_dtype
      : !torch.vtensor<[4,8],ui8>, !torch.float, !torch.int, !torch.int,
        !torch.int, !torch.int, !torch.optional<int>
        -> !torch.vtensor<[4,8],f32>
  return %result : !torch.vtensor<[4,8],f32>
}

// -----

// A negative axis is normalized before constructing the parameter maps.
// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL: func.func @per_channel(
// CHECK: iree_linalg_ext.quantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map1, #map]
// CHECK-SAME: zp_unsigned
// CHECK-SAME: ins({{.*}} : tensor<4x8xf32>, tensor<8xf32>, tensor<8xi8>)
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map1, #map]
// CHECK-SAME: zp_unsigned
func.func @per_channel(
    %input: !torch.vtensor<[4,8],f32>,
    %scales: !torch.vtensor<[8],f32>,
    %zero_points: !torch.vtensor<[8],ui8>)
    -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int -1
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  %out_dtype = torch.derefine %none : !torch.none to !torch.optional<int>
  %quantized = torch.quantized_decomposed.quantize_per_channel
      %input, %scales, %zero_points, %axis, %minimum, %maximum, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8],f32>,
        !torch.vtensor<[8],ui8>, !torch.int, !torch.int, !torch.int, !torch.int
        -> !torch.vtensor<[4,8],si8>
  %result = torch.quantized_decomposed.dequantize_per_channel
      %quantized, %scales, %zero_points, %axis, %minimum, %maximum, %dtype,
      %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[8],f32>,
        !torch.vtensor<[8],ui8>, !torch.int, !torch.int, !torch.int, !torch.int,
        !torch.optional<int> -> !torch.vtensor<[4,8],f32>
  return %result : !torch.vtensor<[4,8],f32>
}

// -----

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func.func @symmetric_per_channel(
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map]
// CHECK-SAME: ins({{.*}} : tensor<4x8xi8>, tensor<4xf32>)
func.func @symmetric_per_channel(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 0
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  %out_dtype = torch.derefine %none : !torch.none to !torch.optional<int>
  %result = torch.quantized_decomposed.dequantize_per_channel
      %input, %scales, %none, %axis, %minimum, %maximum, %dtype, %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4],f32>, !torch.none,
        !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
        -> !torch.vtensor<[4,8],f32>
  return %result : !torch.vtensor<[4,8],f32>
}

// -----

// The parameter maps of a single channel index no dimension, so the unit scale
// is collapsed to a rank-0 tensor.
// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> ()>
// CHECK-LABEL: func.func @single_channel(
// CHECK: %[[SCALE:.+]] = tensor.collapse_shape %{{.+}} []
// CHECK-SAME: tensor<1xf32> into tensor<f32>
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map]
// CHECK-SAME: ins(%{{.+}}, %[[SCALE]] : tensor<1x8xi8>, tensor<f32>)
func.func @single_channel(
    %input: !torch.vtensor<[1,8],si8>,
    %scales: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,8],f32> {
  %axis = torch.constant.int 0
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  %out_dtype = torch.derefine %none : !torch.none to !torch.optional<int>
  %result = torch.quantized_decomposed.dequantize_per_channel
      %input, %scales, %none, %axis, %minimum, %maximum, %dtype, %out_dtype
      : !torch.vtensor<[1,8],si8>, !torch.vtensor<[1],f32>, !torch.none,
        !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
        -> !torch.vtensor<[1,8],f32>
  return %result : !torch.vtensor<[1,8],f32>
}

// -----

// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @per_channel_group(
// CHECK: %[[BLOCKED_INPUT:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0], [1, 2]]
// CHECK-SAME: output_shape [4, 2, 4] : tensor<4x8xf32> into tensor<4x2x4xf32>
// CHECK: %[[QUANTIZED:.+]] = iree_linalg_ext.quantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map1, #map]
// CHECK-SAME: ins(%[[BLOCKED_INPUT]], %{{.+}}, %{{.+}} : tensor<4x2x4xf32>,
// CHECK-SAME: tensor<4x2xf32>, tensor<4x2xi8>)
// CHECK: %[[STORAGE:.+]] = tensor.collapse_shape %[[QUANTIZED]]
// CHECK-SAME: {{\[}}[0], [1, 2]] : tensor<4x2x4xi8> into tensor<4x8xi8>
// CHECK: %[[BLOCKED_STORAGE:.+]] = tensor.expand_shape %[[STORAGE]]
// CHECK-SAME: {{\[}}[0], [1, 2]] output_shape [4, 2, 4]
// CHECK-SAME: tensor<4x8xi8> into tensor<4x2x4xi8>
// CHECK: %[[DEQUANTIZED:.+]] = iree_linalg_ext.dequantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map1, #map]
// CHECK-SAME: ins(%[[BLOCKED_STORAGE]], %{{.+}}, %{{.+}} : tensor<4x2x4xi8>,
// CHECK-SAME: tensor<4x2xf32>, tensor<4x2xi8>)
// CHECK: tensor.collapse_shape %[[DEQUANTIZED]]
// CHECK-SAME: {{\[}}[0], [1, 2]] : tensor<4x2x4xf32> into tensor<4x8xf32>
func.func @per_channel_group(
    %input: !torch.vtensor<[4,8],f32>,
    %scales: !torch.vtensor<[4,2],f32>,
    %zero_points: !torch.vtensor<[4,2],si8>) -> !torch.vtensor<[4,8],f32> {
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 4
  %out_dtype = torch.constant.int 6
  %quantized = torch.quantized_decomposed.quantize_per_channel_group
      %input, %scales, %zero_points, %minimum, %maximum, %dtype, %group_size
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[4,2],f32>,
        !torch.vtensor<[4,2],si8>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[4,8],si8>
  %result = torch.quantized_decomposed.dequantize_per_channel_group
      %quantized, %scales, %zero_points, %minimum, %maximum, %dtype,
      %group_size, %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4,2],f32>,
        !torch.vtensor<[4,2],si8>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,8],f32>
  return %result : !torch.vtensor<[4,8],f32>
}

// -----

// A group spanning the whole row has one scale per row, so the scales are
// collapsed and indexed by the row.
// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func.func @group_spanning_row(
// CHECK-NOT: tensor.expand_shape
// CHECK: %[[SCALES:.+]] = tensor.collapse_shape %{{.+}} {{\[}}[0, 1]]
// CHECK-SAME: tensor<4x1xf32> into tensor<4xf32>
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map]
// CHECK-SAME: ins(%{{.+}}, %[[SCALES]] : tensor<4x8xi8>, tensor<4xf32>)
func.func @group_spanning_row(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[4,1],f32>) -> !torch.vtensor<[4,8],f32> {
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 8
  %none = torch.constant.none
  %out_dtype = torch.constant.int 6
  %result = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %none, %minimum, %maximum, %dtype, %group_size,
      %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4,1],f32>, !torch.none,
        !torch.int, !torch.int, !torch.int, !torch.int, !torch.int
        -> !torch.vtensor<[4,8],f32>
  return %result : !torch.vtensor<[4,8],f32>
}

// -----

// Each vector along the last dimension is a token with its own scale and zero
// point. PT2E keeps a unit last dimension in their shape, which is dropped so
// that LinalgExt indexes them by the leading dimensions.
// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @per_token(
// CHECK-NOT: tensor.expand_shape
// CHECK: %[[SCALES:.+]] = tensor.collapse_shape %{{.+}} {{\[}}[0], [1, 2]]
// CHECK-SAME: tensor<2x?x1xf64> into tensor<2x?xf64>
// CHECK: %[[ZERO_POINTS:.+]] = tensor.collapse_shape %{{.+}} {{\[}}[0], [1, 2]]
// CHECK-SAME: tensor<2x?x1xi64> into tensor<2x?xi64>
// CHECK: %[[QUANTIZED:.+]] = iree_linalg_ext.quantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map1, #map]
// CHECK-SAME: ins(%{{.+}}, %[[SCALES]], %[[ZERO_POINTS]] : tensor<2x?x8xf32>,
// CHECK-SAME: tensor<2x?xf64>, tensor<2x?xi64>)
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map1, #map]
// CHECK-SAME: ins(%[[QUANTIZED]], %{{.+}}, %{{.+}} : tensor<2x?x8xi8>,
// CHECK-SAME: tensor<2x?xf64>, tensor<2x?xi64>)
func.func @per_token(
    %input: !torch.vtensor<[2,?,8],f32>,
    %scales: !torch.vtensor<[2,?,1],f64>,
    %zero_points: !torch.vtensor<[2,?,1],si64>) -> !torch.vtensor<[2,?,8],f32> {
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %out_dtype = torch.constant.int 6
  %quantized = torch.quantized_decomposed.quantize_per_token
      %input, %scales, %zero_points, %minimum, %maximum, %dtype
      : !torch.vtensor<[2,?,8],f32>, !torch.vtensor<[2,?,1],f64>,
        !torch.vtensor<[2,?,1],si64>, !torch.int, !torch.int, !torch.int
        -> !torch.vtensor<[2,?,8],si8>
  %result = torch.quantized_decomposed.dequantize_per_token
      %quantized, %scales, %zero_points, %minimum, %maximum, %dtype,
      %out_dtype
      : !torch.vtensor<[2,?,8],si8>, !torch.vtensor<[2,?,1],f64>,
        !torch.vtensor<[2,?,1],si64>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[2,?,8],f32>
  return %result : !torch.vtensor<[2,?,8],f32>
}

// -----

// Dynamic bounds cannot be represented as LinalgExt attributes. The pass
// leaves this op for the general Torch-to-Linalg lowering that follows it.
// CHECK-LABEL: func.func @dynamic_bounds(
// CHECK: torch.quantized_decomposed.quantize_per_tensor
// CHECK-NOT: iree_linalg_ext.quantize_affine
func.func @dynamic_bounds(
    %input: !torch.vtensor<[4,8],f32>, %minimum: !torch.int)
    -> !torch.vtensor<[4,8],si8> {
  %scale = torch.constant.float 3.000000e-01
  %zero_point = torch.constant.int 0
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %result = torch.quantized_decomposed.quantize_per_tensor
      %input, %scale, %zero_point, %minimum, %maximum, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,8],si8>
  return %result : !torch.vtensor<[4,8],si8>
}

// -----

// An axis outside the input rank cannot index the parameter maps, so the op is
// left for the Torch-to-Linalg lowering.
// CHECK-LABEL: func.func @out_of_range_axis(
// CHECK: torch.quantized_decomposed.dequantize_per_channel
// CHECK-NOT: iree_linalg_ext.dequantize_affine
func.func @out_of_range_axis(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4,8],f32> {
  %axis = torch.constant.int 2
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  %out_dtype = torch.derefine %none : !torch.none to !torch.optional<int>
  %result = torch.quantized_decomposed.dequantize_per_channel
      %input, %scales, %none, %axis, %minimum, %maximum, %dtype, %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4],f32>, !torch.none,
        !torch.int, !torch.int, !torch.int, !torch.int, !torch.optional<int>
        -> !torch.vtensor<[4,8],f32>
  return %result : !torch.vtensor<[4,8],f32>
}

// -----

// A group size that does not divide the last dimension cannot split the input
// into groups, so the op is left for the Torch-to-Linalg lowering.
// CHECK-LABEL: func.func @indivisible_group_size(
// CHECK: torch.quantized_decomposed.dequantize_per_channel_group
// CHECK-NOT: iree_linalg_ext.dequantize_affine
func.func @indivisible_group_size(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[4,8],f32> {
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 3
  %none = torch.constant.none
  %out_dtype = torch.constant.int 6
  %result = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %none, %minimum, %maximum, %dtype, %group_size,
      %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4,2],f32>, !torch.none,
        !torch.int, !torch.int, !torch.int, !torch.int, !torch.int
        -> !torch.vtensor<[4,8],f32>
  return %result : !torch.vtensor<[4,8],f32>
}

// -----

// LinalgExt storage is a signless integer, so fp8 storage is left for the
// Torch-to-Linalg lowering.
// CHECK-LABEL: func.func @float_storage(
// CHECK: torch.quantized_decomposed.quantize_per_tensor
// CHECK-NOT: iree_linalg_ext.quantize_affine
func.func @float_storage(%input: !torch.vtensor<[4,8],f32>)
    -> !torch.vtensor<[4,8],f8E4M3FN> {
  %scale = torch.constant.float 3.000000e-01
  %zero_point = torch.constant.int 0
  %minimum = torch.constant.int -448
  %maximum = torch.constant.int 448
  %dtype = torch.constant.int 24
  %result = torch.quantized_decomposed.quantize_per_tensor
      %input, %scale, %zero_point, %minimum, %maximum, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,8],f8E4M3FN>
  return %result : !torch.vtensor<[4,8],f8E4M3FN>
}

// -----

// LinalgExt cannot mix distinct float types of equal width, so bf16 values
// with f16 scales are left for the Torch-to-Linalg lowering.
// CHECK-LABEL: func.func @equal_width_scale(
// CHECK: torch.quantized_decomposed.quantize_per_channel
// CHECK-NOT: iree_linalg_ext.quantize_affine
func.func @equal_width_scale(
    %input: !torch.vtensor<[4,8],bf16>,
    %scales: !torch.vtensor<[4],f16>,
    %zero_points: !torch.vtensor<[4],si8>) -> !torch.vtensor<[4,8],si8> {
  %axis = torch.constant.int 0
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %result = torch.quantized_decomposed.quantize_per_channel
      %input, %scales, %zero_points, %axis, %minimum, %maximum, %dtype
      : !torch.vtensor<[4,8],bf16>, !torch.vtensor<[4],f16>,
        !torch.vtensor<[4],si8>, !torch.int, !torch.int, !torch.int,
        !torch.int -> !torch.vtensor<[4,8],si8>
  return %result : !torch.vtensor<[4,8],si8>
}

// -----

// Inductor applies a per-tensor scale in f32 and rounds once to the result
// dtype, so the scale stays f32 for an f16 result.
// CHECK-LABEL: func.func @per_tensor_f16_result(
// CHECK: %[[SCALE:.+]] = arith.truncf %{{.+}} : f64 to f32
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: ins(%{{.+}}, %[[SCALE]], %{{.+}} : tensor<4x8xi8>, f32, i64)
// CHECK-SAME: outs({{.*}} : tensor<4x8xf16>)
func.func @per_tensor_f16_result(%input: !torch.vtensor<[4,8],si8>)
    -> !torch.vtensor<[4,8],f16> {
  %scale = torch.constant.float 3.000000e-01
  %zero_point = torch.constant.int 0
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %half = torch.constant.int 5
  %out_dtype = torch.derefine %half : !torch.int to !torch.optional<int>
  %result = torch.quantized_decomposed.dequantize_per_tensor
      %input, %scale, %zero_point, %minimum, %maximum, %dtype, %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int,
        !torch.int, !torch.int, !torch.optional<int>
        -> !torch.vtensor<[4,8],f16>
  return %result : !torch.vtensor<[4,8],f16>
}

// -----

// Inductor also computes f64 results in f32 before widening them.
// CHECK-LABEL: func.func @per_tensor_f64_result(
// CHECK: %[[SCALE:.+]] = arith.truncf %{{.+}} : f64 to f32
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: ins(%{{.+}}, %[[SCALE]], %{{.+}} : tensor<4x8xi8>, f32, i64)
// CHECK-SAME: outs({{.*}} : tensor<4x8xf64>)
func.func @per_tensor_f64_result(%input: !torch.vtensor<[4,8],si8>)
    -> !torch.vtensor<[4,8],f64> {
  %scale = torch.constant.float 3.000000e-01
  %zero_point = torch.constant.int 0
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %double = torch.constant.int 7
  %out_dtype = torch.derefine %double : !torch.int to !torch.optional<int>
  %result = torch.quantized_decomposed.dequantize_per_tensor
      %input, %scale, %zero_point, %minimum, %maximum, %dtype, %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int,
        !torch.int, !torch.int, !torch.optional<int>
        -> !torch.vtensor<[4,8],f64>
  return %result : !torch.vtensor<[4,8],f64>
}

// -----

// Exports store the zero points of symmetric weights as float zeros. They are
// dropped like a None zero point.
// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @float_zero_point_zeros(
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map]
// CHECK-SAME: ins({{.*}} : tensor<4x2x4xi8>, tensor<4x2xf32>)
func.func @float_zero_point_zeros(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[4,8],f32> {
  %zero_points = torch.vtensor.literal(dense<0.000000e+00> : tensor<4x2xf32>)
      : !torch.vtensor<[4,2],f32>
  %minimum = torch.constant.int -7
  %maximum = torch.constant.int 7
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 4
  %out_dtype = torch.constant.int 6
  %result = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %zero_points, %minimum, %maximum, %dtype, %group_size,
      %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4,2],f32>,
        !torch.vtensor<[4,2],f32>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,8],f32>
  return %result : !torch.vtensor<[4,8],f32>
}

// -----

// LinalgExt zero points are integers, so other float zero points, such as the
// fractional ones of HQQ, are left for the Torch-to-Linalg lowering.
// CHECK-LABEL: func.func @nonzero_float_zero_point(
// CHECK: torch.quantized_decomposed.dequantize_per_channel_group
// CHECK-NOT: iree_linalg_ext.dequantize_affine
func.func @nonzero_float_zero_point(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[4,8],f32> {
  %zero_points = torch.vtensor.literal(dense<5.000000e-01> : tensor<4x2xf32>)
      : !torch.vtensor<[4,2],f32>
  %minimum = torch.constant.int -8
  %maximum = torch.constant.int 7
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 4
  %out_dtype = torch.constant.int 6
  %result = torch.quantized_decomposed.dequantize_per_channel_group
      %input, %scales, %zero_points, %minimum, %maximum, %dtype, %group_size,
      %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.vtensor<[4,2],f32>,
        !torch.vtensor<[4,2],f32>, !torch.int, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,8],f32>
  return %result : !torch.vtensor<[4,8],f32>
}

// -----

// Each token has its own scale and zero point, which torch.export gives without
// a unit last dimension, so LinalgExt indexes them by the leading dimensions.
// Like torchao, it computes the bf16 input in the wider f32 scale type.
// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @torchao_per_token(
// CHECK: iree_linalg_ext.quantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map1, #map]
// CHECK-SAME: quant_min = -128
// CHECK-SAME: quant_max = 127
// CHECK-SAME: ins({{.*}} : tensor<?x3x16xbf16>, tensor<?x3xf32>,
// CHECK-SAME: tensor<?x3xi8>)
func.func @torchao_per_token(
    %input: !torch.vtensor<[?,3,16],bf16>,
    %scale: !torch.vtensor<[?,3],f32>,
    %zero_point: !torch.vtensor<[?,3],si8>) -> !torch.vtensor<[?,3,16],si8> {
  %one = torch.constant.int 1
  %token_size = torch.constant.int 16
  %dtype = torch.constant.int 1
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %block_size = torch.prim.ListConstruct %one, %one, %token_size
      : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %result = torch.torchao.quantize_affine
      %input, %block_size, %scale, %zero_point, %dtype, %minimum, %maximum
      : !torch.vtensor<[?,3,16],bf16>, !torch.list<int>,
        !torch.vtensor<[?,3],f32>, !torch.vtensor<[?,3],si8>, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[?,3,16],si8>
  return %result : !torch.vtensor<[?,3,16],si8>
}

// -----

// torch.export gives the block entry of a dynamic dimension as its size, which
// spans the dimension, and per-tensor parameters as rank-0 tensors, which the
// parameter maps index by no dimension.
// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> ()>
// CHECK-LABEL: func.func @torchao_dynamic_per_tensor(
// CHECK: iree_linalg_ext.quantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map1, #map]
// CHECK-SAME: ins({{.*}} : tensor<2x?x64xf32>, tensor<f32>, tensor<i32>)
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: ins({{.*}} : tensor<2x?x64xi8>, tensor<f32>, tensor<i32>)
func.func @torchao_dynamic_per_tensor(
    %input: !torch.vtensor<[2,?,64],f32>,
    %scale: !torch.vtensor<[],f32>,
    %zero_point: !torch.vtensor<[],si32>) -> !torch.vtensor<[2,?,64],f32> {
  %dim = torch.constant.int 1
  %batch = torch.constant.int 2
  %features = torch.constant.int 64
  %dtype = torch.constant.int 1
  %out_dtype = torch.constant.int 6
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %size = torch.aten.size.int %input, %dim
      : !torch.vtensor<[2,?,64],f32>, !torch.int -> !torch.int
  %block_size = torch.prim.ListConstruct %batch, %size, %features
      : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %quantized = torch.torchao.quantize_affine
      %input, %block_size, %scale, %zero_point, %dtype, %minimum, %maximum
      : !torch.vtensor<[2,?,64],f32>, !torch.list<int>, !torch.vtensor<[],f32>,
        !torch.vtensor<[],si32>, !torch.int, !torch.int, !torch.int
        -> !torch.vtensor<[2,?,64],si8>
  %result = torch.torchao.dequantize_affine
      %quantized, %block_size, %scale, %zero_point, %dtype, %minimum,
      %maximum, %out_dtype
      : !torch.vtensor<[2,?,64],si8>, !torch.list<int>, !torch.vtensor<[],f32>,
        !torch.vtensor<[],si32>, !torch.int, !torch.int, !torch.int, !torch.int
        -> !torch.vtensor<[2,?,64],f32>
  return %result : !torch.vtensor<[2,?,64],f32>
}

// -----

// 2x4 blocks split both dimensions, so the scales are indexed by the block row
// and column of the expanded input. None bounds are the bounds of the storage
// dtype.
// CHECK: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-LABEL: func.func @torchao_symmetric_per_block(
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map]
// CHECK-SAME: quant_min = -128
// CHECK-SAME: quant_max = 127
// CHECK-SAME: ins({{.*}} : tensor<2x2x2x4xi8>, tensor<2x2xf32>)
func.func @torchao_symmetric_per_block(
    %input: !torch.vtensor<[4,8],si8>,
    %scale: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[4,8],bf16> {
  %block_rows = torch.constant.int 2
  %block_columns = torch.constant.int 4
  %dtype = torch.constant.int 1
  %out_dtype = torch.constant.int 15
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %block_rows, %block_columns
      : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.torchao.dequantize_affine
      %input, %block_size, %scale, %none, %dtype, %none, %none, %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.list<int>, !torch.vtensor<[2,2],f32>,
        !torch.none, !torch.int, !torch.none, !torch.none, !torch.int
        -> !torch.vtensor<[4,8],bf16>
  return %result : !torch.vtensor<[4,8],bf16>
}

// -----

// LinalgExt operands are signless, so unsigned storage and zero points set the
// unsigned flags. None bounds of uint8 storage are 0 and 255.
// CHECK-LABEL: func.func @torchao_unsigned(
// CHECK: iree_linalg_ext.quantize_affine
// CHECK-SAME: quant_min = 0
// CHECK-SAME: quant_max = 255
// CHECK-SAME: storage_unsigned
// CHECK-SAME: zp_unsigned
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: quant_min = 0
// CHECK-SAME: quant_max = 255
// CHECK-SAME: input_unsigned
// CHECK-SAME: zp_unsigned
func.func @torchao_unsigned(
    %input: !torch.vtensor<[4,8],f32>,
    %scale: !torch.vtensor<[4,1],f32>,
    %zero_point: !torch.vtensor<[4,1],ui8>) -> !torch.vtensor<[4,8],f32> {
  %one = torch.constant.int 1
  %row_size = torch.constant.int 8
  %dtype = torch.constant.int 0
  %out_dtype = torch.constant.int 6
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %one, %row_size
      : (!torch.int, !torch.int) -> !torch.list<int>
  %quantized = torch.torchao.quantize_affine
      %input, %block_size, %scale, %zero_point, %dtype, %none, %none
      : !torch.vtensor<[4,8],f32>, !torch.list<int>, !torch.vtensor<[4,1],f32>,
        !torch.vtensor<[4,1],ui8>, !torch.int, !torch.none, !torch.none
        -> !torch.vtensor<[4,8],ui8>
  %result = torch.torchao.dequantize_affine
      %quantized, %block_size, %scale, %zero_point, %dtype, %none, %none,
      %out_dtype
      : !torch.vtensor<[4,8],ui8>, !torch.list<int>, !torch.vtensor<[4,1],f32>,
        !torch.vtensor<[4,1],ui8>, !torch.int, !torch.none, !torch.none,
        !torch.int -> !torch.vtensor<[4,8],f32>
  return %result : !torch.vtensor<[4,8],f32>
}

// -----

// Expanding a dimension into its blocks needs a static size, so an op whose
// block splits a dynamic dimension is left for the Torch-to-Linalg lowering.
// CHECK-LABEL: func.func @torchao_dynamic_blocked_dimension(
// CHECK: torch.torchao.dequantize_affine
// CHECK-NOT: iree_linalg_ext.dequantize_affine
func.func @torchao_dynamic_blocked_dimension(
    %input: !torch.vtensor<[4,?],si8>,
    %scale: !torch.vtensor<[4,?],f32>) -> !torch.vtensor<[4,?],f32> {
  %one = torch.constant.int 1
  %group_size = torch.constant.int 4
  %dtype = torch.constant.int 1
  %out_dtype = torch.constant.int 6
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %one, %group_size
      : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.torchao.dequantize_affine
      %input, %block_size, %scale, %none, %dtype, %none, %none, %out_dtype
      : !torch.vtensor<[4,?],si8>, !torch.list<int>, !torch.vtensor<[4,?],f32>,
        !torch.none, !torch.int, !torch.none, !torch.none, !torch.int
        -> !torch.vtensor<[4,?],f32>
  return %result : !torch.vtensor<[4,?],f32>
}

// -----

// torchao exports also store the zero points of symmetric weights as float
// zeros, which are dropped as for PT2E.
// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @torchao_float_zero_point_zeros(
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map]
// CHECK-SAME: ins({{.*}} : tensor<4x2x4xi8>, tensor<4x2xf32>)
func.func @torchao_float_zero_point_zeros(
    %input: !torch.vtensor<[4,8],si8>,
    %scale: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[4,8],f32> {
  %zero_point = torch.vtensor.literal(dense<0.000000e+00> : tensor<4x2xf32>)
      : !torch.vtensor<[4,2],f32>
  %one = torch.constant.int 1
  %group_size = torch.constant.int 4
  %dtype = torch.constant.int 1
  %out_dtype = torch.constant.int 6
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %one, %group_size
      : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.torchao.dequantize_affine
      %input, %block_size, %scale, %zero_point, %dtype, %none, %none, %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.list<int>, !torch.vtensor<[4,2],f32>,
        !torch.vtensor<[4,2],f32>, !torch.int, !torch.none, !torch.none,
        !torch.int -> !torch.vtensor<[4,8],f32>
  return %result : !torch.vtensor<[4,8],f32>
}
