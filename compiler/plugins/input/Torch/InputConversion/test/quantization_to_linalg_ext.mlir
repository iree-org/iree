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

// Per-channel-group parameters index the input with its last dimension split
// into groups.
// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @per_channel_group(
// CHECK: iree_linalg_ext.quantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map1, #map]
// CHECK-SAME: ins({{.*}} : tensor<4x2x4xf32>, tensor<4x2xf32>, tensor<4x2xi8>)
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map1, #map]
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

// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @symmetric_per_channel_group(
// CHECK: iree_linalg_ext.dequantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map]
// CHECK-SAME: ins({{.*}} : tensor<4x2x4xi8>, tensor<4x2xf32>)
func.func @symmetric_per_channel_group(
    %input: !torch.vtensor<[4,8],si8>,
    %scales: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[4,8],f32> {
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 4
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

// Groups that do not evenly split the last dimension cannot be expressed by
// splitting it, so the op is left for the Torch-to-Linalg lowering.
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

// LinalgExt dequantizes to floats only, so an integer `out_dtype` is left for
// the Torch-to-Linalg lowering.
// CHECK-LABEL: func.func @integer_dequantize_result(
// CHECK: torch.quantized_decomposed.dequantize_per_tensor
// CHECK-NOT: iree_linalg_ext.dequantize_affine
func.func @integer_dequantize_result(%input: !torch.vtensor<[4,8],si8>)
    -> !torch.vtensor<[4,8],si32> {
  %scale = torch.constant.float 3.000000e-01
  %zero_point = torch.constant.int 0
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %int32 = torch.constant.int 3
  %out_dtype = torch.derefine %int32 : !torch.int to !torch.optional<int>
  %result = torch.quantized_decomposed.dequantize_per_tensor
      %input, %scale, %zero_point, %minimum, %maximum, %dtype, %out_dtype
      : !torch.vtensor<[4,8],si8>, !torch.float, !torch.int, !torch.int,
        !torch.int, !torch.int, !torch.optional<int>
        -> !torch.vtensor<[4,8],si32>
  return %result : !torch.vtensor<[4,8],si32>
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

// quantize_affine computes in the element type of the scale, so f16 scales
// for f32 values are left for the Torch-to-Linalg lowering.
// CHECK-LABEL: func.func @narrow_scale_quantize(
// CHECK: torch.quantized_decomposed.quantize_per_channel_group
// CHECK-NOT: iree_linalg_ext.quantize_affine
func.func @narrow_scale_quantize(
    %input: !torch.vtensor<[4,8],f32>,
    %scales: !torch.vtensor<[4,2],f16>,
    %zero_points: !torch.vtensor<[4,2],si8>) -> !torch.vtensor<[4,8],si8> {
  %minimum = torch.constant.int -128
  %maximum = torch.constant.int 127
  %dtype = torch.constant.int 1
  %group_size = torch.constant.int 4
  %result = torch.quantized_decomposed.quantize_per_channel_group
      %input, %scales, %zero_points, %minimum, %maximum, %dtype, %group_size
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[4,2],f16>,
        !torch.vtensor<[4,2],si8>, !torch.int, !torch.int, !torch.int,
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
