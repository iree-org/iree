// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(torch-iree-torch-quantization-to-linalg-ext))" %s | FileCheck %s

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> ()>
// CHECK-LABEL: func.func @per_tensor(
// CHECK: iree_linalg_ext.quantize_affine
// CHECK-SAME: indexing_maps = [#map, #map1, #map1, #map]
// CHECK-SAME: quant_max = 255 : i64
// CHECK-SAME: quant_min = 0 : i64
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
  %dtype = torch.constant.int 2
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
  %dtype = torch.constant.int 2
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
  %dtype = torch.constant.int 2
  %result = torch.quantized_decomposed.quantize_per_tensor
      %input, %scale, %zero_point, %minimum, %maximum, %dtype
      : !torch.vtensor<[4,8],f32>, !torch.float, !torch.int, !torch.int,
        !torch.int, !torch.int -> !torch.vtensor<[4,8],si8>
  return %result : !torch.vtensor<[4,8],si8>
}
