// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-padding-level{tiling-level=partial_reduction}),canonicalize,cse)" --split-input-file %s | FileCheck  %s


// A max reduction should have the operand padded with negative infinity.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: max_reduction
//       CHECK: %[[NEGINF:.+]] = arith.constant 0xFF800000 : f32
//       CHECK: tensor.pad
//       CHECK: tensor.yield %[[NEGINF]] : f32
//   CHECK-NOT: tensor.pad
func.func @max_reduction(%arg0: tensor<1x?xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1],
                       iterator_types = ["parallel", "reduction"]}
                       ins(%arg0 : tensor<1x?xf32>) outs(%arg1 : tensor<1xf32>)
   attrs =  {lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 4096]}>}
   {
   ^bb0(%in: f32, %out: f32):
     %1 = arith.maxnumf %in, %out : f32
     linalg.yield %1 : f32
   } -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// A min reduction should have the operand padded with positive infinity.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: min_reduction
//       CHECK: %[[POSINF:.+]] = arith.constant 0x7F800000 : f32
//       CHECK: tensor.pad
//       CHECK: tensor.yield %[[POSINF]] : f32
//   CHECK-NOT: tensor.pad
func.func @min_reduction(%arg0: tensor<1x?xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1],
                       iterator_types = ["parallel", "reduction"]}
                       ins(%arg0 : tensor<1x?xf32>) outs(%arg1 : tensor<1xf32>)
   attrs =  {lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 4096]}>}
   {
   ^bb0(%in: f32, %out: f32):
     %1 = arith.minnumf %in, %out : f32
     linalg.yield %1 : f32
   } -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// This reduction corresponds to the step in softmax where sum_{i} exp(x_i - MAX) is computed.
// The padding of the input tensor should be with negative infinity.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: sum_exp_sub_reduction
//       CHECK: %[[NEGINF:.+]] = arith.constant 0xFF800000 : f32
//       CHECK: tensor.pad
//       CHECK: tensor.yield %[[NEGINF]] : f32
//   CHECK-NOT: tensor.pad
func.func @sum_exp_sub_reduction(%arg0: tensor<1x?xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map1],
                       iterator_types = ["parallel", "reduction"]}
                       ins(%arg0, %arg1 : tensor<1x?xf32>, tensor<1xf32>) outs(%arg2 : tensor<1xf32>)
    attrs =  {lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 4096]}>}
    {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.subf %in, %in_0 : f32
      %2 = math.exp %1 : f32
      %3 = arith.addf %2, %out : f32
      linalg.yield %3 : f32
    } -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// This reduction corresponds to a standard inner product / dot product / matrix multiplication.
// The 2 inputs are of dynamic size, and are both padded with 0.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: standard_inner_product
//       CHECK: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f16
//       CHECK: tensor.pad
//       CHECK: tensor.yield %[[ZERO]] : f16
//       CHECK: tensor.pad
//       CHECK: tensor.yield %[[ZERO]] : f16
//   CHECK-NOT: tensor.pad
func.func @standard_inner_product(%arg0 : tensor<1x?xf16>, %arg1 : tensor<1x?xf16>, %arg2 : tensor<1xf16>) -> tensor<1xf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map1],
                       iterator_types = ["parallel", "reduction"]}
                       ins(%arg0, %arg1 : tensor<1x?xf16>, tensor<1x?xf16>) outs(%arg2 : tensor<1xf16>)
    attrs =  {lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 4096]}>}
    {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %1 = arith.mulf %in, %in_0 : f16
      %2 = arith.addf %out, %1 : f16
      linalg.yield %2 : f16
    } -> tensor<1xf16>
  return %0 : tensor<1xf16>
}

// -----

// The standard inner product is sum_{i} x_i * y_i, and the correct padding value in this case is 0. In other words
// the inner product between (x_0, x_1, x_2) and (y_0, y_1, y_2) is the same as the inner product
// between (x_0, x_1, x_2, 0, 0) and (y_0, y_1, y_2, 0, 0). If we switch things up and instead want to compute
// prod_{i} x_i + y_i, we can pad with 0 for x and 1 for y (or vica versa). In other words,
// (x_0 + y_0) * (x_1 + y_1) * (x_2 + y_2) is the same as
// (x_0 + y_0) * (x_1 + y_1) * (x_2 + y_2) + (0 + 1) * (0 + 1).

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: product_of_sum_reduction
//  CHECK-DAG: %[[ONE:.+]] = arith.constant 1.000000e+00 : f16
//  CHECK-DAG: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f16
//       CHECK: tensor.pad
//   CHECK-DAG: tensor.yield %[[ZERO]] : f16
//   CHECK-DAG: tensor.yield %[[ONE]] : f16
//   CHECK-NOT: tensor.pad
func.func @product_of_sum_reduction(%arg0 : tensor<1x?xf16>, %arg1 : tensor<1x?xf16>, %arg2 : tensor<1xf16>) -> tensor<1xf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map1],
                       iterator_types = ["parallel", "reduction"]}
                       ins(%arg0, %arg1 : tensor<1x?xf16>, tensor<1x?xf16>) outs(%arg2 : tensor<1xf16>)
    attrs =  {lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 4096]}>}
    {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %1 = arith.addf %in, %in_0 : f16
      %2 = arith.mulf %out, %1 : f16
      linalg.yield %2 : f16
    } -> tensor<1xf16>
  return %0 : tensor<1xf16>
}


// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: standard_inner_product_with_trunc
//       CHECK: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK: tensor.pad
//       CHECK: tensor.yield %[[ZERO]] : f32
//       CHECK: tensor.pad
//       CHECK: tensor.yield %[[ZERO]] : f32
//   CHECK-NOT: tensor.pad
func.func @standard_inner_product_with_trunc(%arg0 : tensor<1x?xf32>, %arg1 : tensor<1x?xf32>, %arg2 : tensor<1xf16>) -> tensor<1xf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map1],
                       iterator_types = ["parallel", "reduction"]}
                       ins(%arg0, %arg1 : tensor<1x?xf32>, tensor<1x?xf32>) outs(%arg2 : tensor<1xf16>)
    attrs =  {lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 4096], truncation = true}>}
    {
    ^bb0(%in: f32, %in_0: f32, %out: f16):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.truncf %1 : f32 to f16
      %3 = arith.addf %out, %2 : f16
      linalg.yield %3 : f16
    } -> tensor<1xf16>
  return %0 : tensor<1xf16>
}
