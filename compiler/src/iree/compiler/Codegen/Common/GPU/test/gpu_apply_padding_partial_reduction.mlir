// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-padding-level{tiling-level=partial_reduction}),canonicalize,cse)" --split-input-file %s | FileCheck  %s


// This reduction corresponds to the sum-reduction step in (stable) softmax
// where sum_{i} exp(x_i - max_{j} exp(x_j)) is computed. Two options for padding here are:

// 1) pad the input with -infinity. This'd work because e^(-infinity) = 0,
//    and 0 is the additive identity.

// 2) pad with any value/poison. Then do a selection on the yielded
//    value based on the index, yielding 0 for indices corresponding
//    to the padded region.
//
// The current implementation (see applyPaddingLevel) uses option 2, because
// it is a more general purpose solution.

//  CHECK: #[[MAP:.+]]     = affine_map<()[s0, s1] -> (-s1 + (s0 ceildiv 4096) * 4096)>
//  CHECK:                   sum_exp_sub_reduction
//  CHECK: %[[ZEROF32:.+]] = arith.constant 0.000000e+00 : f32
//  CHECK: %[[ONE:.+]]     = arith.constant 1 : index
//  CHECK: %[[DIMARG0:.+]] = tensor.dim %arg0, %[[ONE]] : tensor<1x?xf32>
//  CHECK: %[[CEIL:.+]]    = affine.apply #[[MAP]]()[%[[DIMARG0]], %[[DIMARG0]]]
//  CHECK:                   tensor.pad %arg0 low[0, 0] high[0, %[[CEIL]]
//  CHECK:                   linalg.generic
//  CHECK:                   ^bb0(
//  CHECK:                   arith.subf
//  CHECK:  %[[EXP:.+]]    = math.exp
//  CHECK:  %[[INDEX1:.+]] = linalg.index 1 : index
//  CHECK:  %[[CMP:.+]]    = arith.cmpi ult, %[[INDEX1]], %[[DIMARG0]] : index
//  CHECK:  %[[SELECT:.+]] = arith.select %[[CMP]], %[[EXP:.+]], %[[ZEROF32]] : f32
//  CHECK:  %[[ADD:.+]]   = arith.addf %[[SELECT]], %out : f32
//  CHECK:  linalg.yield %[[ADD]] : f32

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
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


// Tests of max and min reductions.
//
// Note from arith.minnumf and arith.maxnumf documentation:
// "if one of the arguments is NaN, then the result is the other argument"
// So we check that the selected value in the padded region is one of the
// NaN values.

// CHECK-LABEL: max_reduction
//   CHECK-DAG: %[[NANVAL:.+]]  = arith.constant 0xFFC00000 : f32
//   CHECK-DAG: %[[C1:.+]]      = arith.constant 1 : index
//   CHECK-DAG: %[[DIMARG0:.+]] = tensor.dim %arg0, %[[C1]] : tensor<1x?xf32>
//       CHECK:                   linalg.generic
//       CHECK:                   ^bb0(%[[IN:[A-Za-z0-9]+]]
//       CHECK: %[[INDEX1:.+]]  = linalg.index 1 : index
//       CHECK: %[[CMP:.+]]     = arith.cmpi ult, %[[INDEX1]], %[[DIMARG0]] : index
//       CHECK: %[[SELECT:.+]]  = arith.select %[[CMP]], %[[IN]], %[[NANVAL]] : f32
//       CHECK: %[[MAX:.+]]     = arith.maxnumf %[[SELECT]], %out : f32
//       CHECK: linalg.yield      %[[MAX]] : f32
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
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

// CHECK-LABEL: min_reduction
//   CHECK-DAG: %[[NANVAL:.+]]  = arith.constant 0x7FC00000 : f32
//   CHECK-DAG: %[[C1:.+]]      = arith.constant 1 : index
//   CHECK-DAG: %[[DIMARG0:.+]] = tensor.dim %arg0, %[[C1]] : tensor<1x?xf32>
//       CHECK:                   linalg.generic
//       CHECK:                   ^bb0(%[[IN:[A-Za-z0-9]+]]
//       CHECK: %[[INDEX1:.+]]  = linalg.index 1 : index
//       CHECK: %[[CMP:.+]]     = arith.cmpi ult, %[[INDEX1]], %[[DIMARG0]] : index
//       CHECK: %[[SELECT:.+]]  = arith.select %[[CMP]], %[[IN]], %[[NANVAL]] : f32
//       CHECK: %[[MIN:.+]]     = arith.minnumf %[[SELECT]], %out : f32
//       CHECK: linalg.yield      %[[MIN]] : f32
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
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


// This reduction corresponds to a standard inner product.

// CHECK-LABEL: standard_inner_product
//   CHECK-DAG: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f16
//   CHECK-DAG: %[[C1:.+]]      = arith.constant 1 : index
//   CHECK-DAG: %[[DIMARG0:.+]] = tensor.dim %arg0, %[[C1]] : tensor<1x?xf16>
//       CHECK: linalg.generic
//       CHECK: ^bb0(
//       CHECK: %[[MUL:.+]] = arith.mulf
//       CHECK: %[[INDEX1:.+]]  = linalg.index 1 : index
//       CHECK: %[[CMP:.+]]     = arith.cmpi ult, %[[INDEX1]], %[[DIMARG0]] : index
//       CHECK: %[[SELECT:.+]]  = arith.select %[[CMP]], %[[MUL]], %[[ZERO]] : f16
//       CHECK: %[[ADD:.+]]    = arith.addf %out, %[[SELECT]] : f16
//       CHECK: linalg.yield %[[ADD]] : f16
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
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

// Inner product where the accumulation (add) is in f16 but the multiplication is in f32
// Check for an f16 zero as the reduction identity.

// CHECK-LABEL: standard_inner_product_with_trunc
//   CHECK-DAG: %[[ZERO:.+]]    = arith.constant 0.000000e+00 : f16
//   CHECK-DAG: %[[C1:.+]]      = arith.constant 1 : index
//   CHECK-DAG: %[[DIMARG0:.+]] = tensor.dim %arg0, %[[C1]] : tensor<1x?xf32>
//       CHECK:                   tensor.pad
//       CHECK:                   tensor.pad
//   CHECK-NOT:                   tensor.pad
//       CHECK:                   linalg.generic
//       CHECK:                   ^bb0(
//       CHECK: %[[MUL:.+]]     = arith.mulf
//       CHECK: %[[TRUNC:.+]]   = arith.truncf %[[MUL]] : f32 to f16
//       CHECK: %[[INDEX1:.+]]  = linalg.index 1 : index
//       CHECK: %[[CMP:.+]]     = arith.cmpi ult, %[[INDEX1]], %[[DIMARG0]] : index
//       CHECK: %[[SELECT:.+]]  = arith.select %[[CMP]], %[[TRUNC]], %[[ZERO]] : f16
//       CHECK: %[[ADD:.+]]     = arith.addf %out, %[[SELECT]] : f16
//       CHECK: linalg.yield %[[ADD]] : f16
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
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


// -----

// In this example, the reduction type is multiplicative, so we check that
// the value selected in the padded part of the iteration space is 1, the multiplicative identity.

// CHECK-LABEL: product_of_sum_reduction
//       CHECK: %[[ONE:.+]] = arith.constant 1.000000e+00 : f16
//       CHECK: %[[ADD:.+]] = arith.addf
//       CHECK: %[[CMP:.+]] = arith.cmpi
//       CHECK: arith.select %[[CMP]], %[[ADD]], %[[ONE]] : f16
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
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

// Reductions in multiple dimensions have a 2-D region to check for padding.
// Check for 2 compare ops, and an 'and' to combine them

// CHECK-LABEL: multi_dim_reduction
//  CHECK-SAME: (%[[ARG0:[0-9a-zA-Z]+]]: tensor<?x?xf16>, %
//   CHECK-DAG: %[[ZEROF16:.+]] = arith.constant 0.000000e+00 : f16
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG: %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf16>
//   CHECK-DAG: %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf16>
//       CHECK: linalg.generic
//   CHECK-DAG: %[[MUL:.+]] = arith.mulf
//   CHECK-DAG: %[[INDEX0:.+]] = linalg.index 0 : index
//   CHECK-DAG: %[[INDEX1:.+]] = linalg.index 1 : index
//   CHECK-DAG: %[[CMP0:.+]] = arith.cmpi ult, %[[INDEX0]], %[[DIM1]] : index
//   CHECK-DAG: %[[CMP1:.+]] = arith.cmpi ult, %[[INDEX1]], %[[DIM0]] : index
//       CHECK: %[[AND:.+]] = arith.andi %[[CMP0]], %[[CMP1]] : i1
//       CHECK: %[[SELECT:.+]] = arith.select %[[AND]], %[[MUL]], %[[ZEROF16]] : f16
//       CHECK: %[[ADD:.+]] = arith.addf %out, %[[SELECT]] : f16
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> ()>
func.func @multi_dim_reduction(%arg0 : tensor<?x?xf16>, %arg1 : tensor<?x?xf16>, %arg2 : tensor<f16>) -> tensor<f16> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2],
                       iterator_types = ["reduction", "reduction"]}
                       ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg2 : tensor<f16>)
    attrs =  {lowering_config = #iree_gpu.lowering_config<{partial_reduction = [4096, 4096]}>}
    {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %1 = arith.mulf %in, %in_0 : f16
      %2 = arith.addf %out, %1 : f16
      linalg.yield %2 : f16
    } -> tensor<f16>
  return %0 : tensor<f16>
}

// -----


// Multiple reductions in parallel in a linalg.generic op.

// CHECK-LABEL: minmax_reduction
//   CHECK-DAG: %[[NAN0:.+]] = arith.constant 0xFFC00000 : f32
//   CHECK-DAG: %[[NAN1:.+]] = arith.constant 0x7FC00000 : f32
//   CHECK-DAG: %[[SELECT0:.+]] = arith.select {{.*}} %[[NAN0]] : f32
//   CHECK-DAG: %[[SELECT0:.+]] = arith.select {{.*}} %[[NAN1]] : f32
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @minmax_reduction(%arg0: tensor<1x?xf32>, %arg1: tensor<1xf32>, %arg2 : tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) {
  %0:2 = linalg.generic {indexing_maps = [#map, #map1, #map1],
                       iterator_types = ["parallel", "reduction"]}
                       ins(%arg0 : tensor<1x?xf32>) outs(%arg1, %arg2 : tensor<1xf32>, tensor<1xf32>)
   attrs =  {lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 4096]}>}
   {
   ^bb0(%in: f32, %out0: f32, %out1 : f32):
     %1 = arith.minnumf %in, %out0 : f32
     %2 = arith.maxnumf %in, %out1 : f32
     linalg.yield %1, %2 : f32, f32
   } -> (tensor<1xf32>, tensor<1xf32>)
  return %0#0, %0#1: tensor<1xf32>, tensor<1xf32>
}
