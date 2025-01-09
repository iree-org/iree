// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-lower-to-ukernels,cse,canonicalize))" %s | FileCheck %s

#config = #iree_gpu.lowering_config<{ukernel = #iree_gpu.ukernel_config<name = "some_ukernel", def_attrs = {vm.import.module = "rocm"}>}>
func.func @argmax_f32i64_with_selected_ukernel(%arg0 : tensor<1x?xf32>) -> tensor<1xi64> {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      }
      ins(%arg0 : tensor<1x?xf32>) outs(%3, %1 : tensor<1xf32>, tensor<1xi64>)
      attrs = {
        // The lowering_config.ukernel is what is essential to the lowering.
        lowering_config = #config} {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<1xf32>, tensor<1xi64>)
  return %4#1 : tensor<1xi64>
}

//CHECK-LABEL: func @argmax_f32i64_with_selected_ukernel(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x?xf32>
//  CHECK-DAG:   %[[C1_index:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C0_i64:.+]] = arith.constant 0
//  CHECK-DAG:   %[[FILL:.+]] = linalg.fill ins(%[[C0_i64]]
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic
//  CHECK-SAME:      "some_ukernel"
// CHECK-SAME:       ins(%[[ARG0]] :
// CHECK-SAME:       outs(%[[FILL]] :
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

func.func @argmax_f32i64_without_selected_ukernel(%arg0 : tensor<1x?xf32>) -> tensor<1xi64> {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      }
      ins(%arg0 : tensor<1x?xf32>) outs(%3, %1 : tensor<1xf32>, tensor<1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<1xf32>, tensor<1xi64>)
  return %4#1 : tensor<1xi64>
}

//CHECK-LABEL: func @argmax_f32i64_without_selected_ukernel(
//      CHECK-NOT: iree_codegen.ukernel.generic
//      CHECK: linalg.generic

// -----

func.func @multi_mma_mfma_i32_16x16x32_i8(%a : tensor<1x2x8x1x1x2x8xi8>, %b : tensor<1x2x1x2x1x1x2x8xi8>, %c : tensor<1x1x1x8x2x1x1x4xi32>) -> tensor<1x1x1x8x2x1x1x4xi32> {
  %d = iree_gpu.multi_mma %a, %b, %c {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic =  MFMA_I32_16x16x32_I8, unroll_m = 8, unroll_n = 2, subgroups_n = 4, unroll_k = 2>,
    lowering_config = #iree_gpu.lowering_config<{
      reduction = [0, 0, 0],
      ukernel = #iree_gpu.ukernel_config<name = "iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8", def_attrs = {vm.import.module = "rocm"}>,
      workgroup = [1, 1, 0]}>
  } : tensor<1x2x8x1x1x2x8xi8>, tensor<1x2x1x2x1x1x2x8xi8> into tensor<1x1x1x8x2x1x1x4xi32>
  return %d : tensor<1x1x1x8x2x1x1x4xi32>
}

// CHECK-LABEL: func @multi_mma_mfma_i32_16x16x32_i8(
//   CHECK-DAG:    %c2_i32 = arith.constant 2 : i32
//   CHECK-DAG:    %c8_i32 = arith.constant 8 : i32
//   CHECK-DAG:    %c1_i32 = arith.constant 1 : i32
//   CHECK-DAG:    %c4_i32 = arith.constant 4 : i32
//       CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic
//  CHECK-SAME:      "iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8"
//  CHECK-SAME:      (%c2_i32, %c8_i32, %c1_i32, %c2_i32, %c4_i32, %c2_i32 : i32, i32, i32, i32, i32, i32)
