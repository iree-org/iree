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
//  CHECK-DAG:   %[[NEG_INF:.+]] = arith.constant 0xFF800000 : f32
//  CHECK-DAG:   %[[FILL_IDX:.+]] = linalg.fill ins(%[[C0_i64]]
//  CHECK-DAG:   %[[FILL_VAL:.+]] = linalg.fill ins(%[[NEG_INF]]
//      CHECK:   %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic
// CHECK-SAME:      "some_ukernel"
// CHECK-SAME:       ins(%[[ARG0]] :
// CHECK-SAME:       outs(%[[FILL_VAL]], %[[FILL_IDX]] :
//      CHECK:   return %[[MICRO_KERNEL]]#1

// -----

#config = #iree_gpu.lowering_config<{
  ukernel = #iree_gpu.ukernel_config<name = "some_ukernel", def_attrs = {vm.import.module = "rocm"}>
}>

func.func @argmax_bf16i64_with_selected_ukernel(%arg0 : tensor<1x?xbf16>) -> tensor<1xi64> {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF80 : bf16
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xbf16>
  %3 = linalg.fill ins(%cst : bf16) outs(%2 : tensor<1xbf16>) -> tensor<1xbf16>
  %4:2 = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      }
      ins(%arg0 : tensor<1x?xbf16>)
      outs(%3, %1 : tensor<1xbf16>, tensor<1xi64>)
      attrs = {lowering_config = #config} {
  ^bb0(%in: bf16, %out: bf16, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : bf16
    %8 = arith.cmpf ogt, %in, %out : bf16
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : bf16, i64
  } -> (tensor<1xbf16>, tensor<1xi64>)
  return %4#1 : tensor<1xi64>
}

//CHECK-LABEL: func @argmax_bf16i64_with_selected_ukernel(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x?xbf16>
//  CHECK-DAG:   %[[C1_index:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C0_i64:.+]] = arith.constant 0
//  CHECK-DAG:   %[[NEG_INF:.+]] = arith.constant 0xFF80 : bf16
//  CHECK-DAG:   %[[FILL_IDX:.+]] = linalg.fill ins(%[[C0_i64]]
//  CHECK-DAG:   %[[FILL_VAL:.+]] = linalg.fill ins(%[[NEG_INF]]
//      CHECK:   %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic
// CHECK-SAME:      "some_ukernel"
// CHECK-SAME:       ins(%[[ARG0]] :
// CHECK-SAME:       outs(%[[FILL_VAL]], %[[FILL_IDX]] :
//      CHECK:   return %[[MICRO_KERNEL]]#1

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

func.func @argmax_invalid_index_without_selected_ukernel(
    %input: tensor<131072xf32>,
    %init_val_arg: tensor<f32>,
    %init_idx_arg: tensor<i64>
) -> tensor<i64> {
  %c0_i64 = arith.constant 0 : i64
  %cst_min = arith.constant 0xFF800000 : f32  // -inf
  %init_val = linalg.fill ins(%cst_min : f32)
              outs(%init_val_arg : tensor<f32>) -> tensor<f32>
  %init_idx = linalg.fill ins(%c0_i64 : i64)
              outs(%init_idx_arg : tensor<i64>) -> tensor<i64>

  // Argmax-style reduction with a matcher-breaking intermediate op (`addi`).
  %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    } ins(%input : tensor<131072xf32>)
      outs(%init_val, %init_idx : tensor<f32>, tensor<i64>) {
    ^bb0(%in: f32, %val: f32, %idx: i64):
      %i = linalg.index 0 : index
      %cast = arith.index_cast %i : index to i64
      // Breaks isArgmaxOp matching.
      %plus = arith.addi %cast, %c0_i64 : i64
      %maxval = arith.maximumf %in, %val : f32
      %cmp = arith.cmpf ogt, %in, %val : f32
      %sel = arith.select %cmp, %plus, %idx : i64
      linalg.yield %maxval, %sel : f32, i64
  } -> (tensor<f32>, tensor<i64>)

  return %result#1 : tensor<i64>
}

// CHECK-LABEL: func @argmax_invalid_index_without_selected_ukernel(
//      CHECK-NOT: iree_codegen.ukernel.generic
//      CHECK: linalg.generic

// -----

func.func @multi_mma_mfma_i32_16x16x32_i8(%a : tensor<1x2x8x1x1x2x8xi8>, %b : tensor<1x2x1x2x1x1x2x8xi8>, %c : tensor<1x1x1x8x2x1x1x4xi32>) -> tensor<1x1x1x8x2x1x1x4xi32> {
  %d = iree_gpu.multi_mma %a, %b, %c {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic =  MFMA_I32_16x16x32_I8, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2>,
    lowering_config = #iree_gpu.lowering_config<{
      reduction = [0, 0, 0],
      ukernel = #iree_gpu.ukernel_config<name = "iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8", def_attrs = {vm.import.module = "rocm"}, shared_memory_bytes = 16384>,
      workgroup = [1, 1, 0]}>
  } : tensor<1x2x8x1x1x2x8xi8>, tensor<1x2x1x2x1x1x2x8xi8> into tensor<1x1x1x8x2x1x1x4xi32>
  return %d : tensor<1x1x1x8x2x1x1x4xi32>
}

// CHECK-LABEL: func @multi_mma_mfma_i32_16x16x32_i8(
//       CHECK:   bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16384xi8>
//       CHECK:   iree_codegen.ukernel.generic
//  CHECK-SAME:      "iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8"

// -----

func.func @multi_mma_mfma_i32_16x16x32_i8_one_subgroup_no_shared_memory(%a : tensor<1x2x8x1x1x2x8xi8>, %b : tensor<1x2x1x2x1x1x2x8xi8>, %c : tensor<1x1x1x8x2x1x1x4xi32>) -> tensor<1x1x1x8x2x1x1x4xi32> {
  %d = iree_gpu.multi_mma %a, %b, %c {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic =  MFMA_I32_16x16x32_I8, intrinsics_m = 8, intrinsics_n = 2, intrinsics_k = 2>,
    lowering_config = #iree_gpu.lowering_config<{
      reduction = [0, 0, 0],
      ukernel = #iree_gpu.ukernel_config<name = "iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8", def_attrs = {vm.import.module = "rocm"}, shared_memory_bytes = 0>,
      workgroup = [1, 1, 0]}>
  } : tensor<1x2x8x1x1x2x8xi8>, tensor<1x2x1x2x1x1x2x8xi8> into tensor<1x1x1x8x2x1x1x4xi32>
  return %d : tensor<1x1x1x8x2x1x1x4xi32>
}

// CHECK-LABEL: func @multi_mma_mfma_i32_16x16x32_i8_one_subgroup_no_shared_memory(
//       CHECK:   iree_codegen.null_pointer
//       CHECK:   iree_codegen.ukernel.generic
//  CHECK-SAME:      "iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8"
