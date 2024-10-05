// RUN: iree-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-greedily-distribute-to-threads, canonicalize, cse))" | \
// RUN:   FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @simple_generic(%3: tensor<64x256xf32>, %4: tensor<64x256xf32>, %5: tensor<64x256xf32>) -> tensor<64x256xf32>
    attributes {
      translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {}>
    } {
  %6 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]
    } ins(%3, %4 : tensor<64x256xf32>, tensor<64x256xf32>) outs(%5 : tensor<64x256xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %7 = arith.addf %in, %in_0 : f32
    linalg.yield %7 : f32
  } -> tensor<64x256xf32>
  return %6 : tensor<64x256xf32>
}

// CHECK-LABEL: func.func @simple_generic
//       CHECK:   scf.forall
//       CHECK:     linalg.generic {{.*}} outs({{.*}}: tensor<1x4xf32>)
//       CHECK:     scf.forall.in_parallel
//       CHECK:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @fuse_destination(%3: tensor<64x64xf32>, %4: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {
      translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {}>
    } {
  %empty = tensor.empty() : tensor<64x64xf32>
  %cst = arith.constant 0.0 : f32
  %5 = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %7 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @fuse_destination
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<64x64xf32>
//       CHECK:   scf.forall {{.*}} shared_outs(%[[INIT:.+]] = %[[EMPTY]]
//       CHECK:     linalg.fill {{.*}} -> tensor<1x1xf32>

// Additionally verify that reduction dimensions do not get tiled.
//       CHECK:     linalg.matmul ins({{.*}}: tensor<1x64xf32>, tensor<64x1xf32>)

// -----

func.func @in_nested_region(%3: tensor<64x64xf32>, %4: tensor<64x64xf32>, %5: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {
      translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {}>
    } {
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %6 = scf.for %arg0 = %c0 to %c64 step %c8 iter_args(%arg1 = %5) -> (tensor<64x64xf32>) {
    %extracted_slice = tensor.extract_slice %3[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    %extracted_slice_0 = tensor.extract_slice %4[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    %7 = linalg.add
      ins(%extracted_slice, %extracted_slice_0 : tensor<64x8xf32>, tensor<64x8xf32>)
      outs(%extracted_slice_1 : tensor<64x8xf32>) -> tensor<64x8xf32>
    %insert = tensor.insert_slice %7 into %arg1[0, %arg0] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<64x64xf32>
    scf.yield %insert : tensor<64x64xf32>
  }
  return %6 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @in_nested_region
//       CHECK:   scf.for
//       CHECK:     scf.forall
//       CHECK:       linalg.add {{.*}} -> tensor<1x4xf32>

// -----

func.func @do_not_redistribute_in_forall(%3: tensor<64x64xf32>, %4: tensor<64x64xf32>, %5: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {
      translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {}>
    } {
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %6 = scf.for %arg0 = %c0 to %c64 step %c8 iter_args(%arg1 = %5) -> (tensor<64x64xf32>) {
    %extracted_slice = tensor.extract_slice %3[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    %extracted_slice_0 = tensor.extract_slice %4[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    %1 = scf.forall (%arg5, %arg6) = (0, 0) to (64, 8) step (1, 4) shared_outs(%arg7 = %extracted_slice_1) -> (tensor<64x8xf32>) {
      %extracted_slice_2 = tensor.extract_slice %extracted_slice[%arg5, %arg6] [1, 4] [1, 1] : tensor<64x8xf32> to tensor<1x4xf32>
      %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg5, %arg6] [1, 4] [1, 1] : tensor<64x8xf32> to tensor<1x4xf32>
      %extracted_slice_4 = tensor.extract_slice %arg7[%arg5, %arg6] [1, 4] [1, 1] : tensor<64x8xf32> to tensor<1x4xf32>
      %2 = linalg.add ins(%extracted_slice_2, %extracted_slice_3 : tensor<1x4xf32>, tensor<1x4xf32>) outs(%extracted_slice_4 : tensor<1x4xf32>) -> tensor<1x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %2 into %arg7[%arg5, %arg6] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<64x8xf32>
      }
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
    %7 = linalg.add
      ins(%1, %1 : tensor<64x8xf32>, tensor<64x8xf32>)
      outs(%1 : tensor<64x8xf32>) -> tensor<64x8xf32>
    %insert = tensor.insert_slice %7 into %arg1[0, %arg0] [64, 8] [1, 1] : tensor<64x8xf32> into tensor<64x64xf32>
    scf.yield %insert : tensor<64x64xf32>
  }
  %8 = linalg.add
    ins(%6, %6 : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%6 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %8 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @do_not_redistribute_in_forall
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<64x64xf32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<64x64xf32>
//  CHECK-SAME:   %[[ARG2:[A-Za-z0-9]+]]: tensor<64x64xf32>
//       CHECK:   scf.for {{.*}} iter_args(%[[FOR_ITER:.+]] = %[[ARG2]])
//       CHECK:     %[[INIT:.+]] = tensor.extract_slice %[[FOR_ITER]]

// Verify that the existing forall stays the same.
//       CHECK:     scf.forall {{.*}} shared_outs(%[[ITER:.+]] = %[[INIT]])
//       CHECK:       %[[DEST:.+]] = tensor.extract_slice %[[ITER]]
//       CHECK:       linalg.add {{.*}} outs(%[[DEST]] : tensor<1x4xf32>
//       CHECK:       scf.forall.in_parallel

//       CHECK:     %[[DIST_ADD:.+]] = scf.forall
//       CHECK:       linalg.add
//       CHECK:       scf.forall.in_parallel
//       CHECK:     %[[INSERT:.+]] = tensor.insert_slice %[[DIST_ADD]]
//       CHECK:     scf.yield %[[INSERT]]
//       CHECK:   %[[RES_ADD:.+]] = scf.forall
//       CHECK:     linalg.add
//       CHECK:     scf.forall.in_parallel
//       CHECK:   return %[[RES_ADD]]

// -----

func.func @multiple_use_tilable_op(%3: tensor<64x256xf32>, %4: tensor<64x256xf32>) -> (tensor<64x256xf32>, tensor<256x64xf32>)
    attributes {
      translation_info = #iree_codegen.translation_info<LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {}>
    } {
  %add_empty = tensor.empty() : tensor<64x256xf32>
  %6 = linalg.add
    ins(%3, %4 : tensor<64x256xf32>, tensor<64x256xf32>)
    outs(%add_empty : tensor<64x256xf32>) -> tensor<64x256xf32>
  %transpose_empty = tensor.empty() : tensor<256x64xf32>
  %7 = linalg.transpose
    ins(%6 : tensor<64x256xf32>)
    outs(%transpose_empty : tensor<256x64xf32>) permutation = [1, 0]
  return %6, %7 : tensor<64x256xf32>, tensor<256x64xf32>
}

// CHECK-LABEL: func.func @multiple_use_tilable_op
//       CHECK:   %[[ADD_DIST:.+]] = scf.forall
//       CHECK:     %[[ADD:.+]] = linalg.add {{.*}} -> tensor<1x4xf32>
//       CHECK:     scf.forall.in_parallel
//       CHECK:       tensor.parallel_insert_slice %[[ADD]]
//       CHECK:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]
//       CHECK:   %[[T_DIST:.+]] = scf.forall
//       CHECK:     %[[FUSED_ADD:.+]] = linalg.add {{.*}} -> tensor<4x1xf32>
//       CHECK:     %[[T:.+]] = linalg.transpose ins(%[[FUSED_ADD]]
//       CHECK:     scf.forall.in_parallel
//       CHECK:       tensor.parallel_insert_slice %[[T]]
//       CHECK:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]
//       CHECK:   return %[[ADD_DIST]], %[[T_DIST]]
