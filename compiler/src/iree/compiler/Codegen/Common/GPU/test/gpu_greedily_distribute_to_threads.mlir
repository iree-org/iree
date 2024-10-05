// RUN: iree-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-greedily-distribute-to-threads, canonicalize, cse))" | \
// RUN:   FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @add_tensor(%3: tensor<64x256xf32>, %4: tensor<64x256xf32>, %5: tensor<64x256xf32>) -> tensor<64x256xf32>
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

// CHECK-LABEL: func.func @add_tensor
//       CHECK:   scf.forall
//       CHECK:     linalg.generic
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
//       CHECK:     linalg.fill
//       CHECK:     linalg.matmul

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
    %extracted_slice_0 = tensor.extract_slice %4[%arg0, 0] [8, 64] [1, 1] : tensor<64x64xf32> to tensor<8x64xf32>
    %7 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<64x8xf32>, tensor<8x64xf32>) outs(%arg1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    scf.yield %7 : tensor<64x64xf32>
  }
  return %6 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @in_nested_region
//       CHECK:   scf.for
//       CHECK:     scf.forall
//       CHECK:       linalg.matmul
