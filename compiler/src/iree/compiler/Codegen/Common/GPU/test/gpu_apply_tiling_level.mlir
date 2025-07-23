// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level{allow-zero-slices=false}, canonicalize, cse))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level, canonicalize, cse))" %s | FileCheck %s --check-prefix=NOZERO
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level{tiling-level=thread}, canonicalize, cse))" %s | FileCheck %s --check-prefix=THREAD
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level{tiling-level=subgroup}, canonicalize, cse))" %s | FileCheck %s --check-prefix=SUBGROUP
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level{tiling-level=partial_reduction}, canonicalize, cse))" %s | FileCheck %s --check-prefix=PARTRED
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level{normalize-loops}, canonicalize, cse))" %s | FileCheck %s --check-prefix=NORM-REDUCTION

#config = #iree_gpu.lowering_config<{thread = [2, 16], subgroup = [2, 16]}>
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_tensor(%3: tensor<64x256xf32>, %4: tensor<64x256xf32>, %5: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %6 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
      } ins(%3, %4 : tensor<64x256xf32>, tensor<64x256xf32>) outs(%5 : tensor<64x256xf32>) attrs =  {lowering_config = #config} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<64x256xf32>
    return %6 : tensor<64x256xf32>
  }
}

// Verify that no loops are generated without a reduction configuration.
// CHECK-LABEL: func.func @add_tensor
//   CHECK-NOT:   scf.for

// THREAD-LABEL: func.func @add_tensor
//       THREAD:   scf.forall ({{.*}}) = (0, 0) to (64, 256) step (2, 16)
//       THREAD:     linalg.generic {{.*}} ins(%{{.*}}: tensor<2x16xf32>, tensor<2x16xf32>)
//       THREAD:     scf.forall.in_parallel
//       THREAD:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

// SUBGROUP-LABEL: func.func @add_tensor
//       SUBGROUP:   scf.forall ({{.*}}) = (0, 0) to (64, 256) step (2, 16)
//       SUBGROUP:     linalg.generic {{.*}} ins(%{{.*}}: tensor<2x16xf32>, tensor<2x16xf32>)
//       SUBGROUP:     scf.forall.in_parallel
//       SUBGROUP:   mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]

// -----

#config = #iree_gpu.lowering_config<{thread = [0, 16]}>
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @sequential_forall_mappings(%3: tensor<4x256xf32>, %4: tensor<4x256xf32>, %5: tensor<4x256xf32>) -> tensor<4x256xf32> {
    %6 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
      } ins(%3, %4 : tensor<4x256xf32>, tensor<4x256xf32>) outs(%5 : tensor<4x256xf32>) attrs =  {lowering_config = #config} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<4x256xf32>
    return %6 : tensor<4x256xf32>
  }
}

// Verify that no loops are generated without a reduction configuration.
// CHECK-LABEL: func.func @sequential_forall_mappings
//   CHECK-NOT:   scf.for

// THREAD-LABEL: func.func @sequential_forall_mappings
//       THREAD:   scf.forall ({{.*}}) = (0) to (256) step (16)
//       THREAD:     linalg.generic {{.*}} ins(%{{.*}}: tensor<4x16xf32>, tensor<4x16xf32>)
//       THREAD:     scf.forall.in_parallel
//       THREAD:   mapping = [#gpu.thread<linear_dim_0>]

// -----

func.func @matmul_transpose_b(%5: tensor<64x64xf32>, %6: tensor<64x1280xf16>, %7: tensor<64x1280xf16>) -> tensor<64x64xf32> {
  %c4 = arith.constant 4 : index
  %c1280 = arith.constant 1280 : index
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %8 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %9 = tensor.empty() : tensor<64x1280xf16>
  %10 = tensor.empty() : tensor<64x1280xf16>
  %11 = scf.for %arg0 = %c0 to %c1280 step %c4 iter_args(%arg1 = %8) -> (tensor<64x64xf32>) {
    %extracted_slice = tensor.extract_slice %6[0, %arg0] [64, 4] [1, 1] : tensor<64x1280xf16> to tensor<64x4xf16>
    %extracted_slice_0 = tensor.extract_slice %9[0, %arg0] [64, 4] [1, 1] : tensor<64x1280xf16> to tensor<64x4xf16>
    %12 = linalg.copy {lowering_config = #iree_gpu.lowering_config<{thread = [1, 1]}>} ins(%extracted_slice : tensor<64x4xf16>) outs(%extracted_slice_0 : tensor<64x4xf16>) -> tensor<64x4xf16>
    %extracted_slice_1 = tensor.extract_slice %7[0, %arg0] [64, 4] [1, 1] : tensor<64x1280xf16> to tensor<64x4xf16>
    %extracted_slice_2 = tensor.extract_slice %10[0, %arg0] [64, 4] [1, 1] : tensor<64x1280xf16> to tensor<64x4xf16>
    %13 = linalg.copy {lowering_config = #iree_gpu.lowering_config<{thread = [1, 1]}>} ins(%extracted_slice_1 : tensor<64x4xf16>) outs(%extracted_slice_2 : tensor<64x4xf16>) -> tensor<64x4xf16>
    %14 = linalg.matmul
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ]
      {lowering_config = #iree_gpu.lowering_config<{thread = [4, 4]}>}
      ins(%12, %13 : tensor<64x4xf16>, tensor<64x4xf16>)
      outs(%arg1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    scf.yield %14 : tensor<64x64xf32>
  }
  return %11 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @matmul_transpose_b

// THREAD-LABEL: func.func @matmul_transpose_b
//       THREAD:   scf.forall ({{.*}}) in (64, 4)
//       THREAD:     linalg.copy
//       THREAD:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]
//       THREAD:   scf.forall ({{.*}}) in (64, 4)
//       THREAD:     linalg.copy
//       THREAD:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]
//       THREAD:   scf.forall ({{.*}}) = (0, 0) to (64, 64) step (4, 4)
//       THREAD:     linalg.matmul
//       THREAD:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

// -----

#config = #iree_gpu.lowering_config<{reduction = [0, 8]}>
#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
func.func @reduction(%3: tensor<128x384xf32>) -> tensor<128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<128xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
  %5 = linalg.generic {
    indexing_maps = [#map1, #map2],
    iterator_types = ["parallel", "reduction"]
    } ins(%3 : tensor<128x384xf32>) outs(%4 : tensor<128xf32>) attrs =  {lowering_config = #config} {
  ^bb0(%in: f32, %out: f32):
    %7 = arith.addf %in, %out : f32
    linalg.yield %7 : f32
  } -> tensor<128xf32>
  return %5 : tensor<128xf32>
}

// CHECK-LABEL: func.func @reduction
//       CHECK:   %[[FILL:.+]] = linalg.fill {{.*}} tensor<128xf32>
//       CHECK:   scf.for %{{.*}} = %c0 to %c384 step %c8 iter_args(%{{.*}} = %[[FILL]])
//       CHECK:     linalg.generic {{.*}} ins(%{{.*}} : tensor<128x8xf32>)
//       CHECK:     scf.yield

// Verify that no tiling happens in the thread case.
// THREAD-LABEL: func.func @reduction
//   THREAD-NOT:   scf.forall

// -----

#config = #iree_gpu.lowering_config<{reduction = [0, 0, 8]}>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_fuse(%3: tensor<64x64xf32>, %4: tensor<64x64xf32>, %5: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %cst = arith.constant 1.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %6 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]
    } ins(%3 : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %8 = arith.addf %in, %cst : f32
    linalg.yield %8 : f32
  } -> tensor<64x64xf32>
  %7 = linalg.matmul {lowering_config = #config} ins(%6, %4 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %7 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @matmul_fuse
//       CHECK:   scf.for %{{.*}} = %c0 to %c64 step %c8
//       CHECK:     %[[ELEMWISE:.+]] = linalg.generic {{.*}} ins(%{{.*}} : tensor<64x8xf32>)
//       CHECK:     %[[MM:.+]] = linalg.matmul {{.*}} ins(%[[ELEMWISE]], {{.*}} : tensor<64x8xf32>, tensor<8x64xf32>)

// -----

#config = #iree_gpu.lowering_config<{reduction = [0, 0, 8], thread = [8, 8, 0]}>
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul_fuse_destination(%3: tensor<64x64xf32>, %4: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %empty = tensor.empty() : tensor<64x64xf32>
  %cst = arith.constant 0.0 : f32
  %5 = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %7 = linalg.matmul {lowering_config = #config} ins(%3, %4 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %7 : tensor<64x64xf32>
}

// Verify that destinations are not fused for reduction tiling.
// CHECK-LABEL: func.func @matmul_fuse_destination
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%{{.*}} : tensor<64x64xf32>)
//       CHECK:   scf.for %{{.*}} = %c0 to %c64 step %c8 iter_args(%[[ITER:.+]] = %[[FILL]]
//       CHECK:     linalg.matmul

// THREAD-LABEL: func.func @matmul_fuse_destination
//       THREAD:   %[[EMPTY:.+]] = tensor.empty() : tensor<64x64xf32>
//       THREAD:   scf.forall {{.*}} shared_outs(%[[INIT:.+]] = %[[EMPTY]]
//       THREAD:     linalg.fill
//       THREAD:     linalg.matmul

// -----

#config = #iree_gpu.lowering_config<{thread = [8, 8]}>
func.func @matmul_cleanup(%3: tensor<64x64xf32>, %4: tensor<64x64xf32>, %5: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %6 = scf.for %arg0 = %c0 to %c64 step %c8 iter_args(%arg1 = %5) -> (tensor<64x64xf32>) {
    %extracted_slice = tensor.extract_slice %3[0, %arg0] [64, 8] [1, 1] : tensor<64x64xf32> to tensor<64x8xf32>
    %extracted_slice_0 = tensor.extract_slice %4[%arg0, 0] [8, 64] [1, 1] : tensor<64x64xf32> to tensor<8x64xf32>
    %7 = linalg.matmul {lowering_config = #config} ins(%extracted_slice, %extracted_slice_0 : tensor<64x8xf32>, tensor<8x64xf32>) outs(%arg1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    scf.yield %7 : tensor<64x64xf32>
  }
  return %6 : tensor<64x64xf32>
}

// THREAD-LABEL: func.func @matmul_cleanup
//  THREAD-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<64x64xf32>
//  THREAD-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<64x64xf32>
//       THREAD:   scf.for %{{.*}} = %c0 to %c64 step %c8
//       THREAD:     scf.forall
//   THREAD-DAG:       %[[LHS:.+]] = tensor.extract_slice %[[A]]
//   THREAD-DAG:       %[[RHS:.+]] = tensor.extract_slice %[[B]]
//       THREAD:       %[[MM:.+]] = linalg.matmul {{.*}} ins(%[[LHS]], %[[RHS]] : tensor<8x8xf32>, tensor<8x8xf32>)

// -----

#config = #iree_gpu.lowering_config<{reduction = [0, 0, 8], subgroup = [2, 4]}>
#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]

module {
  func.func @tensor_multi_mma_inner_tiled(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
    %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
      lowering_config = #config
    } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
    return %0 : tensor<?x?x4xf32>
  }
}

// CHECK-LABEL: func.func @tensor_multi_mma_inner_tiled
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<?x?x4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<?x?x4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: tensor<?x?x4xf32>

//   CHECK-DAG:   %[[MDIM:.+]] = tensor.dim %[[ACC]], %c0 : tensor<?x?x4xf32>
//   CHECK-DAG:   %[[NDIM:.+]] = tensor.dim %[[ACC]], %c1 : tensor<?x?x4xf32>
//   CHECK-DAG:   %[[KDIM:.+]] = tensor.dim %[[LHS]], %c1 : tensor<?x?x4xf16>
//       CHECK:   scf.for %[[I:.+]] = %c0 to %[[KDIM]] step %c8 iter_args(%[[INIT:.+]] = %[[ACC]])
//       CHECK:     %[[MIN:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 8)>(%[[I]])[%[[KDIM]]]
//       CHECK:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]]
//  CHECK-SAME:       [0, %[[I]], 0] [%[[MDIM]], %[[MIN]], 4] [1, 1, 1] : tensor<?x?x4xf16> to tensor<?x?x4xf16>
//       CHECK:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]]
//  CHECK-SAME:       [%[[I]], 0, 0] [%[[MIN]], %[[NDIM]], 4] [1, 1, 1] : tensor<?x?x4xf16> to tensor<?x?x4xf16>
//       CHECK:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[INIT]]
//  CHECK-SAME:       [0, 0, 0] [%[[MDIM]], %[[NDIM]], 4] [1, 1, 1] : tensor<?x?x4xf32> to tensor<?x?x4xf32>
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]]) {{.*}} lowering_config
//       CHECK:     tensor.insert_slice %[[MMA]] into %[[INIT]]
//  CHECK-SAME:       [0, 0, 0] [%[[MDIM]], %[[NDIM]], 4] [1, 1, 1]
//       CHECK:     scf.yield
//       CHECK:   return

// SUBGROUP-LABEL: func.func @tensor_multi_mma_inner_tiled
//  SUBGROUP-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<?x?x4xf16>
//  SUBGROUP-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<?x?x4xf16>
//  SUBGROUP-SAME:   %[[ACC:[A-Za-z0-9]+]]: tensor<?x?x4xf32>

//   SUBGROUP-DAG:   %[[MDIM:.+]] = tensor.dim %[[ACC]], %c0 : tensor<?x?x4xf32>
//   SUBGROUP-DAG:   %[[NDIM:.+]] = tensor.dim %[[ACC]], %c1 : tensor<?x?x4xf32>
//   SUBGROUP-DAG:   %[[KDIM:.+]] = tensor.dim %[[LHS]], %c1 : tensor<?x?x4xf16>

//       SUBGROUP:   scf.forall (%[[IDX:.+]], %[[IDY:.+]]) = (0, 0) to (%[[MDIM]], %[[NDIM]]) step (2, 4)
//  SUBGROUP-SAME:     shared_outs(%[[INIT:.+]] = %[[ACC]])
//   SUBGROUP-DAG:     %[[MMIN:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 2)>(%[[IDX]])[%[[MDIM]]]
//   SUBGROUP-DAG:     %[[NMIN:.+]] = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 4)>(%[[IDY]])[%[[NDIM]]]

//       SUBGROUP:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]]
//  SUBGROUP-SAME:       [%[[IDX]], 0, 0] [%[[MMIN]], %[[KDIM]], 4] [1, 1, 1] : tensor<?x?x4xf16> to tensor<?x?x4xf16>
//       SUBGROUP:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]]
//  SUBGROUP-SAME:       [0, %[[IDY]], 0] [%[[KDIM]], %[[NMIN]], 4] [1, 1, 1] : tensor<?x?x4xf16> to tensor<?x?x4xf16>
//       SUBGROUP:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[INIT]]
//  SUBGROUP-SAME:       [%[[IDX]], %[[IDY]], 0] [%[[MMIN]], %[[NMIN]], 4] [1, 1, 1] : tensor<?x?x4xf32> to tensor<?x?x4xf32>
//       SUBGROUP:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]]) {{.*}} lowering_config
//       SUBGROUP:     scf.forall.in_parallel
//       SUBGROUP:       tensor.parallel_insert_slice %[[MMA]] into %[[INIT]]
//       SUBGROUP:   return

// -----

// This test only checks when a tensor.pad gets fused when tiling. We disable
// tensor.pad fusion by default, because it generates a gaurd to prevent
// empty slices, which is hard to vectorize.
//
// However, if we already know no zero slices will be generated, we can fuse
// the pad directly.

#map = affine_map<()[s0] -> (s0 * -16 + 19, 16)>
#map1 = affine_map<()[s0] -> (-s0 + 16)>
module {
  func.func @fuse_pad_no_zero_slice(%arg0: tensor<?x17xf32>, %arg1: tensor<17x17xf32>, %arg2: index, %arg3: index) -> tensor<?x17xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = affine.min #map()[%arg2]
    %1 = tensor.empty() : tensor<16x32xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %3 = affine.apply #map1()[%0]
    %padded = tensor.pad %arg0 low[0, 0] high[%3, 7] {
    ^bb0(%arg4: index, %arg5: index):
      tensor.yield %cst : f32
    } : tensor<?x17xf32> to tensor<16x24xf32>
    %padded_0 = tensor.pad %arg1 low[0, 0] high[7, 15] {
    ^bb0(%arg4: index, %arg5: index):
      tensor.yield %cst : f32
    } : tensor<17x17xf32> to tensor<24x32xf32>
    %4 = linalg.matmul {lowering_config = #iree_gpu.lowering_config<{reduction = [0, 0, 8]}>} ins(%padded, %padded_0 : tensor<16x24xf32>, tensor<24x32xf32>) outs(%2 : tensor<16x32xf32>) -> tensor<16x32xf32>
    %extracted_slice = tensor.extract_slice %4[0, 0] [%0, 17] [1, 1] : tensor<16x32xf32> to tensor<?x17xf32>
    return %extracted_slice : tensor<?x17xf32>
  }
}

// Only fuse pad when no-zero-slices is true.

// CHECK-LABEL: @fuse_pad_no_zero_slice
// CHECK: tensor.pad
// CHECK: tensor.pad
// CHECK: scf.for
// CHECK-NOT: tensor.pad
// CHECK: linalg.matmul

// NOZERO-LABEL: @fuse_pad_no_zero_slice
// NOZERO-NOT: tensor.pad
// NOZERO: scf.for
// NOZERO: tensor.pad
// NOZERO: tensor.pad
// NOZERO: linalg.matmul

// -----

func.func @distribute_multi_result_generic(
  %arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<3x4xf32>) -> (tensor<3x4x5xf32>, tensor<3x4x5xf32>)
      attributes {
        translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 1, 1] subgroup_size = 32, {}>
      } {
  %empty = tensor.empty() : tensor<3x4x5xf32>
  %0:2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    ], iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1, %arg2 : tensor<3x4x5xf32>, tensor<3x4xf32>, tensor<3x4xf32>)
    outs(%empty, %empty : tensor<3x4x5xf32>, tensor<3x4x5xf32>)
    attrs =  {lowering_config = #iree_gpu.derived_thread_config} {
    ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32):
      %7 = arith.subf %in, %in_1 : f32
      %8 = math.exp %7 : f32
      %9 = arith.divf %8, %in_2 : f32
      linalg.yield %8, %9 : f32, f32
    } -> (tensor<3x4x5xf32>, tensor<3x4x5xf32>)
  return %0#0, %0#1 : tensor<3x4x5xf32>, tensor<3x4x5xf32>
}

// THREAD-LABEL: @distribute_multi_result_generic
//       THREAD:   %[[FORALL:.+]]:2 = scf.forall
//       THREAD:     linalg.generic
//  THREAD-SAME:       outs(%{{.*}}, %{{.*}}: tensor<1x1x?xf32>, tensor<1x1x?xf32>)
//       THREAD:   return %[[FORALL]]#0, %[[FORALL]]#1

//  -----

func.func @dont_yield_replacement_in_reduction_tiling(%arg0: tensor<4x77xbf16>, %arg1: tensor<4x77xf32>) -> (tensor<4x77xf32>, tensor<4xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ], iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<4x77xbf16>) outs(%arg1 : tensor<4x77xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %6 = arith.extf %in : bf16 to f32
      linalg.yield %6 : f32
    } -> tensor<4x77xf32>
  %1 = tensor.empty() : tensor<4xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<4xf32>) -> tensor<4xf32>
  %3 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ], iterator_types = ["parallel", "reduction"]}
    ins(%0 : tensor<4x77xf32>) outs(%2 : tensor<4xf32>)
    attrs =  {lowering_config = #iree_gpu.lowering_config<{reduction = [0, 7]}>} {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.maxnumf %in, %out : f32
      linalg.yield %6 : f32
    } -> tensor<4xf32>
  return %0, %3 : tensor<4x77xf32>, tensor<4xf32>
}

//  CHECK-LABEL: @dont_yield_replacement_in_reduction_tiling
//        CHECK: scf.for
// Note: if we yield replacement then we would see a large result of
// tensor<4x77xf32> also being yielded which is not what we want in such a case.
//   CHECK-SAME:  -> (tensor<4xf32>) {

// -----

#config = #iree_gpu.lowering_config<{thread = [2, 16], subgroup = [2, 16]}>
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @cleanup_slices(%3: tensor<260x70xf32>, %4: tensor<64x256xf32>, %5: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %empty = tensor.empty() : tensor<70x260xf32>
    %transpose = linalg.transpose ins(%3 : tensor<260x70xf32>) outs(%empty : tensor<70x260xf32>) permutation = [1, 0]
    %slice = tensor.extract_slice %transpose [0, 0] [64, 256] [1, 1] : tensor<70x260xf32> to tensor<64x256xf32>
    %6 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
      } ins(%slice, %4 : tensor<64x256xf32>, tensor<64x256xf32>) outs(%5 : tensor<64x256xf32>) attrs =  {lowering_config = #config} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<64x256xf32>
    return %6 : tensor<64x256xf32>
  }
}

// THREAD-LABEL: func.func @cleanup_slices
//       THREAD:   scf.forall ({{.*}}) = (0, 0) to (64, 256) step (2, 16)
//       THREAD:     linalg.transpose ins(%{{.*}}: tensor<16x2xf32>)
//       THREAD:     linalg.generic {{.*}} ins(%{{.*}}: tensor<2x16xf32>, tensor<2x16xf32>)
//       THREAD:     scf.forall.in_parallel
//       THREAD:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

// -----

#config = #iree_gpu.lowering_config<{thread = [1, 1, 5]}>
module {
  func.func @swap_expand_shape_with_extract_slice(%0: tensor<60xf32>) -> tensor<2x3x10xf32> {
    %expand = tensor.expand_shape %0 [[0, 1, 2]] output_shape [2, 3, 10] : tensor<60xf32> into tensor<2x3x10xf32>
    %empty = tensor.empty() : tensor<2x3x10xf32>
    %exp = linalg.exp {lowering_config = #config} ins(%expand : tensor<2x3x10xf32>) outs(%empty : tensor<2x3x10xf32>) -> tensor<2x3x10xf32>
    return %exp : tensor<2x3x10xf32>
  }
}

// THREAD-LABEL: func.func @swap_expand_shape_with_extract_slice
//       THREAD:   scf.forall (%[[X:[A-Za-z0-9]+]], %[[Y:[A-Za-z0-9]+]], %[[Z:[A-Za-z0-9]+]])
//       THREAD:     %[[LINEAR_IDX:.+]] = affine.linearize_index disjoint [%[[X]], %[[Y]], %[[Z]]] by (2, 3, 10)
//       THREAD:     %[[SLICE:.+]] = tensor.extract_slice %{{.*}}[%[[LINEAR_IDX]]] [5] [1] : tensor<60xf32> to tensor<5xf32>
//       THREAD:     %[[EXPAND:.+]] = tensor.expand_shape %[[SLICE]] {{\[\[}}0, 1, 2]] output_shape [1, 1, 5]
//       THREAD:     linalg.exp {{.*}} ins(%[[EXPAND]]

// -----

#config = #iree_gpu.lowering_config<{thread = [1, 2, 0]}>
module {
  func.func @swap_expand_shape_with_extract_slice_full_inner_dim(%0: tensor<120xf32>) -> tensor<3x4x10xf32> {
    %expand = tensor.expand_shape %0 [[0, 1, 2]] output_shape [3, 4, 10] : tensor<120xf32> into tensor<3x4x10xf32>
    %empty = tensor.empty() : tensor<3x4x10xf32>
    %exp = linalg.exp {lowering_config = #config} ins(%expand : tensor<3x4x10xf32>) outs(%empty : tensor<3x4x10xf32>) -> tensor<3x4x10xf32>
    return %exp : tensor<3x4x10xf32>
  }
}

// THREAD-LABEL: func.func @swap_expand_shape_with_extract_slice_full_inner_dim
//       THREAD:   %[[C0:.+]] = arith.constant 0 : index
//       THREAD:   scf.forall (%[[X:[A-Za-z0-9]+]], %[[Y:[A-Za-z0-9]+]])
//       THREAD:     %[[LINEAR_IDX:.+]] = affine.linearize_index disjoint [%[[X]], %[[Y]], %[[C0]]] by (3, 4, 10)
//       THREAD:     %[[SLICE:.+]] = tensor.extract_slice %{{.*}}[%[[LINEAR_IDX]]] [20] [1] : tensor<120xf32> to tensor<20xf32>
//       THREAD:     %[[EXPAND:.+]] = tensor.expand_shape %[[SLICE]] {{\[\[}}0, 1, 2]] output_shape [1, 2, 10]
//       THREAD:     linalg.exp {{.*}} ins(%[[EXPAND]]

// -----

#config = #iree_gpu.lowering_config<{thread = [1, 2, 5]}>
module {
  func.func @no_swap_expand_shape_with_extract_slice_non_contiguous(%0: tensor<120xf32>) -> tensor<3x4x10xf32> {
    %expand = tensor.expand_shape %0 [[0, 1, 2]] output_shape [3, 4, 10] : tensor<120xf32> into tensor<3x4x10xf32>
    %empty = tensor.empty() : tensor<3x4x10xf32>
    %exp = linalg.exp {lowering_config = #config} ins(%expand : tensor<3x4x10xf32>) outs(%empty : tensor<3x4x10xf32>) -> tensor<3x4x10xf32>
    return %exp : tensor<3x4x10xf32>
  }
}

// THREAD-LABEL: func.func @no_swap_expand_shape_with_extract_slice_non_contiguous
//       THREAD:   tensor.expand_shape
//       THREAD:   scf.forall
//       THREAD:     linalg.exp

// -----

#config = #iree_gpu.lowering_config<{thread = [1, 2, 0, 1, 4]}>
module {
  func.func @swap_expand_shape_with_extract_slice_multiple_expanded_dims(%0: tensor<120x56xf32>) -> tensor<3x4x10x7x8xf32> {
    %expand = tensor.expand_shape %0 [[0, 1, 2], [3, 4]] output_shape [3, 4, 10, 7, 8] : tensor<120x56xf32> into tensor<3x4x10x7x8xf32>
    %empty = tensor.empty() : tensor<3x4x10x7x8xf32>
    %exp = linalg.exp {lowering_config = #config}
      ins(%expand : tensor<3x4x10x7x8xf32>) outs(%empty : tensor<3x4x10x7x8xf32>) -> tensor<3x4x10x7x8xf32>
    return %exp : tensor<3x4x10x7x8xf32>
  }
}

// THREAD-LABEL: func.func @swap_expand_shape_with_extract_slice_multiple_expanded_dims
//       THREAD:   %[[C0:.+]] = arith.constant 0 : index
//       THREAD:   scf.forall (%[[ID0:[A-Za-z0-9]+]], %[[ID1:[A-Za-z0-9]+]], %[[ID2:[A-Za-z0-9]+]], %[[ID3:[A-Za-z0-9]+]])
//       THREAD:     %[[LINEAR_IDX0:.+]] = affine.linearize_index disjoint [%[[ID0]], %[[ID1]], %[[C0]]] by (3, 4, 10)
//       THREAD:     %[[LINEAR_IDX1:.+]] = affine.linearize_index disjoint [%[[ID2]], %[[ID3]]] by (7, 8)
//       THREAD:     %[[SLICE:.+]] = tensor.extract_slice %{{.*}}[%[[LINEAR_IDX0]], %[[LINEAR_IDX1]]] [20, 4] [1, 1]
//       THREAD:     %[[EXPAND:.+]] = tensor.expand_shape %[[SLICE]] {{\[\[}}0, 1, 2], [3, 4]] output_shape [1, 2, 10, 1, 4]
//       THREAD:     linalg.exp {{.*}} ins(%[[EXPAND]]

// -----

// Partial reduction tiling tests
#config = #iree_gpu.lowering_config<{partial_reduction = [0, 8]}>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
func.func @partial_reduction(%3: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0  = arith.constant 0 : index
  %par_dim = tensor.dim %3, %c0 : tensor<?x?xf32>
  %empty = tensor.empty(%par_dim) : tensor<?xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %5 = linalg.generic {
    indexing_maps = [#map1, #map2],
    iterator_types = ["parallel", "reduction"]
    } ins(%3 : tensor<?x?xf32>) outs(%4 : tensor<?xf32>) attrs =  {lowering_config = #config} {
  ^bb0(%in: f32, %out: f32):
    %7 = arith.addf %in, %out : f32
    linalg.yield %7 : f32
  } -> tensor<?xf32>
  return %5 : tensor<?xf32>
}

// We only check if the correct tiling implementation was used. We do not
// check if the tiling implementation itself is correct (it should be tested
// in the partial tiling unit tests).
// PARTRED-LABEL: func.func @partial_reduction
//   PARTRED-DAG:  %[[DIM0:.+]]  = tensor.dim %{{.*}}, %c0
//   PARTRED-DAG:  %[[DIM1:.+]]  = tensor.dim %{{.*}}, %c1
//   PARTRED-DAG:   %[[FULL:.+]] = linalg.fill {{.*}} tensor<?xf32>
//   PARTRED-DAG:   %[[PART:.+]] = linalg.fill {{.*}} tensor<?x8xf32>
//       PARTRED:   %[[OUT:.+]] = scf.for %{{.*}} = %c0 to %[[DIM1]] step %c8 iter_args(%{{.*}} = %[[PART]])
//       PARTRED:     linalg.generic
//  PARTRED-SAME:       iterator_types = ["parallel", "parallel"]
//  PARTRED-SAME:       ins(%{{.*}}  : tensor<?x?xf32>) outs(%{{.*}} : tensor<?x?xf32>)
//  PARTRED-SAME:       attrs = {lowering_config =
//       PARTRED:   scf.yield
//       PARTRED:   linalg.reduce ins(%[[OUT]] : tensor<?x8xf32>)
//  PARTRED-SAME:                 outs(%[[FULL]] : tensor<?xf32>)

// -----

#config = #iree_gpu.lowering_config<{reduction = [0, 32]}>
func.func @swap_collapse_shape_with_extract_slice(%arg0: tensor<32x3x3x288xf32>) -> tensor<32x2592xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<32x3x3x288xf32> into tensor<32x2592xf32>
  %empty = tensor.empty() : tensor<32x2592xf32>
  %0 = linalg.copy {lowering_config = #config} ins(%collapsed : tensor<32x2592xf32>) outs(%empty : tensor<32x2592xf32>) -> tensor<32x2592xf32>
  return %0: tensor<32x2592xf32>
}

// NORM-REDUCTION-LABEL: func.func @swap_collapse_shape_with_extract_slice
//   NORM-REDUCTION-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   NORM-REDUCTION-DAG:   %[[C81:.+]] = arith.constant 81 : index
//   NORM-REDUCTION-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       NORM-REDUCTION:   scf.for %[[ARG1:.+]] = %[[C0]] to %[[C81]] step %[[C1]]
//       NORM-REDUCTION:     %[[APPLY:.+]] = affine.apply affine_map<(d0) -> (d0 * 32)>(%[[ARG1]])
//       NORM-REDUCTION:     %[[IDX:.+]]:3 = affine.delinearize_index %[[APPLY]] into (3, 3, 288) : index, index, index
//       NORM-REDUCTION:     %[[SLICE:.+]] = tensor.extract_slice %{{.*}}[0, %[[IDX]]#0, %[[IDX]]#1, %[[IDX]]#2] [32, 1, 1, 32] [1, 1, 1, 1] : tensor<32x3x3x288xf32> to tensor<32x1x1x32xf32>
//       NORM-REDUCTION:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[SLICE]] {{\[}}[0], [1, 2, 3]] : tensor<32x1x1x32xf32> into tensor<32x32xf32>
//       NORM-REDUCTION:     linalg.copy {{.*}} ins(%[[COLLAPSE]]

// Without loop normalization, no swap would happen.
//                CHECK:   tensor.collapse_shape
//                CHECK:   scf.for
//                CHECK:     tensor.extract_slice
//            CHECK-NOT:     tensor.collapse_shape
//                CHECK:     linalg.copy

// -----

#config = #iree_gpu.lowering_config<{reduction = [0, 30]}>
func.func @no_swap_collapse_shape_with_extract_slice(%arg0: tensor<32x3x3x288xf32>) -> tensor<32x2592xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<32x3x3x288xf32> into tensor<32x2592xf32>
  %empty = tensor.empty() : tensor<32x2592xf32>
  %0 = linalg.copy {lowering_config = #config} ins(%collapsed : tensor<32x2592xf32>) outs(%empty : tensor<32x2592xf32>) -> tensor<32x2592xf32>
  return %0: tensor<32x2592xf32>
}

// No swap would happen when collapsed size is not divisible by offset multiplier.
// NORM-REDUCTION-LABEL: func.func @no_swap_collapse_shape_with_extract_slice
//       NORM-REDUCTION:   tensor.collapse_shape
//       NORM-REDUCTION:   scf.for
//       NORM-REDUCTION:     tensor.extract_slice
//   NORM-REDUCTION-NOT:     tensor.collapse_shape
//       NORM-REDUCTION:     linalg.copy

// -----

#config = #iree_gpu.lowering_config<{reduction = [0, 32]}>
func.func @no_swap_collapse_shape_with_extract_slice_2(%arg0: tensor<32x2x2x16xf32>) -> tensor<32x64xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<32x2x2x16xf32> into tensor<32x64xf32>
  %empty = tensor.empty() : tensor<32x64xf32>
  %0 = linalg.copy {lowering_config = #config} ins(%collapsed : tensor<32x64xf32>) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  return %0: tensor<32x64xf32>
}

// No swap would happen when the last expanded size is not divisible by collapse size.
// NORM-REDUCTION-LABEL: func.func @no_swap_collapse_shape_with_extract_slice_2
//       NORM-REDUCTION:   tensor.collapse_shape
//       NORM-REDUCTION:   scf.for
//       NORM-REDUCTION:     tensor.extract_slice
//   NORM-REDUCTION-NOT:     tensor.collapse_shape
//       NORM-REDUCTION:     linalg.copy
