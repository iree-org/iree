// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level{tiling-level=thread}, canonicalize, cse))" %s | FileCheck %s

#config = #iree_gpu.derived_thread_config
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @inferred_add_tensor(%3: tensor<64x256xf32>, %4: tensor<64x256xf32>, %5: tensor<64x256xf32>) -> tensor<64x256xf32>
      attributes {
        translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [16, 32, 1] subgroup_size = 64, {}>
      } {
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

// CHECK-LABEL: func.func @inferred_add_tensor
//       CHECK:   scf.forall ({{.*}}) = (0, 0) to (64, 256) step (1, 4)
//       CHECK:     linalg.generic {{.*}} ins(%{{.*}}: tensor<1x4xf32>, tensor<1x4xf32>)
//       CHECK:     scf.forall.in_parallel
//       CHECK:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

// -----

#config = #iree_gpu.derived_thread_config
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @inferred_dynamic(%3: tensor<?x?xf32>, %4: tensor<?x?xf32>, %5: tensor<?x?xf32>) -> tensor<?x?xf32>
      attributes {
        translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [16, 32, 1] subgroup_size = 64, {}>
      } {
    %6 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
      } ins(%3, %4 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) attrs =  {lowering_config = #config} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<?x?xf32>
    return %6 : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @inferred_dynamic
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[DIM0:.+]] = tensor.dim %[[A]], %c0 : tensor<?x?xf32>
//   CHECK-DAG:   %[[DIM1:.+]] = tensor.dim %[[A]], %c1 : tensor<?x?xf32>
//       CHECK:   scf.forall ({{.*}}) = (0, 0) to (%[[DIM0]], %[[DIM1]]) step (1, 4)
//       CHECK:     linalg.generic {{.*}} ins(%{{.*}}: tensor<1x?xf32>, tensor<1x?xf32>)
//       CHECK:     scf.forall.in_parallel
//       CHECK:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

// -----

#config = #iree_gpu.derived_thread_config
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @inferred_small_inner_dim(%3: tensor<8x2xf32>, %4: tensor<8x2xf32>, %5: tensor<8x2xf32>) -> tensor<8x2xf32>
      attributes {
        translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [16, 32, 1] subgroup_size = 64, {}>
      } {
    %6 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
      } ins(%3, %4 : tensor<8x2xf32>, tensor<8x2xf32>) outs(%5 : tensor<8x2xf32>) attrs =  {lowering_config = #config} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.addf %in, %in_0 : f32
      linalg.yield %7 : f32
    } -> tensor<8x2xf32>
    return %6 : tensor<8x2xf32>
  }
}

// CHECK-LABEL: func.func @inferred_small_inner_dim
//       CHECK:   scf.forall ({{.*}}) = (0, 0) to (8, 2) step (1, 2)
//       CHECK:     linalg.generic {{.*}} ins(%{{.*}}: tensor<1x2xf32>, tensor<1x2xf32>)
//       CHECK:     scf.forall.in_parallel
//       CHECK:   mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @inferred_small_inner_dim_fill_vector_sizes(%0: tensor<4x16x8x4x16x2x4xf16>, %1: tensor<4x16x8x4x16x2x4xf16>) -> tensor<4x16x8x4x16x2x4xf16>
      attributes {
        translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {}>
      } {
    %2 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config}
        ins(%0 : tensor<4x16x8x4x16x2x4xf16>)
        outs(%1 : tensor<4x16x8x4x16x2x4xf16>) -> tensor<4x16x8x4x16x2x4xf16>
    return %2 : tensor<4x16x8x4x16x2x4xf16>
  }
}

// CHECK-LABEL: func.func @inferred_small_inner_dim_fill_vector_sizes
//       CHECK:   scf.forall ({{.*}}) = (0, 0, 0, 0, 0, 0, 0) to (4, 16, 8, 4, 16, 2, 4) step (1, 1, 1, 1, 1, 2, 4)
//       CHECK:     linalg.copy{{.*}}ins({{.*}} : tensor<1x1x1x1x1x2x4xf16>) outs({{.*}} : tensor<1x1x1x1x1x2x4xf16>)
//       CHECK:     scf.forall.in_parallel
//       CHECK:   mapping = [#gpu.thread<linear_dim_6>, {{.*}}, #gpu.thread<linear_dim_0>]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @inferred_small_inner_dim_dont_fill_non_contiguous(
    %0: tensor<4x16x4x4xf16>, %1: tensor<4x16x4x4xf16>) -> tensor<4x16x4x4xf16>
      attributes {
        translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {}>
      } {
    %2 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config}
        ins(%0 : tensor<4x16x4x4xf16>)
        outs(%1 : tensor<4x16x4x4xf16>) -> tensor<4x16x4x4xf16>
    return %2 : tensor<4x16x4x4xf16>
  }
}

// CHECK-LABEL: func.func @inferred_small_inner_dim_dont_fill_non_contiguous
//       CHECK:   scf.forall ({{.*}}) = (0, 0, 0, 0) to (4, 16, 4, 4) step (1, 1, 1, 4)
//       CHECK:     linalg.copy{{.*}}ins({{.*}} : tensor<1x1x1x4xf16>) outs({{.*}} : tensor<1x1x1x4xf16>)
//       CHECK:     scf.forall.in_parallel
//       CHECK:   mapping = [#gpu.thread<linear_dim_3>, {{.*}}, #gpu.thread<linear_dim_0>]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @inferred_unaligned(%0: tensor<70xf16>, %1: tensor<70xf16>) -> tensor<70xf16>
      attributes {
        translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {}>
      } {
    %2 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config}
        ins(%0 : tensor<70xf16>)
        outs(%1 : tensor<70xf16>) -> tensor<70xf16>
    return %2 : tensor<70xf16>
  }
}

// CHECK-LABEL: func.func @inferred_unaligned
//       CHECK:   scf.forall ({{.*}}) = (0) to (70) step (8)
//       CHECK:     linalg.copy{{.*}}ins({{.*}} : tensor<?xf16>) outs({{.*}} : tensor<?xf16>)
//       CHECK:     scf.forall.in_parallel
//       CHECK:   mapping = [#gpu.thread<linear_dim_0>]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @inferred_smaller_load(%0: tensor<128xf16>, %1: tensor<128xf16>) -> tensor<128xf16>
      attributes {
        translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {}>
      } {
    %2 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config}
        ins(%0 : tensor<128xf16>)
        outs(%1 : tensor<128xf16>) -> tensor<128xf16>
    return %2 : tensor<128xf16>
  }
}

// CHECK-LABEL: func.func @inferred_smaller_load
//       CHECK:   scf.forall ({{.*}}) = (0) to (128) step (2)
//       CHECK:     linalg.copy{{.*}}ins({{.*}} : tensor<2xf16>) outs({{.*}} : tensor<2xf16>)
//       CHECK:     scf.forall.in_parallel
//       CHECK:   mapping = [#gpu.thread<linear_dim_0>]

// -----

#config = #iree_gpu.derived_thread_config
module {
  func.func @inferred_im2col(%2: tensor<2x34x34x128xf16>, %3: tensor<2x128x8xf16>) -> tensor<2x128x8xf16>
      attributes {
        translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [16, 32, 1] subgroup_size = 64, {}>
      } {
    %4 = iree_linalg_ext.im2col {lowering_config = #config}
      strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
      m_offset = [0] * [1] k_offset = [0] * [1]
      batch_pos = [0] m_pos = [2, 3] k_pos = [1]
      input_k_perm = [0, 1, 2]
      ins(%2 : tensor<2x34x34x128xf16>)
      outs(%3 : tensor<2x128x8xf16>) -> tensor<2x128x8xf16>
    return %4 : tensor<2x128x8xf16>
  }
}

// CHECK-LABEL: func.func @inferred_im2col
//       CHECK:   scf.forall ({{.*}}) = (0, 0, 0) to (2, 128, 8) step (1, 1, 4)
//       CHECK:     iree_linalg_ext.im2col {{.*}} ins(%{{.*}}: tensor<1x34x34x128xf16>) outs({{.*}}: tensor<1x1x4xf16>)
//       CHECK:     scf.forall.in_parallel
//       CHECK:   mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

// -----

#config = #iree_gpu.derived_thread_config
module {
  func.func @inferred_im2col_batch_last(%2: tensor<16x26x18x32xbf16>, %3: tensor<32x1x1x32xbf16>) -> tensor<32x1x1x32xbf16>
      attributes {
        translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {}>
      } {
    %4 = iree_linalg_ext.im2col {lowering_config = #config}
      strides = [1, 1] dilations = [1, 1] kernel_size = [24, 16]
      m_offset = [0, 0] * [3, 1] k_offset = [0] * [1]
      batch_pos = [3] m_pos = [1, 2] k_pos = [0]
      input_k_perm = [0, 1, 2]
      ins(%2 : tensor<16x26x18x32xbf16>)
      outs(%3 : tensor<32x1x1x32xbf16>) -> tensor<32x1x1x32xbf16>
    return %4 : tensor<32x1x1x32xbf16>
  }
}

// CHECK-LABEL: func.func @inferred_im2col_batch_last
//       CHECK:   scf.forall ({{.*}}) = (0, 0, 0, 0) to (32, 1, 1, 32) step (4, 1, 1, 1)
//       CHECK:     iree_linalg_ext.im2col {{.*}} ins(%{{.*}}: tensor<16x26x18x4xbf16>) outs({{.*}}: tensor<4x1x1x1xbf16>)
//       CHECK:     scf.forall.in_parallel
//       CHECK:   mapping = [#gpu.thread<linear_dim_3>, #gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

// -----

#config = #iree_gpu.derived_thread_config
func.func @scatter(%arg0: tensor<3x32x16xf32>, %arg1: tensor<3x1xi32>) -> tensor<3x32x16xf32>
      attributes {
        translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64, {}>
      } {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<3x32x16xf32>
  %1 = iree_linalg_ext.scatter {lowering_config = #config} dimension_map = [0] unique_indices(true)
    ins(%arg0, %arg1 : tensor<3x32x16xf32>, tensor<3x1xi32>) outs(%0 : tensor<3x32x16xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    iree_linalg_ext.yield %arg2 : f32
  } -> tensor<3x32x16xf32>
  return %1 : tensor<3x32x16xf32>
}

// CHECK-LABEL: func.func @scatter
//       CHECK:   scf.forall ({{.*}}) = (0, 0, 0) to (3, 32, 16) step (1, 1, 4)
//       CHECK:     linalg_ext.scatter
//       CHECK:     scf.forall.in_parallel
