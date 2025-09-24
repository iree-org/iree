// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-promote-matmul-operands))" | FileCheck %s

#lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1]}>

func.func @matmul(%a: tensor<32x1024xf32>, %b: tensor<1024x128xf32>) -> tensor<32x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<32x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
  %mm = linalg.matmul {lowering_config = #lowering_config}
    ins(%a, %b : tensor<32x1024xf32>, tensor<1024x128xf32>) outs(%fill : tensor<32x128xf32>) -> tensor<32x128xf32>
  return %mm : tensor<32x128xf32>
}

// CHECK-LABEL: func.func @matmul
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<32x1024xf32>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<1024x128xf32>
//       CHECK:   %[[PA:.+]] = linalg.copy {{.*}} ins(%[[A]] : tensor<32x1024xf32>)
//       CHECK:   %[[PB:.+]] = linalg.copy {{.*}} ins(%[[B]] : tensor<1024x128xf32>)
//       CHECK:   linalg.matmul {{.*}} ins(%[[PA]], %[[PB]] : tensor<32x1024xf32>, tensor<1024x128xf32>)

// -----

#lowering_config = #iree_gpu.lowering_config<{promote_operands = []}>

func.func @empty_config(%a: tensor<1x1024xf32>, %b: tensor<1024x128xf32>) -> tensor<1x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<1x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x128xf32>) -> tensor<1x128xf32>
  %mm = linalg.matmul {lowering_config = #lowering_config}
    ins(%a, %b : tensor<1x1024xf32>, tensor<1024x128xf32>) outs(%fill : tensor<1x128xf32>) -> tensor<1x128xf32>
  return %mm : tensor<1x128xf32>
}

// Verify that no copies are generated with an empty lowering config
// CHECK-LABEL: func.func @empty_config
//   CHECK-NOT:   linalg.copy
//       CHECK: return

// -----

#lowering_config = #iree_gpu.lowering_config<{promote_operands = [0]}>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @lhs_only_matmul(%a: tensor<32x1024xf32>, %b: tensor<1024x128xf32>) -> tensor<32x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<32x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
  %mm = linalg.generic {
    indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"],
    lowering_config = #lowering_config}
    ins(%a, %b : tensor<32x1024xf32>, tensor<1024x128xf32>) outs(%fill : tensor<32x128xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %7 = arith.mulf %in, %in_0 : f32
    %8 = arith.addf %out, %7 : f32
    linalg.yield %8 : f32
  } -> tensor<32x128xf32>
  return %mm : tensor<32x128xf32>
}

// CHECK-LABEL: func.func @lhs_only_matmul
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<32x1024xf32>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<1024x128xf32>
//       CHECK:   %[[PA:.+]] = linalg.copy {{.*}} ins(%[[A]] : tensor<32x1024xf32>)
//       CHECK:   linalg.generic {{.*}} ins(%[[PA]], %[[B]] : tensor<32x1024xf32>, tensor<1024x128xf32>)

// -----

#lowering_config = #iree_gpu.lowering_config<{promote_operands = [0]}>

func.func @no_promote_fill(%b: tensor<128x128xf32>) -> tensor<4x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x128xf32>) -> tensor<4x128xf32>
  %mm = linalg.matmul {lowering_config = #lowering_config}
    ins(%fill, %b : tensor<4x128xf32>, tensor<128x128xf32>) outs(%fill : tensor<4x128xf32>) -> tensor<4x128xf32>
  return %mm : tensor<4x128xf32>
}

// Verify that fills are not promoted.
// CHECK-LABEL: func.func @no_promote_fill
//   CHECK-NOT:   iree_gpu.derived_thread_config
//       CHECK: return

// -----

#lowering_config = #iree_gpu.lowering_config<{promote_operands = [0]}>

func.func @promote_pad(%a : tensor<4x127xf32>, %b: tensor<128x128xf32>) -> tensor<4x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x128xf32>) -> tensor<4x128xf32>
  %padded = tensor.pad %a low[0, 0] high[0, 1] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : f32
  } : tensor<4x127xf32> to tensor<4x128xf32>
  %mm = linalg.matmul {lowering_config = #lowering_config}
    ins(%padded, %b : tensor<4x128xf32>, tensor<128x128xf32>) outs(%fill : tensor<4x128xf32>) -> tensor<4x128xf32>
  return %mm : tensor<4x128xf32>
}

// Verify that pad is promoted with linalg.copy
// CHECK-LABEL: func.func @promote_pad
//   CHECK:   tensor.pad
//   CHECK:   linalg.copy
// CHECK-SAME: derived_thread_config
//       CHECK: return

// -----

#lowering_config = #iree_gpu.lowering_config<{promote_operands = [2]}>
func.func @promote_result(%a : tensor<?x?xf32>, %b : tensor<?x?xf32>, %mdim : index, %ndim : index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty(%mdim, %ndim) : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %mm = linalg.matmul {lowering_config = #lowering_config}
    ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %mm : tensor<?x?xf32>
}

// CHECK-LABEL: func @promote_result(
//       CHECK:   %[[MATMUL:.+]] = linalg.matmul
//       CHECK:   %[[ALLOC:.+]] = bufferization.alloc_tensor
//       CHECK:   %[[COPY1:.+]] = linalg.copy
//  CHECK-SAME:       ins(%[[MATMUL]] : tensor<?x?xf32>) outs(%[[ALLOC]] : tensor<?x?xf32>)
//  CHECK-SAME:       -> tensor<?x?xf32>
//       CHECK:   %[[COPY2:.+]] = linalg.copy
//  CHECK-SAME:       {lowering_config = #iree_gpu.derived_thread_config}
//  CHECK-SAME:       ins(%[[COPY1]] : tensor<?x?xf32>)
//       CHECK:   return %[[COPY2]] : tensor<?x?xf32>

// -----

#lowering_config = #iree_gpu.lowering_config<{promote_operands = [2]}>
func.func @promote_padded_result(%a : tensor<?x?xf32>, %b : tensor<?x?xf32>, %mdim : index, %ndim : index, %pad : index, %slice : index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty(%mdim, %ndim) : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %padded_fill = tensor.pad %fill low[0, 0] high[%pad, %pad] {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  %mm = linalg.matmul {lowering_config = #lowering_config}
    ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>) outs(%padded_fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  %mm_slice = tensor.extract_slice %mm [0, 0] [%slice, %slice] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %mm_slice : tensor<?x?xf32>
}

// CHECK-LABEL: func @promote_padded_result(
//       CHECK:   %[[MATMUL:.+]] = linalg.matmul
//       CHECK:   %[[ALLOC:.+]] = bufferization.alloc_tensor
//       CHECK:   %[[COPY1:.+]] = linalg.copy
//  CHECK-SAME:       ins(%[[MATMUL]] : tensor<?x?xf32>) outs(%[[ALLOC]] : tensor<?x?xf32>)
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[COPY1]]
//       CHECK:   %[[COPY2:.+]] = linalg.copy
//  CHECK-SAME:       {lowering_config = #iree_gpu.derived_thread_config}
//  CHECK-SAME:       ins(%[[EXTRACT]] : tensor<?x?xf32>)
//       CHECK:   return %[[COPY2]] : tensor<?x?xf32>

// -----

#lowering_config = #iree_gpu.lowering_config<{
  promote_operands = [0, 1],
  promotion_types = [#iree_gpu.derived_thread_config, #iree_gpu.use_global_load_dma]}>

func.func @matmul_global_load_dma(%a: tensor<32x1024xf32>, %b: tensor<1024x128xf32>) -> tensor<32x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<32x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
  %mm = linalg.matmul {lowering_config = #lowering_config}
    ins(%a, %b : tensor<32x1024xf32>, tensor<1024x128xf32>) outs(%fill : tensor<32x128xf32>) -> tensor<32x128xf32>
  return %mm : tensor<32x128xf32>
}

// CHECK-LABEL: func.func @matmul
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<32x1024xf32>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<1024x128xf32>
//       CHECK:   %[[PA:.+]] = linalg.copy
//  CHECK-SAME:     lowering_config = #iree_gpu.derived_thread_config
//  CHECK-SAME:     ins(%[[A]] : tensor<32x1024xf32>)
//       CHECK:   %[[PB:.+]] = linalg.copy
//  CHECK-SAME:     lowering_config = #iree_gpu.use_global_load_dma
//  CHECK-SAME:     ins(%[[B]] : tensor<1024x128xf32>)
//       CHECK:   linalg.matmul {{.*}} ins(%[[PA]], %[[PB]] : tensor<32x1024xf32>, tensor<1024x128xf32>)

// -----

#lowering_config = #iree_gpu.lowering_config<{
  promote_operands = [0, 1],
  promotion_types = [
    #iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>,
    #iree_gpu.promote_with_cache_swizzle<#iree_gpu.use_global_load_dma>]}>

func.func @promote_with_cache_swizzle(%a: tensor<2x34x34x128xf32>, %b: tensor<2x8x256xf32>) -> tensor<2x128x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<2x128x256xf32>
  %im2col_empty = tensor.empty() : tensor<2x128x8xf32>

  %im2col = iree_linalg_ext.im2col
    strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
    m_offset = [0] * [1] k_offset = [0] * [1]
    batch_pos = [0] m_pos = [2, 3] k_pos = [1]
    input_k_perm = [0, 1, 2]
    ins(%a : tensor<2x34x34x128xf32>)
    outs(%im2col_empty : tensor<2x128x8xf32>) -> tensor<2x128x8xf32>

  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<2x128x256xf32>) -> tensor<2x128x256xf32>
  %mm = linalg.batch_matmul {lowering_config = #lowering_config}
    ins(%im2col, %b : tensor<2x128x8xf32>, tensor<2x8x256xf32>) outs(%fill : tensor<2x128x256xf32>) -> tensor<2x128x256xf32>
  return %mm : tensor<2x128x256xf32>
}

// CHECK-LABEL: func.func @promote_with_cache_swizzle
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<2x34x34x128xf32>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<2x8x256xf32>
//   CHECK-DAG:   %[[SWIZZLE_A:.+]] = iree_gpu.buffer_resource_cast %[[A]] cacheSwizzleStride(%c128)
//   CHECK-DAG:   %[[SWIZZLE_B:.+]] = iree_gpu.buffer_resource_cast %[[B]] cacheSwizzleStride(%c256)
//       CHECK:   %[[PA:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     lowering_config = #iree_gpu.derived_thread_config
//  CHECK-SAME:     ins(%[[SWIZZLE_A]]
//       CHECK:   %[[PB:.+]] = linalg.copy
//  CHECK-SAME:     lowering_config = #iree_gpu.use_global_load_dma
//  CHECK-SAME:     ins(%[[SWIZZLE_B]]
//       CHECK:   linalg.batch_matmul {{.*}} ins(%[[PA]], %[[PB]]
