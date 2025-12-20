// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-promote-matmul-operands,cse),canonicalize)" | FileCheck %s

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
//       CHECK:   %[[BARRIER:.+]] = iree_codegen.fusion_barrier %[[COPY1]]
//       CHECK:   %[[COPY2:.+]] = linalg.copy
//  CHECK-SAME:       {lowering_config = #iree_gpu.derived_thread_config}
//  CHECK-SAME:       ins(%[[BARRIER]] : tensor<?x?xf32>)
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
//       CHECK:   %[[BARRIER:.+]] = iree_codegen.fusion_barrier %[[COPY1]]
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[BARRIER]]
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

// CHECK-LABEL: func.func @matmul_global_load_dma
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
    input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
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
//   CHECK-DAG:   %[[SWIZZLE_A:.+]] = iree_gpu.buffer_resource_cast %[[A]] cacheSwizzleStride(%c512)
//   CHECK-DAG:   %[[SWIZZLE_B:.+]] = iree_gpu.buffer_resource_cast %[[B]] cacheSwizzleStride(%c1024)
//       CHECK:   %[[PA:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     lowering_config = #iree_gpu.derived_thread_config
//  CHECK-SAME:     ins(%[[SWIZZLE_A]]
//       CHECK:   %[[PB:.+]] = linalg.copy
//  CHECK-SAME:     lowering_config = #iree_gpu.use_global_load_dma
//  CHECK-SAME:     ins(%[[SWIZZLE_B]]
//       CHECK:   linalg.batch_matmul {{.*}} ins(%[[PA]], %[[PB]]


// -----

#lowering_config = #iree_gpu.lowering_config<{
  promote_operands = [0, 1],
  promotion_types = [
    #iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>,
    #iree_gpu.promote_with_cache_swizzle<#iree_gpu.use_global_load_dma>]}>

func.func @promote_with_cache_swizzle_f4(%a: tensor<2x34x34x128xf4E2M1FN>, %b: tensor<2x8x256xf4E2M1FN>) -> tensor<2x128x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<2x128x256xf32>
  %im2col_empty = tensor.empty() : tensor<2x128x8xf4E2M1FN>

  %im2col = iree_linalg_ext.im2col
    strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
    m_offset = [0] * [1] k_offset = [0] * [1]
    batch_pos = [0] m_pos = [2, 3] k_pos = [1]
    input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
    ins(%a : tensor<2x34x34x128xf4E2M1FN>)
    outs(%im2col_empty : tensor<2x128x8xf4E2M1FN>) -> tensor<2x128x8xf4E2M1FN>

  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<2x128x256xf32>) -> tensor<2x128x256xf32>
  %mm = linalg.batch_matmul {lowering_config = #lowering_config}
    ins(%im2col, %b : tensor<2x128x8xf4E2M1FN>, tensor<2x8x256xf4E2M1FN>) outs(%fill : tensor<2x128x256xf32>) -> tensor<2x128x256xf32>
  return %mm : tensor<2x128x256xf32>
}

// CHECK-LABEL: func.func @promote_with_cache_swizzle_f4
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<2x34x34x128xf4E2M1FN>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<2x8x256xf4E2M1FN>
//   CHECK-DAG:   %[[SWIZZLE_A:.+]] = iree_gpu.buffer_resource_cast %[[A]] cacheSwizzleStride(%c64)
//   CHECK-DAG:   %[[SWIZZLE_B:.+]] = iree_gpu.buffer_resource_cast %[[B]] cacheSwizzleStride(%c128)
//       CHECK:   %[[PA:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     lowering_config = #iree_gpu.derived_thread_config
//  CHECK-SAME:     ins(%[[SWIZZLE_A]]
//       CHECK:   %[[PB:.+]] = linalg.copy
//  CHECK-SAME:     lowering_config = #iree_gpu.use_global_load_dma
//  CHECK-SAME:     ins(%[[SWIZZLE_B]]
//       CHECK:   linalg.batch_matmul {{.*}} ins(%[[PA]], %[[PB]]

// -----
#lowering_config = #iree_gpu.lowering_config<{
  promote_operands = [0, 1],
  promotion_types = [
    #iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>,
    #iree_gpu.promote_with_cache_swizzle<#iree_gpu.use_global_load_dma>]}>

func.func @promote_with_cache_swizzle_f4_no_stride(%a: tensor<2x34x34x129xf4E2M1FN>, %b: tensor<2x8x256xf4E2M1FN>) -> tensor<2x129x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<2x129x256xf32>
  %im2col_empty = tensor.empty() : tensor<2x129x8xf4E2M1FN>

  %im2col = iree_linalg_ext.im2col
    strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
    m_offset = [0] * [1] k_offset = [0] * [1]
    batch_pos = [0] m_pos = [2, 3] k_pos = [1]
    input_k_perm = [0, 1, 2] output_perm = [0, 1, 2]
    ins(%a : tensor<2x34x34x129xf4E2M1FN>)
    outs(%im2col_empty : tensor<2x129x8xf4E2M1FN>) -> tensor<2x129x8xf4E2M1FN>

  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<2x129x256xf32>) -> tensor<2x129x256xf32>
  %mm = linalg.batch_matmul {lowering_config = #lowering_config}
    ins(%im2col, %b : tensor<2x129x8xf4E2M1FN>, tensor<2x8x256xf4E2M1FN>) outs(%fill : tensor<2x129x256xf32>) -> tensor<2x129x256xf32>
  return %mm : tensor<2x129x256xf32>
}

// CHECK-LABEL: func.func @promote_with_cache_swizzle_f4_no_stride
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<2x34x34x129xf4E2M1FN>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<2x8x256xf4E2M1FN>
//   CHECK-DAG:   %[[SWIZZLE_A:.+]] = iree_gpu.buffer_resource_cast %[[A]] cacheSwizzleStride(%c0)
//   CHECK-DAG:   %[[SWIZZLE_B:.+]] = iree_gpu.buffer_resource_cast %[[B]] cacheSwizzleStride(%c128)
//       CHECK:   %[[PA:.+]] = iree_linalg_ext.im2col
//  CHECK-SAME:     lowering_config = #iree_gpu.derived_thread_config
//  CHECK-SAME:     ins(%[[SWIZZLE_A]]
//       CHECK:   %[[PB:.+]] = linalg.copy
//  CHECK-SAME:     lowering_config = #iree_gpu.use_global_load_dma
//  CHECK-SAME:     ins(%[[SWIZZLE_B]]
//       CHECK:   linalg.batch_matmul {{.*}} ins(%[[PA]], %[[PB]]

// -----

#lowering_config = #iree_gpu.lowering_config<{
  promote_operands = [0, 1],
  promotion_types = [
    #iree_gpu.swizzle_operand<copy_config = #iree_gpu.use_global_load_dma, swizzle = #iree_codegen.xor_shuffle<128, 16>>,
    #iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config, swizzle = #iree_codegen.xor_shuffle<256, 32>>]}>

func.func @promote_with_swizzle_operand(%a: tensor<32x64xf32>, %b: tensor<64x128xf32>) -> tensor<32x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<32x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
  %mm = linalg.matmul {lowering_config = #lowering_config}
    ins(%a, %b : tensor<32x64xf32>, tensor<64x128xf32>) outs(%fill : tensor<32x128xf32>) -> tensor<32x128xf32>
  return %mm : tensor<32x128xf32>
}

// SwizzleOperand attribute creates swizzle_hint op with xor_shuffle
// and flattens/expands the tensor for shared memory swizzling.
// CHECK-LABEL: func.func @promote_with_swizzle_operand
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<32x64xf32>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<64x128xf32>
//       CHECK:   %[[EMPTY_A:.+]] = tensor.empty() : tensor<2048xf32>
//       CHECK:   %[[SWIZZLE_A:.+]] = iree_codegen.swizzle_hint %[[EMPTY_A]][#iree_codegen.xor_shuffle<128, 16>] : tensor<2048xf32>
//       CHECK:   %[[EXPAND_A:.+]] = tensor.expand_shape %[[SWIZZLE_A]] {{\[\[}}0, 1{{\]\]}} output_shape [32, 64] : tensor<2048xf32> into tensor<32x64xf32>
//       CHECK:   %[[COPY_A:.+]] = linalg.copy
//  CHECK-SAME:     lowering_config = #iree_gpu.use_global_load_dma
//  CHECK-SAME:     ins(%[[A]] : tensor<32x64xf32>) outs(%[[EXPAND_A]] : tensor<32x64xf32>)
//       CHECK:   %[[EMPTY_B:.+]] = tensor.empty() : tensor<8192xf32>
//       CHECK:   %[[SWIZZLE_B:.+]] = iree_codegen.swizzle_hint %[[EMPTY_B]][#iree_codegen.xor_shuffle<256, 32>] : tensor<8192xf32>
//       CHECK:   %[[EXPAND_B:.+]] = tensor.expand_shape %[[SWIZZLE_B]] {{\[\[}}0, 1{{\]\]}} output_shape [64, 128] : tensor<8192xf32> into tensor<64x128xf32>
//       CHECK:   %[[COPY_B:.+]] = linalg.copy
//  CHECK-SAME:     lowering_config = #iree_gpu.derived_thread_config
//  CHECK-SAME:     ins(%[[B]] : tensor<64x128xf32>) outs(%[[EXPAND_B]] : tensor<64x128xf32>)
//       CHECK:   linalg.matmul {{.*}} ins(%[[COPY_A]], %[[COPY_B]] : tensor<32x64xf32>, tensor<64x128xf32>)

// -----

#lowering_config = #iree_gpu.lowering_config<{
  promote_operands = [1],
  promotion_types = [
    #iree_gpu.swizzle_operand<copy_config = #iree_gpu.use_global_load_dma, swizzle = #iree_codegen.xor_shuffle<64, 8>>]}>

func.func @promote_with_swizzle_operand_f16(%a: tensor<32x64xf16>, %b: tensor<64x128xf16>) -> tensor<32x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<32x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x128xf32>) -> tensor<32x128xf32>
  %mm = linalg.matmul {lowering_config = #lowering_config}
    ins(%a, %b : tensor<32x64xf16>, tensor<64x128xf16>) outs(%fill : tensor<32x128xf32>) -> tensor<32x128xf32>
  return %mm : tensor<32x128xf32>
}

// SwizzleOperand with f16 element type.
// CHECK-LABEL: func.func @promote_with_swizzle_operand_f16
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<32x64xf16>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<64x128xf16>
//       CHECK:   %[[EMPTY_B:.+]] = tensor.empty() : tensor<8192xf16>
//       CHECK:   %[[SWIZZLE_B:.+]] = iree_codegen.swizzle_hint %[[EMPTY_B]][#iree_codegen.xor_shuffle<64, 8>] : tensor<8192xf16>
//       CHECK:   %[[EXPAND_B:.+]] = tensor.expand_shape %[[SWIZZLE_B]] {{\[\[}}0, 1{{\]\]}} output_shape [64, 128] : tensor<8192xf16> into tensor<64x128xf16>
//       CHECK:   %[[COPY_B:.+]] = linalg.copy
//  CHECK-SAME:     lowering_config = #iree_gpu.use_global_load_dma
//  CHECK-SAME:     ins(%[[B]] : tensor<64x128xf16>) outs(%[[EXPAND_B]] : tensor<64x128xf16>)
//       CHECK:   linalg.matmul {{.*}} ins(%[[A]], %[[COPY_B]] : tensor<32x64xf16>, tensor<64x128xf16>)

// -----

#lowering_config = #iree_gpu.lowering_config<{
  promote_operands = [0],
  promotion_types = [
    #iree_gpu.swizzle_operand<copy_config = #iree_gpu.use_global_load_dma, swizzle = #iree_codegen.xor_shuffle<128, 16>>]}>

func.func @swizzle_operand_no_promote_fill(%b: tensor<128x128xf32>) -> tensor<4x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<4x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x128xf32>) -> tensor<4x128xf32>
  %mm = linalg.matmul {lowering_config = #lowering_config}
    ins(%fill, %b : tensor<4x128xf32>, tensor<128x128xf32>) outs(%fill : tensor<4x128xf32>) -> tensor<4x128xf32>
  return %mm : tensor<4x128xf32>
}

// Verify that fills are not promoted even with swizzle_operand.
// CHECK-LABEL: func.func @swizzle_operand_no_promote_fill
//   CHECK-NOT:   iree_codegen.swizzle_hint
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   linalg.matmul
//       CHECK: return

// -----

#lowering_config = #iree_gpu.lowering_config<{
  promote_operands = [0, 1],
  promotion_types = [
    #iree_gpu.swizzle_operand<copy_config = #iree_gpu.use_global_load_dma, swizzle = #iree_codegen.xor_shuffle<128, 16>>,
    #iree_gpu.swizzle_operand<copy_config = #iree_gpu.derived_thread_config, swizzle = #iree_codegen.xor_shuffle<256, 32>>]}>

func.func @promote_with_multiple_swizzle_operand(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %mm = linalg.matmul {lowering_config = #lowering_config}
    ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>) outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %mm : tensor<64x64xf32>
}

// SwizzleOperand attribute creates swizzle_hint op with xor_shuffle
// and flattens/expands the tensor for shared memory swizzling.
// CHECK-LABEL: func.func @promote_with_multiple_swizzle_operand
//       CHECK:   %[[EMPTY_A:.+]] = tensor.empty() : tensor<4096xf32>
//       CHECK:   %[[SWIZZLE_A:.+]] = iree_codegen.swizzle_hint %[[EMPTY_A]][#iree_codegen.xor_shuffle<128, 16>] : tensor<4096xf32>
//       CHECK:   %[[SWIZZLE_B:.+]] = iree_codegen.swizzle_hint %[[EMPTY_A]][#iree_codegen.xor_shuffle<256, 32>] : tensor<4096xf32>

