// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-llvmgpu-configure-tensor-layouts, canonicalize, cse))' %s | FileCheck %s

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                             subgroup_m_count = 1,
                                             subgroup_n_count = 1>}>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @matmul_96x64x16_mfma(%lhs: tensor<96x16xf16>,
                           %rhs: tensor<64x16xf16>,
                           %init: tensor<96x64xf32>)
                           -> tensor<96x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<96x16xf16>, tensor<64x16xf16>)
                        outs(%init: tensor<96x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<96x64xf32>
  return %out : tensor<96x64xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [3, 2], outer_tile = [1, 1], thread_tile = [32, 2], element_tile = [1, 4], subgroup_strides = [0, 0], thread_strides = [1, 32]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [2, 2], outer_tile = [1, 1], thread_tile = [32, 2], element_tile = [1, 4], subgroup_strides = [0, 0], thread_strides = [1, 32]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [3, 2], outer_tile = [4, 1], thread_tile = [2, 32], element_tile = [4, 1], subgroup_strides = [0, 0], thread_strides = [32, 1]>

// CHECK-LABEL: func.func @matmul_96x64x16_mfma

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]]) {shared_memory_conversion}
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]]) {shared_memory_conversion}
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<WMMA_F32_16x16x16_F16>,
                                             subgroup_m_count = 1,
                                             subgroup_n_count = 1>}>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @matmul_96x64x16_wmma(%lhs: tensor<96x16xf16>,
                           %rhs: tensor<64x16xf16>,
                           %init: tensor<96x64xf32>)
                           -> tensor<96x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<96x16xf16>, tensor<64x16xf16>)
                        outs(%init: tensor<96x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<96x64xf32>
  return %out : tensor<96x64xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 1], outer_tile = [1, 1], thread_tile = [16, 1], element_tile = [1, 16], subgroup_strides = [0, 0], thread_strides = [1, 0]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [4, 1], outer_tile = [1, 1], thread_tile = [16, 1], element_tile = [1, 16], subgroup_strides = [0, 0], thread_strides = [1, 0]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 4], outer_tile = [8, 1], thread_tile = [2, 16], element_tile = [1, 1], subgroup_strides = [0, 0], thread_strides = [16, 1]>

// CHECK-LABEL: func.func @matmul_96x64x16_wmma

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]]) {shared_memory_conversion}
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]]) {shared_memory_conversion}
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                             subgroup_m_count = 4,
                                             subgroup_n_count = 1>}>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @matmul_128x64x16_multi_subgroup(%lhs: tensor<128x16xf16>,
                                          %rhs: tensor<64x16xf16>,
                                          %init: tensor<128x64xf32>)
                                          -> tensor<128x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<128x16xf16>, tensor<64x16xf16>)
                        outs(%init: tensor<128x64xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<128x64xf32>
  return %out : tensor<128x64xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [4, 1]
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1]
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [4, 1]

// CHECK-LABEL: func.func @matmul_128x64x16_multi_subgroup

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]]) {shared_memory_conversion}
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]]) {shared_memory_conversion}
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<LLVMGPUVectorDistribute
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64,
      {mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                             subgroup_m_count = 2,
                                             subgroup_n_count = 2>}>

#maps = [
  affine_map<(bm, bn, m, n, k) -> (bm, m, k)>,
  affine_map<(bm, bn, m, n, k) -> (bn, n, k)>,
  affine_map<(bm, bn, m, n, k) -> (bm, m, bn, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
}

func.func @packed_matmul_128x128x128(%lhs: tensor<8x16x16xf16>,
                                     %rhs: tensor<8x16x16xf16>,
                                     %init: tensor<8x16x8x16xf32>)
                                     -> tensor<8x16x8x16xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<8x16x16xf16>, tensor<8x16x16xf16>)
                        outs(%init: tensor<8x16x8x16xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<8x16x8x16xf32>
  return %out : tensor<8x16x8x16xf32>
}


// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [2, 1, 1], batch_tile = [4, 1, 1], outer_tile = [1, 1, 1], thread_tile = [1, 16, 4], element_tile = [1, 1, 4], subgroup_strides = [2, 0, 0], thread_strides = [0, 1, 16]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [2, 1, 1], batch_tile = [4, 1, 1], outer_tile = [1, 1, 1], thread_tile = [1, 16, 4], element_tile = [1, 1, 4], subgroup_strides = [1, 0, 0], thread_strides = [0, 1, 16]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [2, 1, 2, 1], batch_tile = [4, 1, 4, 1], outer_tile = [1, 1, 1, 1], thread_tile = [1, 4, 1, 16], element_tile = [1, 4, 1, 1], subgroup_strides = [2, 0, 1, 0], thread_strides = [0, 16, 0, 1]>
// CHECK-LABEL: func.func @packed_matmul_128x128x128

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]]) {shared_memory_conversion}
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]]) {shared_memory_conversion}
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]
