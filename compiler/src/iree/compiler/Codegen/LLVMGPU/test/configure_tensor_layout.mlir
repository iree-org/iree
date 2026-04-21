// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-llvmgpu-configure-tensor-layouts, canonicalize, cse))' %s | FileCheck %s

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
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

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#scan_config = #iree_gpu.lowering_config<{
  subgroup_basis = [[1, 1], [0, 1]],
  lane_basis = [[1, 8], [0, 1]],
  thread = [1, 8]
}>

func.func @scan_dim1(%input: tensor<4x16xf32>,
                     %output: tensor<4x16xf32>,
                     %accum: tensor<4xf32>)
                     -> (tensor<4x16xf32>, tensor<4xf32>)
                     attributes {translation_info = #translation} {
  %result:2 = iree_linalg_ext.scan {lowering_config = #scan_config}
      dimension(1) inclusive(true)
      ins(%input : tensor<4x16xf32>)
      outs(%output, %accum : tensor<4x16xf32>, tensor<4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      iree_linalg_ext.yield %sum : f32
  } -> tensor<4x16xf32>, tensor<4xf32>
  return %result#0, %result#1 : tensor<4x16xf32>, tensor<4xf32>
}

// CHECK-DAG: #[[$SCAN_LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [4, 1], outer_tile = [1, 1], thread_tile = [1, 8], element_tile = [1, 8], subgroup_strides = [0, 0], thread_strides = [0, 1]>
// CHECK-DAG: #[[$SCAN_ACC_LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1], batch_tile = [4], outer_tile = [1], thread_tile = [1], element_tile = [1], subgroup_strides = [0], thread_strides = [0]>
// CHECK-LABEL: func.func @scan_dim1
// CHECK-DAG: %[[INPUT_LAYOUT:.+]] = iree_vector_ext.to_layout %arg0 to layout(#[[$SCAN_LAYOUT]]) : tensor<4x16xf32>
// CHECK-DAG: %[[OUTPUT_LAYOUT:.+]] = iree_vector_ext.to_layout %arg1 to layout(#[[$SCAN_LAYOUT]]) : tensor<4x16xf32>
// CHECK-DAG: %[[ACC_LAYOUT:.+]] = iree_vector_ext.to_layout %arg2 to layout(#[[$SCAN_ACC_LAYOUT]]) : tensor<4xf32>
// CHECK: %[[SCAN0:.+]]:2 = iree_linalg_ext.scan
// CHECK-SAME: ins(%[[INPUT_LAYOUT]] : tensor<4x16xf32>)
// CHECK-SAME: outs(%[[OUTPUT_LAYOUT]], %[[ACC_LAYOUT]] : tensor<4x16xf32>, tensor<4xf32>)
// CHECK-DAG: iree_vector_ext.to_layout %[[SCAN0]]#0 to layout(#[[$SCAN_LAYOUT]]) : tensor<4x16xf32>
// CHECK-DAG: iree_vector_ext.to_layout %[[SCAN0]]#1 to layout(#[[$SCAN_ACC_LAYOUT]]) : tensor<4xf32>

// -----

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_96x64x16_wmmar3(%lhs: tensor<96x16xf16>,
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

// CHECK-LABEL: func.func @matmul_96x64x16_wmmar3

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_96x64x16_wmmar4(%lhs: tensor<96x16xf16>,
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

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 1], outer_tile = [1, 1], thread_tile = [16, 2], element_tile = [1, 8], subgroup_strides = [0, 0], thread_strides = [1, 16]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [4, 1], outer_tile = [1, 1], thread_tile = [16, 2], element_tile = [1, 8], subgroup_strides = [0, 0], thread_strides = [1, 16]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 4], outer_tile = [1, 1], thread_tile = [2, 16], element_tile = [8, 1], subgroup_strides = [0, 0], thread_strides = [16, 1]>

// CHECK-LABEL: func.func @matmul_96x64x16_wmmar4

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [32, 1, 1]
                                              subgroup_size = 32>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<WMMA_F32_16x16x32_F16>,
                                              subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>
}

func.func @matmul_96x64x32_wmma_gfx1250(%lhs: tensor<96x32xf16>,
                                        %rhs: tensor<64x32xf16>,
                                        %init: tensor<96x64xf32>) -> tensor<96x64xf32>
                           attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<96x32xf16>, tensor<64x32xf16>)
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

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 1], outer_tile = [1, 1], thread_tile = [16, 2], element_tile = [1, 16], subgroup_strides = [0, 0], thread_strides = [1, 16]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [4, 1], outer_tile = [1, 1], thread_tile = [16, 2], element_tile = [1, 16], subgroup_strides = [0, 0], thread_strides = [1, 16]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1], batch_tile = [6, 4], outer_tile = [1, 1], thread_tile = [2, 16], element_tile = [8, 1], subgroup_strides = [0, 0], thread_strides = [16, 1]>

// CHECK-LABEL: func.func @matmul_96x64x32_wmma_gfx1250

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                              subgroup_basis = [[4, 1, 1], [0, 1, 2]]}>
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

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]

// -----

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

func.func @linalg_copy(%in : tensor<16x16x16xf16>) -> tensor<16x16x16xf16>
                      attributes { translation_info = #translation } {
  %empty = tensor.empty() : tensor<16x16x16xf16>
  %copied = linalg.copy
            { lowering_config = #iree_gpu.derived_thread_config }
            ins(%in : tensor<16x16x16xf16>)
            outs(%empty : tensor<16x16x16xf16>) -> tensor<16x16x16xf16>
  func.return %copied : tensor<16x16x16xf16>
}

// CHECK-DAG: #[[$LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1], batch_tile = [8, 1, 1], outer_tile = [1, 1, 1], thread_tile = [2, 16, 2], element_tile = [1, 1, 8], subgroup_strides = [0, 0, 0], thread_strides = [32, 2, 1]>

// CHECK-LABEL: func.func @linalg_copy
// CHECK: %[[OUT:.+]] = linalg.copy
// CHECK: to_layout %[[OUT]] to layout(#[[$LAYOUT]])

// -----

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>

#gather_trait = {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"],
    lowering_config = #iree_gpu.derived_thread_config
}

func.func @gather_like(%base : tensor<16384x16x32x128xf16>,
                       %indices : tensor<4x64x4xi64>)
                       -> tensor<4x64x4x16x32x128xf16>
                       attributes { translation_info = #translation } {

  %empty = tensor.empty() : tensor<4x64x4x16x32x128xf16>
  %gather = linalg.generic #gather_trait
            ins(%indices : tensor<4x64x4xi64>)
            outs(%empty : tensor<4x64x4x16x32x128xf16>) {
  ^bb0(%in: i64, %out: f16):
    %idx = arith.index_cast %in : i64 to index
    %iv3 = linalg.index 3 : index
    %iv4 = linalg.index 4 : index
    %iv5 = linalg.index 5 : index
    %extracted = tensor.extract %base[%idx, %iv3, %iv4, %iv5] : tensor<16384x16x32x128xf16>
    linalg.yield %extracted : f16
  } -> tensor<4x64x4x16x32x128xf16>

  func.return %gather : tensor<4x64x4x16x32x128xf16>
}

// CHECK-DAG: #[[$LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1, 1, 1, 1], batch_tile = [4, 64, 4, 16, 8, 1], outer_tile = [1, 1, 1, 1, 1, 1], thread_tile = [1, 1, 1, 1, 4, 16], element_tile = [1, 1, 1, 1, 1, 8], subgroup_strides = [0, 0, 0, 0, 0, 0], thread_strides = [0, 0, 0, 0, 16, 1]>

// CHECK-LABEL: func.func @gather_like
// CHECK: %[[OUT:.+]] = linalg.generic
// CHECK: to_layout %[[OUT]] to layout(#[[$LAYOUT]])

// -----

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

func.func @dynamic_infer_sizes(%in : tensor<4x32x?x128xf16>) -> tensor<1x1x?x128xf16> attributes { translation_info = #translation } {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %d2 = tensor.dim %in, %c2 : tensor<4x32x?x128xf16>
  %45 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 1024)>(%c0)[%d2]
  %extracted_slice_5 = tensor.extract_slice %in[%c0, %c0, %c0, 0] [1, 1, %45, 128] [1, 1, 1, 1] : tensor<4x32x?x128xf16> to tensor<1x1x?x128xf16>
  %49 = tensor.empty(%45) : tensor<1x1x?x128xf16>
  %50 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%extracted_slice_5 : tensor<1x1x?x128xf16>) outs(%49 : tensor<1x1x?x128xf16>) -> tensor<1x1x?x128xf16>
  return %50 : tensor<1x1x?x128xf16>
}

// CHECK-DAG: #[[LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1, 1], batch_tile = [1, 1, 256, 1], outer_tile = [1, 1, 1, 1], thread_tile = [1, 1, 4, 16], element_tile = [1, 1, 1, 8], subgroup_strides = [0, 0, 0, 0], thread_strides = [0, 0, 16, 1]>

// CHECK: %[[EXTRACT:.+]] = tensor.extract_slice %arg0{{.*}} : tensor<4x32x?x128xf16> to tensor<1x1x?x128xf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty({{.*}}) : tensor<1x1x?x128xf16>
// CHECK: %[[COPY:.+]] = linalg.copy {{.*}} ins(%[[EXTRACT]] : tensor<1x1x?x128xf16>) outs(%[[EMPTY]] : tensor<1x1x?x128xf16>)
// CHECK: iree_vector_ext.to_layout %[[COPY]] to layout(#[[LAYOUT]]) : tensor<1x1x?x128xf16>

// -----

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#lowering_config = #iree_gpu.lowering_config<{
    subgroup_basis = [[1, 1, 2, 2], [0, 1, 2, 3]],
    lane_basis = [[1, 1, 8, 8], [0, 1, 2, 3]],
    thread = [0, 0, 8, 8]
}>

func.func @dynamic_infer_sizes_lowering_config(%in : tensor<4x32x?x128xf16>) -> tensor<1x1x?x128xf16> attributes { translation_info = #translation } {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %d2 = tensor.dim %in, %c2 : tensor<4x32x?x128xf16>
  %45 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 128)>(%c0)[%d2]
  %extracted_slice_5 = tensor.extract_slice %in[%c0, %c0, %c0, 0] [1, 1, %45, 128] [1, 1, 1, 1] : tensor<4x32x?x128xf16> to tensor<1x1x?x128xf16>
  %49 = tensor.empty(%45) : tensor<1x1x?x128xf16>
  %50 = linalg.copy {lowering_config = #lowering_config} ins(%extracted_slice_5 : tensor<1x1x?x128xf16>) outs(%49 : tensor<1x1x?x128xf16>) -> tensor<1x1x?x128xf16>
  return %50 : tensor<1x1x?x128xf16>
}

// CHECK-DAG: #[[LAYOUT:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 2, 2], batch_tile = [1, 1, 1, 1], outer_tile = [1, 1, 1, 1], thread_tile = [1, 1, 8, 8], element_tile = [1, 1, 8, 8], subgroup_strides = [0, 0, 2, 1], thread_strides = [0, 0, 8, 1]>

// CHECK: %[[EXTRACT:.+]] = tensor.extract_slice %arg0{{.*}} : tensor<4x32x?x128xf16> to tensor<1x1x?x128xf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty({{.*}}) : tensor<1x1x?x128xf16>
// CHECK: %[[EXTRACTL:.+]] = iree_vector_ext.to_layout %[[EXTRACT]] to layout(#[[LAYOUT]]) : tensor<1x1x?x128xf16>
// CHECK: %[[EMPTYL:.+]] = iree_vector_ext.to_layout %[[EMPTY]] to layout(#[[LAYOUT]]) : tensor<1x1x?x128xf16>
// CHECK: %[[COPY:.+]] = linalg.copy {{.*}} ins(%[[EXTRACTL]] : tensor<1x1x?x128xf16>) outs(%[[EMPTYL]] : tensor<1x1x?x128xf16>)
// CHECK: iree_vector_ext.to_layout %[[COPY]] to layout(#[[LAYOUT]]) : tensor<1x1x?x128xf16>

// -----

// Verify that the batch tile for a dimension that requires ceil division
// (63 / 8 = 8, not 7) is computed correctly.

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [512, 1, 1]
                                              subgroup_size = 64>

#maps = [
  affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
  affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
  affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
]

#traits = {
  indexing_maps = #maps,
  iterator_types = ["parallel", "parallel", "reduction", "parallel"],
  lowering_config = #iree_gpu.lowering_config<{
    lane_basis = [[1, 1, 1, 1, 64], [1, 0, 3, 4]],
    subgroup_basis = [[1, 1, 1, 1, 8], [0, 1, 2, 4]],
    thread = [0, 0, 8, 0]
  }>
}

func.func @contraction_ceildiv_batch(%lhs: tensor<1x1x63xf16>,
                                     %rhs: tensor<1x512x63xf16>,
                                     %init: tensor<1x512x1xf32>)
                                     -> tensor<1x512x1xf32>
                                     attributes { translation_info = #translation } {
  %out = linalg.generic #traits
                        ins(%lhs, %rhs: tensor<1x1x63xf16>, tensor<1x512x63xf16>)
                        outs(%init: tensor<1x512x1xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %mul, %out : f32
      linalg.yield %sum : f32
  } -> tensor<1x512x1xf32>
  return %out : tensor<1x512x1xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<{{.*}}batch_tile = [1, 1, 8]{{.*}}element_tile = [1, 1, 8]{{.*}}>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<{{.*}}batch_tile = [1, 1, 8]{{.*}}element_tile = [1, 1, 8]{{.*}}>

// CHECK-LABEL: func.func @contraction_ceildiv_batch

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]

// -----

#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>

#maps_block = [
  affine_map<(b, m, n, k) -> (b, m, k)>,
  affine_map<(b, m, n, k) -> (b, k, n)>,
  affine_map<(b, m, n, k) -> (b, m, n)>
]

#traits_block = {
  indexing_maps = #maps_block,
  iterator_types = ["parallel", "parallel", "parallel", "reduction"],
  lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x4x2B_F16>,
                                              subgroup_basis = [[1, 1, 1, 1], [0, 1, 2, 3]]}>
}

func.func @batch_matmul_block_intrinsic(%lhs: tensor<4x32x4xf16>,
                                        %rhs: tensor<4x4x32xf16>,
                                        %init: tensor<4x32x32xf32>)
                                        -> tensor<4x32x32xf32>
                                        attributes { translation_info = #translation } {
  %out = linalg.generic #traits_block
                        ins(%lhs, %rhs: tensor<4x32x4xf16>, tensor<4x4x32xf16>)
                        outs(%init: tensor<4x32x32xf32>) {
    ^bb0(%in: f16, %in_1: f16, %out: f32):
      %ex   = arith.extf %in   : f16 to f32
      %ex_1 = arith.extf %in_1 : f16 to f32
      %mul  = arith.mulf %ex, %ex_1 : f32
      %sum  = arith.addf %out, %mul : f32
      linalg.yield %sum : f32
  } -> tensor<4x32x32xf32>
  return %out : tensor<4x32x32xf32>
}

// CHECK-DAG: #[[$NESTED:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1], batch_tile = [2, 1, 1], outer_tile = [1, 1, 1], thread_tile = [2, 32, 1], element_tile = [1, 1, 4], subgroup_strides = [0, 0, 0], thread_strides = [32, 1, 0]>
// CHECK-DAG: #[[$NESTED1:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1], batch_tile = [2, 1, 1], outer_tile = [1, 1, 1], thread_tile = [2, 1, 32], element_tile = [1, 4, 1], subgroup_strides = [0, 0, 0], thread_strides = [32, 0, 1]>
// CHECK-DAG: #[[$NESTED2:.+]] = #iree_vector_ext.nested_layout<subgroup_tile = [1, 1, 1], batch_tile = [2, 1, 1], outer_tile = [1, 4, 1], thread_tile = [1, 2, 32], element_tile = [2, 4, 1], subgroup_strides = [0, 0, 0], thread_strides = [0, 32, 1]>

// CHECK-LABEL: func.func @batch_matmul_block_intrinsic

// CHECK-DAG: %[[LHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED]])
// CHECK-DAG: %[[RHS:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED1]])
// CHECK-DAG: %[[ACC:.+]] = iree_vector_ext.to_layout %{{.*}} to layout(#[[$NESTED2]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[ACC]]
