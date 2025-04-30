// RUN: iree-opt --split-input-file --iree-transform-dialect-interpreter --canonicalize --cse %s | FileCheck %s

// CDNA3 V_MFMA_F32_32x32x8_F16

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 32x8, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [1, 1],
  thread_tile       = [32, 2],
  element_tile     = [1, 4],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 32]
>

// B: shape = 8x32, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [1, 1],
  thread_tile       = [2, 32],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [32, 1]
>

// C: shape = 32x32, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [4, 1],
  thread_tile       = [2, 32],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [32, 1]
>

func.func @contract_to_mfma_32x32x8_mm(%a : vector<32x8xf16>, %b : vector<8x32xf16>, %c : vector<32x32xf32>) -> vector<32x32xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<32x8xf16>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<8x32xf16>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<32x32xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mma = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>
  } %A, %B, %C : vector<32x8xf16>, vector<8x32xf16> into vector<32x32xf32>

  %O = iree_vector_ext.to_layout %output to layout(#layout_c) : vector<32x32xf32>
  return %O : vector<32x32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @contract_to_mfma_32x32x8_mm
// CHECK-SAME: (%[[A:.+]]: vector<32x8xf16>, %[[B:.+]]: vector<8x32xf16>, %[[C:.+]]: vector<32x32xf32>)
// CHECK:       %[[C_SIMT:.+]] = iree_vector_ext.to_simt %[[C]] : vector<32x32xf32> -> vector<1x1x4x1x4x1xf32
// CHECK:       %[[A_SIMT:.+]] = iree_vector_ext.to_simt %[[A]] : vector<32x8xf16>  -> vector<1x1x1x1x1x4xf16>
// CHECK:       %[[B_SIMT:.+]] = iree_vector_ext.to_simt %[[B]] : vector<8x32xf16>  -> vector<1x1x1x1x4x1xf16>
// CHECK:       %[[C_VEC:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<4x1x4x1xf32> from vector<1x1x4x1x4x1xf32>
// CHECK:       %[[A_VEC:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x4xf16> from vector<1x1x1x1x1x4xf16>
// CHECK:       %[[B_VEC:.+]] = vector.extract %[[B_SIMT]][0, 0] : vector<1x1x4x1xf16> from vector<1x1x1x1x4x1xf16>
// CHECK:       %[[A_CAST:.+]] = vector.shape_cast %[[A_VEC]] : vector<1x1x1x4xf16> to vector<4xf16>
// CHECK:       %[[B_CAST:.+]] = vector.shape_cast %[[B_VEC]] : vector<1x1x4x1xf16> to vector<4xf16>
// CHECK:       %[[C_CAST:.+]] = vector.shape_cast %[[C_VEC]] : vector<4x1x4x1xf32> to vector<16xf32>
// CHECK:       %[[MFMA:.+]] = amdgpu.mfma %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]]
// CHECK-SAME:     {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none
// CHECK-SAME:     : vector<4xf16>, vector<4xf16>, vector<16xf32>
// CHECK:       %[[R_CAST:.+]] = vector.shape_cast %[[MFMA]] : vector<16xf32> to vector<4x1x4x1xf32>
// CHECK:       %[[B_OUT:.*]] = vector.broadcast %[[R_CAST]] : vector<4x1x4x1xf32> to vector<1x1x4x1x4x1xf32>
// CHECK:       %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[B_OUT]] : vector<1x1x4x1x4x1xf32> -> vector<32x32xf32>
// CHECK:       return %[[R_SIMD]]

// -----

// CDNA3 V_MFMA_F32_16X16X16_F16

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 16x16, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [1, 1],
  thread_tile       = [16, 4],
  element_tile     = [1, 4],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 16]
>

// B: shape = 16x16, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [1, 1],
  thread_tile       = [4, 16],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [16, 1]
>

// C: shape = 16x16, layout = layoutB

func.func @contract_to_mfma_16x16x16_mm(%a : vector<16x16xf16>, %b : vector<16x16xf16>, %c : vector<16x16xf32>) -> vector<16x16xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<16x16xf16>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<16x16xf16>
  %C = iree_vector_ext.to_layout %c to layout(#layout_b) : vector<16x16xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mma = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } %A, %B, %C : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>

  %O = iree_vector_ext.to_layout %output to layout(#layout_b) : vector<16x16xf32>
  return %O : vector<16x16xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @contract_to_mfma_16x16x16_mm
//  CHECK-SAME: (%[[A:.+]]: vector<16x16xf16>, %[[B:.+]]: vector<16x16xf16>, %[[C:.+]]: vector<16x16xf32>)
//       CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %[[C]] : vector<16x16xf32> -> vector<1x1x1x1x4x1xf32>
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %[[A]] : vector<16x16xf16> -> vector<1x1x1x1x1x4xf16>
//       CHECK:   %[[B_SIMT:.+]] = iree_vector_ext.to_simt %[[B]] : vector<16x16xf16> -> vector<1x1x1x1x4x1xf16>
//       CHECK:   %[[C_VEC:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<1x1x4x1xf32> from vector<1x1x1x1x4x1xf32>
//       CHECK:   %[[A_VEC:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x4xf16> from vector<1x1x1x1x1x4xf16>
//       CHECK:   %[[B_VEC:.+]] = vector.extract %[[B_SIMT]][0, 0] : vector<1x1x4x1xf16> from vector<1x1x1x1x4x1xf16>
//       CHECK:   %[[A_CAST:.+]] = vector.shape_cast %[[A_VEC]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[B_CAST:.+]] = vector.shape_cast %[[B_VEC]] : vector<1x1x4x1xf16> to vector<4xf16>
//       CHECK:   %[[C_CAST:.+]] = vector.shape_cast %[[C_VEC]] : vector<1x1x4x1xf32> to vector<4xf32>
//       CHECK:   %[[MFMA:.+]] = amdgpu.mfma %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]]
//  CHECK-SAME:     {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16>, vector<4xf32>

//       CHECK:   %[[R_CAST:.+]] = vector.shape_cast %[[MFMA]]  : vector<4xf32> to vector<1x1x4x1xf32>
//       CHECK:   %[[B_OUT:.*]]  = vector.broadcast %[[R_CAST]] : vector<1x1x4x1xf32> to vector<1x1x1x1x4x1xf32>
//       CHECK:   %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[B_OUT]] : vector<1x1x1x1x4x1xf32> -> vector<16x16xf32>
//       CHECK:   return %[[R_SIMD]]

// -----

// None-one M/N batch

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 64x8, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 1],
  outer_tile        = [1, 1],
  thread_tile       = [32, 2],
  element_tile     = [1, 4],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 32]
>

// B: shape = 8x32, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [1, 1],
  thread_tile       = [2, 32],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [32, 1]
>

// C: shape = 64x32, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 1],
  outer_tile        = [4, 1],
  thread_tile       = [2, 32],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [32, 1]
>

func.func @contract_to_mfma_32x32x8_mm_mnbatch(%a : vector<64x8xf16>, %b : vector<8x32xf16>, %c : vector<64x32xf32>) -> vector<64x32xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<64x8xf16>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<8x32xf16>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<64x32xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mma = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>
  } %A, %B, %C : vector<64x8xf16>, vector<8x32xf16> into vector<64x32xf32>

  %O = iree_vector_ext.to_layout %output to layout(#layout_c) : vector<64x32xf32>
  return %O : vector<64x32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @contract_to_mfma_32x32x8_mm_mnbatch
//       CHECK:   %[[INIT:.+]] = arith.constant dense<0.000000e+00>
//       CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<64x32xf32> -> vector<2x1x4x1x4x1xf32>
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<64x8xf16> -> vector<2x1x1x1x1x4xf16>
//       CHECK:   %[[C_SLICE0:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<4x1x4x1xf32> from vector<2x1x4x1x4x1xf32
//       CHECK:   %[[A_SLICE0:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x4xf16> from vector<2x1x1x1x1x4xf16>
//       CHECK:   %[[A0_CAST:.+]] = vector.shape_cast %[[A_SLICE0]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[C0_CAST:.+]] = vector.shape_cast %[[C_SLICE0]] : vector<4x1x4x1xf32> to vector<16xf32>
//       CHECK:   %[[MFMA0:.+]] = amdgpu.mfma %[[A0_CAST]] * %{{.+}} + %[[C0_CAST]]
//       CHECK:   %[[R0_CAST:.+]] = vector.shape_cast %[[MFMA0]] : vector<16xf32> to vector<4x1x4x1xf32>
//       CHECK:   %[[C0_INS:.+]] = vector.insert %[[R0_CAST]], %[[INIT]] [0, 0] : vector<4x1x4x1xf32> into vector<2x1x4x1x4x1xf32>
//       CHECK:   %[[C_SLICE1:.+]] = vector.extract %[[C_SIMT]][1, 0] : vector<4x1x4x1xf32> from vector<2x1x4x1x4x1xf32>
//       CHECK:   %[[A_SLICE1:.+]] = vector.extract %[[A_SIMT]][1, 0] : vector<1x1x1x4xf16> from vector<2x1x1x1x1x4xf16>
//       CHECK:   %[[A1_CAST:.+]] = vector.shape_cast %[[A_SLICE1]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[C1_CAST:.+]] = vector.shape_cast %[[C_SLICE1]] : vector<4x1x4x1xf32> to vector<16xf32>
//       CHECK:   %[[MFMA1:.+]] = amdgpu.mfma %[[A1_CAST]] * %{{.+}} + %[[C1_CAST]]
//       CHECK:   %[[R1_CAST:.+]] = vector.shape_cast %[[MFMA1]] : vector<16xf32> to vector<4x1x4x1xf32>
//       CHECK:   %[[C1_INS:.+]] = vector.insert %[[R1_CAST]], %[[C0_INS]] [1, 0] : vector<4x1x4x1xf32> into vector<2x1x4x1x4x1xf32>
//       CHECK:   %[[R:.+]] = iree_vector_ext.to_simd %[[C1_INS]] : vector<2x1x4x1x4x1xf32> -> vector<64x32xf32>
//       CHECK:   return %[[R]]

// -----

// None-one K batch

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 32x16, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 2],
  outer_tile        = [1, 1],
  thread_tile       = [32, 2],
  element_tile     = [1, 4],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 32]
>

// B: shape = 16x32, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 1],
  outer_tile        = [1, 1],
  thread_tile       = [2, 32],
  element_tile     = [4, 1],

  subgroup_strides         = [1, 1],
  thread_strides          = [32, 1]
>

// C: shape = 32x32, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [4, 1],
  thread_tile       = [2, 32],
  element_tile     = [4, 1],

  subgroup_strides         = [1, 1],
  thread_strides          = [32, 1]
>

func.func @contract_to_mfma_32x32x8_mm_kbatch(%a : vector<32x16xf16>, %b : vector<16x32xf16>, %c : vector<32x32xf32>) -> vector<32x32xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<32x16xf16>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<16x32xf16>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<32x32xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mma = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>
  } %A, %B, %C : vector<32x16xf16>, vector<16x32xf16> into vector<32x32xf32>

  %O = iree_vector_ext.to_layout %output to layout(#layout_c) : vector<32x32xf32>
  return %O : vector<32x32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @contract_to_mfma_32x32x8_mm_kbatch(%arg0: vector<32x16xf16>, %arg1: vector<16x32xf16>, %arg2: vector<32x32xf32>) -> vector<32x32xf32> {
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<32x16xf16>
//       CHECK:   %[[B_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<16x32xf16>
//       CHECK:   %[[A_SLICE0:.+]] = vector.extract %[[A_SIMT]][0, 0]
//       CHECK:   %[[B_SLICE0:.+]] = vector.extract %[[B_SIMT]][0, 0]
//       CHECK:   %[[A0_CAST:.+]] = vector.shape_cast %[[A_SLICE0]]
//       CHECK:   %[[B0_CAST:.+]] = vector.shape_cast %[[B_SLICE0]]
//       CHECK:   %[[MFMA0:.+]] = amdgpu.mfma %[[A0_CAST]] * %[[B0_CAST]] + %{{.+}}
//       CHECK:   %[[A_SLICE1:.+]] = vector.extract %[[A_SIMT]][0, 1]
//       CHECK:   %[[B_SLICE1:.+]] = vector.extract %[[B_SIMT]][1, 0]
//       CHECK:   %[[A1_CAST:.+]] = vector.shape_cast %[[A_SLICE1]]
//       CHECK:   %[[B1_CAST:.+]] = vector.shape_cast %[[B_SLICE1]]
//       CHECK:   %[[MFMA1:.+]] = amdgpu.mfma %[[A1_CAST]] * %[[B1_CAST]] + %[[MFMA0]]
//       CHECK:   %[[R_CAST:.+]] = vector.shape_cast %[[MFMA1]]

// -----

// None-one M/N batch with permuted order

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 64x8, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 1],
  outer_tile        = [1, 1],
  thread_tile       = [32, 2],
  element_tile     = [1, 4],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 32]
>

// B: shape = 8x96, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 3],
  outer_tile        = [1, 1],
  thread_tile       = [2, 32],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [32, 1]
>

// C: shape = 64x96, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 3],
  outer_tile        = [4, 1],
  thread_tile       = [2, 32],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [32, 1]
>

func.func @contract_to_mfma_32x32x8_mm_mnbatch_order(%a : vector<64x8xf16>, %b : vector<8x96xf16>, %c : vector<64x96xf32>) -> vector<64x96xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<64x8xf16>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<8x96xf16>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<64x96xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mma = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>
  } %A, %B, %C : vector<64x8xf16>, vector<8x96xf16> into vector<64x96xf32>

  %O = iree_vector_ext.to_layout %output to layout(#layout_c) : vector<64x96xf32>
  return %O : vector<64x96xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @contract_to_mfma_32x32x8_mm_mnbatch_order
//       CHECK:   %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<2x3x4x1x4x1xf32>
//       CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<64x96xf32> -> vector<2x3x4x1x4x1xf32>
//       CHECK:   vector.extract %[[C_SIMT]][0, 0]
//       CHECK:   amdgpu.mfma
//       CHECK:   %[[INS0:.+]] = vector.insert %{{.+}}, %[[INIT]] [0, 0]
//       CHECK:   vector.extract %[[C_SIMT]][0, 1]
//       CHECK:   amdgpu.mfma
//       CHECK:   %[[INS1:.+]] = vector.insert %{{.+}}, %[[INS0]] [0, 1]
//       CHECK:   vector.extract %[[C_SIMT]][0, 2]
//       CHECK:   amdgpu.mfma
//       CHECK:   %[[INS2:.+]] = vector.insert %{{.+}}, %[[INS1]] [0, 2]
//       CHECK:   vector.extract %[[C_SIMT]][1, 0]
//       CHECK:   amdgpu.mfma
//       CHECK:   %[[INS3:.+]] = vector.insert %{{.+}}, %[[INS2]] [1, 0]
//       CHECK:   vector.extract %[[C_SIMT]][1, 1]
//       CHECK:   amdgpu.mfma
//       CHECK:   %[[INS4:.+]] = vector.insert %{{.+}}, %[[INS3]] [1, 1]
//       CHECK:   vector.extract %[[C_SIMT]][1, 2]
//       CHECK:   amdgpu.mfma
//       CHECK:   %[[INS5:.+]] = vector.insert %{{.+}}, %[[INS4]] [1, 2]
//       CHECK:   iree_vector_ext.to_simd %[[INS5]]

// -----

// (M, K) x (N, K) -> (M, N)

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (n, k)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 32x8, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [1, 1],
  thread_tile       = [32, 2],
  element_tile     = [1, 4],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 32]
>

// B: shape = 64x8, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [2, 1],
  outer_tile        = [1, 1],
  thread_tile       = [32, 2],
  element_tile     = [1, 4],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 32]
>

// C: shape = 32x64, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 2],
  outer_tile        = [4, 1],
  thread_tile       = [2, 32],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [32, 1]
>

func.func @contract_to_mfma_32x32x8_mmt(%a : vector<32x8xf16>, %b : vector<64x8xf16>, %c : vector<32x64xf32>) -> vector<32x64xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<32x8xf16>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<64x8xf16>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<32x64xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mma = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>
  } %A, %B, %C : vector<32x8xf16>, vector<64x8xf16> into vector<32x64xf32>

  %O = iree_vector_ext.to_layout %output to layout(#layout_c) : vector<32x64xf32>
  return %O : vector<32x64xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @contract_to_mfma_32x32x8_mmt
// CHECK:   %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<1x2x4x1x4x1xf32>
// CHECK:   %[[B_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<64x8xf16> -> vector<2x1x1x1x1x4xf16>
// CHECK:   vector.extract %[[B_SIMT]][0, 0]
// CHECK:   amdgpu.mfma
// CHECK:   %[[INS0:.+]] = vector.insert %{{.+}}, %[[INIT]] [0, 0]
// CHECK:   vector.extract %[[B_SIMT]][1, 0]
// CHECK:   amdgpu.mfma
// CHECK:   %[[INS1:.+]] = vector.insert %17, %[[INS0]] [0, 1]
// CHECK:   iree_vector_ext.to_simd %[[INS1]] : vector<1x2x4x1x4x1xf32> -> vector<32x64xf32>

// -----

// RDNA3 V_WMMA_F32_16X16X16_F32

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 16x16, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [1, 1],
  thread_tile       = [16, 1],
  element_tile     = [1, 16],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 16]
>

// B: shape = 16x16, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [1, 1],
  thread_tile       = [1, 16],
  element_tile     = [16, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [16, 1]
>

// C: shape = 16x16, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [8, 1],
  thread_tile       = [2, 16],
  element_tile     = [1, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [16, 1]
>

func.func @contract_to_WMMAR3_16x16x16_mm(%a : vector<16x16xf16>, %b : vector<16x16xf16>, %c : vector<16x16xf32>) -> vector<16x16xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<16x16xf16>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<16x16xf16>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<16x16xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mma = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>
  } %A, %B, %C : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>

  %O = iree_vector_ext.to_layout %output to layout(#layout_c) : vector<16x16xf32>
  return %O : vector<16x16xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}


// CHECK-LABEL: func.func @contract_to_WMMAR3_16x16x16_mm
//  CHECK-SAME: (%[[A:.+]]: vector<16x16xf16>, %[[B:.+]]: vector<16x16xf16>, %[[C:.+]]: vector<16x16xf32>)
//       CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %[[C]] : vector<16x16xf32> -> vector<1x1x8x1x1x1xf32>
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %[[A]] : vector<16x16xf16> -> vector<1x1x1x1x1x16xf16>
//       CHECK:   %[[B_SIMT:.+]] = iree_vector_ext.to_simt %[[B]] : vector<16x16xf16> -> vector<1x1x1x1x16x1xf16>
//       CHECK:   %[[C_VEC:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<8x1x1x1xf32> from vector<1x1x8x1x1x1xf32>
//       CHECK:   %[[A_VEC:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x16xf16> from vector<1x1x1x1x1x16xf16>
//       CHECK:   %[[B_VEC:.+]] = vector.extract %[[B_SIMT]][0, 0] : vector<1x1x16x1xf16> from vector<1x1x1x1x16x1xf16>
//       CHECK:   %[[A_CAST:.+]] = vector.shape_cast %[[A_VEC]] : vector<1x1x1x16xf16> to vector<16xf1
//       CHECK:   %[[B_CAST:.+]] = vector.shape_cast %[[B_VEC]] : vector<1x1x16x1xf16> to vector<16xf1
//       CHECK:   %[[C_CAST:.+]] = vector.shape_cast %[[C_VEC]] : vector<8x1x1x1xf32> to vector<8xf32>
//       CHECK:   %[[WMMA:.+]] = amdgpu.wmma %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]]
//       CHECK:   %[[R_CAST:.+]] = vector.shape_cast %[[WMMA]] : vector<8xf32> to vector<8x1x1x1xf32>
//       CHECK:   %[[B_OUT:.*]] = vector.broadcast %[[R_CAST]] : vector<8x1x1x1xf32> to vector<1x1x8x1x1x1xf32>
//       CHECK:   %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[B_OUT]] : vector<1x1x8x1x1x1xf32> -> vector<16x16xf32>
//       CHECK:   return %[[R_SIMD]]

// -----

// RDNA4 V_WMMA_F32_16X16X16_F32

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 16x16, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [16, 2],
  element_tile     = [1, 8],

  subgroup_strides = [1, 1],
  thread_strides   = [1, 16]
>

// B: shape = 16x16, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [2, 16],
  element_tile     = [8, 1],

  subgroup_strides = [1, 1],
  thread_strides   = [16, 1]
>

// C: shape = 16x16, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [2, 16],
  element_tile     = [8, 1],

  subgroup_strides = [1, 1],
  thread_strides   = [16, 1]
>

func.func @contract_to_WMMAR4_16x16x16_mm(%a : vector<16x16xf16>, %b : vector<16x16xf16>, %c : vector<16x16xf32>) -> vector<16x16xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<16x16xf16>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<16x16xf16>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<16x16xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mma = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>
  } %A, %B, %C : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>

  %O = iree_vector_ext.to_layout %output to layout(#layout_c) : vector<16x16xf32>
  return %O : vector<16x16xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}


// CHECK-LABEL: func.func @contract_to_WMMAR4_16x16x16_mm
//  CHECK-SAME: (%[[A:.+]]: vector<16x16xf16>, %[[B:.+]]: vector<16x16xf16>, %[[C:.+]]: vector<16x16xf32>)
//       CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %[[C]] : vector<16x16xf32> -> vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %[[A]] : vector<16x16xf16> -> vector<1x1x1x1x1x8xf16>
//       CHECK:   %[[B_SIMT:.+]] = iree_vector_ext.to_simt %[[B]] : vector<16x16xf16> -> vector<1x1x1x1x8x1xf16>
//       CHECK:   %[[C_VEC:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<1x1x8x1xf32> from vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[A_VEC:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x8xf16> from vector<1x1x1x1x1x8xf16>
//       CHECK:   %[[B_VEC:.+]] = vector.extract %[[B_SIMT]][0, 0] : vector<1x1x8x1xf16> from vector<1x1x1x1x8x1xf16>
//       CHECK:   %[[A_CAST:.+]] = vector.shape_cast %[[A_VEC]] : vector<1x1x1x8xf16> to vector<8xf16>
//       CHECK:   %[[B_CAST:.+]] = vector.shape_cast %[[B_VEC]] : vector<1x1x8x1xf16> to vector<8xf16>
//       CHECK:   %[[C_CAST:.+]] = vector.shape_cast %[[C_VEC]] : vector<1x1x8x1xf32> to vector<8xf32>
//       CHECK:   %[[WMMA:.+]] = amdgpu.wmma %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]]
//       CHECK:   %[[R_CAST:.+]] = vector.shape_cast %[[WMMA]] : vector<8xf32> to vector<1x1x8x1xf32>
//       CHECK:   %[[B_OUT:.*]] = vector.broadcast %[[R_CAST]] : vector<1x1x8x1xf32> to vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[B_OUT]] : vector<1x1x1x1x8x1xf32> -> vector<16x16xf32>
//       CHECK:   return %[[R_SIMD]]

// -----

// Non-native MFMA_F32_32x32x16_F16, i.e CDNA3 V_MFMA_F32_32x32x8_F16 with unrolled_k = 2.
// This non native layout maximizes reads from shared memory to register.

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 32x16, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [1, 1],
  thread_tile       = [32, 2],
  element_tile     = [1, 8],

  subgroup_strides        = [1, 1],
  thread_strides          = [1, 32]
>

// B: shape = 16x32, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [1, 1],
  thread_tile       = [2, 32],
  element_tile     = [8, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [32, 1]
>

// C: shape = 32x32, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile        = [4, 1],
  thread_tile       = [2, 32],
  element_tile     = [4, 1],

  subgroup_strides        = [1, 1],
  thread_strides          = [32, 1]
>

func.func @contract_to_vmfma_32x32x16_mm(%a : vector<32x16xf16>, %b : vector<16x32xf16>, %c : vector<32x32xf32>) -> vector<32x32xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<32x16xf16>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<16x32xf16>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<32x32xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mma = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_32x32x16_F16>
  } %A, %B, %C : vector<32x16xf16>, vector<16x32xf16> into vector<32x32xf32>

  %O = iree_vector_ext.to_layout %output to layout(#layout_c) : vector<32x32xf32>
  return %O : vector<32x32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// Notable things to look out for:
// 1. We are reading 8xf16 instead of 4xf16 for lhs,rhs operands.
// 2. We slice the 8xf16 to 2 different 4xf16 per operand for use on 2 MMAs.
// 3. Result of first mma becomes the second mma's accumulator.

// CHECK-LABEL: func @contract_to_vmfma_32x32x16_mm
// CHECK:       %[[A_CAST:.+]] = vector.shape_cast %{{.+}} : vector<1x1x1x8xf16> to vector<8xf16>
// CHECK:       %[[B_CAST:.+]] = vector.shape_cast %{{.+}} : vector<1x1x8x1xf16> to vector<8xf16>
// CHECK:       %[[C_CAST:.+]] = vector.shape_cast %{{.+}} : vector<4x1x4x1xf32> to vector<16xf32>
// CHECK:       %[[A_SLICE_0:.+]] = vector.extract_strided_slice %[[A_CAST]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:       %[[B_SLICE_0:.+]] = vector.extract_strided_slice %[[B_CAST]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:       %[[MFMA_0:.*]] = amdgpu.mfma %[[A_SLICE_0]] * %[[B_SLICE_0]] + %[[C_CAST]]
// CHECK-SAME:     {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none
// CHECK-SAME:     : vector<4xf16>, vector<4xf16>, vector<16xf32>
// CHECK:       %[[A_SLICE_1:.+]] = vector.extract_strided_slice %[[A_CAST]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:       %[[B_SLICE_1:.+]] = vector.extract_strided_slice %[[B_CAST]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:       %[[MFMA_1:.+]] = amdgpu.mfma %[[A_SLICE_1]] * %[[B_SLICE_1]] + %[[MFMA_0]]
// CHECK-SAME:     {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none
// CHECK-SAME:     : vector<4xf16>, vector<4xf16>, vector<16xf32>
// CHECK:       %[[R_CAST:.+]] = vector.shape_cast %[[MFMA_1]] : vector<16xf32> to vector<4x1x4x1xf32>
// CHECK:       %[[B_OUT:.*]] = vector.broadcast %[[R_CAST]] : vector<4x1x4x1xf32> to vector<1x1x4x1x4x1xf32>
// CHECK:       %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[B_OUT]] : vector<1x1x4x1x4x1xf32> -> vector<32x32xf32>
// CHECK:       return %[[R_SIMD]]
