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
// CHECK:       %[[C_SIMT:.+]] = iree_vector_ext.to_simt %[[C]] : vector<32x32xf32> -> vector<1x1x4x1x4x1xf32>
// CHECK:       %[[A_SIMT:.+]] = iree_vector_ext.to_simt %[[A]] : vector<32x8xf16>  -> vector<1x1x1x1x1x4xf16>
// CHECK:       %[[B_SIMT:.+]] = iree_vector_ext.to_simt %[[B]] : vector<8x32xf16>  -> vector<1x1x1x1x4x1xf16>
// CHECK:       %[[C_VEC:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<4x1x4x1xf32> from vector<1x1x4x1x4x1xf32>
// CHECK:       %[[A_VEC:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x4xf16> from vector<1x1x1x1x1x4xf16>
// CHECK:       %[[B_VEC:.+]] = vector.extract %[[B_SIMT]][0, 0] : vector<1x1x4x1xf16> from vector<1x1x1x1x4x1xf16>
// CHECK:       %[[A_CAST:.+]] = vector.shape_cast %[[A_VEC]] : vector<1x1x1x4xf16> to vector<4xf16>
// CHECK:       %[[B_CAST:.+]] = vector.shape_cast %[[B_VEC]] : vector<1x1x4x1xf16> to vector<4xf16>
// CHECK:       %[[C_CAST:.+]] = vector.shape_cast %[[C_VEC]] : vector<4x1x4x1xf32> to vector<16xf32>
// CHECK:       %[[MFMA:.+]] = amdgpu.mfma 32x32x8 %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]] blgp =  none
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
//       CHECK:   %[[MFMA:.+]] = amdgpu.mfma 16x16x16 %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]] blgp =  none
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
//       CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<64x32xf32> -> vector<2x1x4x1x4x1xf32>
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<64x8xf16> -> vector<2x1x1x1x1x4xf16>
//       CHECK:   %[[C_SLICE0:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<4x1x4x1xf32> from vector<2x1x4x1x4x1xf32>
//       CHECK:   %[[A_SLICE0:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x4xf16> from vector<2x1x1x1x1x4xf16>
//       CHECK:   %[[A0_CAST:.+]] = vector.shape_cast %[[A_SLICE0]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[C0_CAST:.+]] = vector.shape_cast %[[C_SLICE0]] : vector<4x1x4x1xf32> to vector<16xf32>
//       CHECK:   %[[MFMA0:.+]] = amdgpu.mfma 32x32x8 %[[A0_CAST]] * %{{.+}} + %[[C0_CAST]]
//       CHECK:   %[[R0_CAST:.+]] = vector.shape_cast %[[MFMA0]] : vector<16xf32> to vector<4x1x4x1xf32>
//       CHECK:   %[[C_SLICE1:.+]] = vector.extract %[[C_SIMT]][1, 0] : vector<4x1x4x1xf32> from vector<2x1x4x1x4x1xf32>
//       CHECK:   %[[A_SLICE1:.+]] = vector.extract %[[A_SIMT]][1, 0] : vector<1x1x1x4xf16> from vector<2x1x1x1x1x4xf16>
//       CHECK:   %[[A1_CAST:.+]] = vector.shape_cast %[[A_SLICE1]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[C1_CAST:.+]] = vector.shape_cast %[[C_SLICE1]] : vector<4x1x4x1xf32> to vector<16xf32>
//       CHECK:   %[[MFMA1:.+]] = amdgpu.mfma 32x32x8 %[[A1_CAST]] * %{{.+}} + %[[C1_CAST]]
//       CHECK:   %[[R1_CAST:.+]] = vector.shape_cast %[[MFMA1]] : vector<16xf32> to vector<4x1x4x1xf32>
//       CHECK:   %[[R0:.+]]:16 = vector.to_elements %[[R0_CAST]] : vector<4x1x4x1xf32>
//       CHECK:   %[[R1:.+]]:16 = vector.to_elements %[[R1_CAST]] : vector<4x1x4x1xf32>
//       CHECK:   %[[INS:.+]] = vector.from_elements
// CHECK-SAME:       %[[R0]]#0, %[[R0]]#1, %[[R0]]#2, %[[R0]]#3, %[[R0]]#4, %[[R0]]#5, %[[R0]]#6, %[[R0]]#7, %[[R0]]#8, %[[R0]]#9, %[[R0]]#10, %[[R0]]#11, %[[R0]]#12, %[[R0]]#13, %[[R0]]#14, %[[R0]]#15
// CHECK-SAME:       %[[R1]]#0, %[[R1]]#1, %[[R1]]#2, %[[R1]]#3, %[[R1]]#4, %[[R1]]#5, %[[R1]]#6, %[[R1]]#7, %[[R1]]#8, %[[R1]]#9, %[[R1]]#10, %[[R1]]#11, %[[R1]]#12, %[[R1]]#13, %[[R1]]#14, %[[R1]]#15
//       CHECK:   %[[R:.+]] = iree_vector_ext.to_simd %[[INS]] : vector<2x1x4x1x4x1xf32> -> vector<64x32xf32>
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
//       CHECK:   %[[MFMA0:.+]] = amdgpu.mfma 32x32x8 %[[A0_CAST]] * %[[B0_CAST]] + %{{.+}}
//       CHECK:   %[[A_SLICE1:.+]] = vector.extract %[[A_SIMT]][0, 1]
//       CHECK:   %[[B_SLICE1:.+]] = vector.extract %[[B_SIMT]][1, 0]
//       CHECK:   %[[A1_CAST:.+]] = vector.shape_cast %[[A_SLICE1]]
//       CHECK:   %[[B1_CAST:.+]] = vector.shape_cast %[[B_SLICE1]]
//       CHECK:   %[[MFMA1:.+]] = amdgpu.mfma 32x32x8 %[[A1_CAST]] * %[[B1_CAST]] + %[[MFMA0]]
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

//   CHECK-LABEL: func.func @contract_to_mfma_32x32x8_mm_mnbatch_order
//         CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<64x96xf32> -> vector<2x3x4x1x4x1xf32>
//         CHECK:   vector.extract %[[C_SIMT]][0, 0]
//         CHECK:   amdgpu.mfma
//         CHECK:   vector.extract %[[C_SIMT]][0, 1]
//         CHECK:   amdgpu.mfma
//         CHECK:   vector.extract %[[C_SIMT]][0, 2]
//         CHECK:   amdgpu.mfma
//         CHECK:   vector.extract %[[C_SIMT]][1, 0]
//         CHECK:   amdgpu.mfma
//         CHECK:   vector.extract %[[C_SIMT]][1, 1]
//         CHECK:   amdgpu.mfma
//         CHECK:   vector.extract %[[C_SIMT]][1, 2]
//         CHECK:   amdgpu.mfma
// CHECK-COUNT-6:   vector.to_elements {{.*}} : vector<4x1x4x1xf32>
//         CHECK:   %[[INS:.+]] = vector.from_elements
//         CHECK:   iree_vector_ext.to_simd %[[INS]]

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
// CHECK:   %[[B_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<64x8xf16> -> vector<2x1x1x1x1x4xf16>
// CHECK:   vector.extract %[[B_SIMT]][0, 0]
// CHECK:   amdgpu.mfma
// CHECK:   vector.extract %[[B_SIMT]][1, 0]
// CHECK:   amdgpu.mfma
// CHECK:   %[[R0:.+]]:16 = vector.to_elements %{{.+}} : vector<4x1x4x1xf32>
// CHECK:   %[[R1:.+]]:16 = vector.to_elements %{{.+}} : vector<4x1x4x1xf32>
// CHECK:   %[[INS:.+]] = vector.from_elements
// CHECK-SAME:       %[[R0]]#0, %[[R0]]#1, %[[R0]]#2, %[[R0]]#3, %[[R0]]#4, %[[R0]]#5, %[[R0]]#6, %[[R0]]#7, %[[R0]]#8, %[[R0]]#9, %[[R0]]#10, %[[R0]]#11, %[[R0]]#12, %[[R0]]#13, %[[R0]]#14, %[[R0]]#15
// CHECK-SAME:       %[[R1]]#0, %[[R1]]#1, %[[R1]]#2, %[[R1]]#3, %[[R1]]#4, %[[R1]]#5, %[[R1]]#6, %[[R1]]#7, %[[R1]]#8, %[[R1]]#9, %[[R1]]#10, %[[R1]]#11, %[[R1]]#12, %[[R1]]#13, %[[R1]]#14, %[[R1]]#15
// CHECK:   iree_vector_ext.to_simd %[[INS]] : vector<1x2x4x1x4x1xf32> -> vector<32x64xf32>

// -----

// RDNA3 V_WMMA_F32_16X16X16_F16

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
//       CHECK:   %[[A_CAST:.+]] = vector.shape_cast %[[A_VEC]] : vector<1x1x1x16xf16> to vector<16xf16>
//       CHECK:   %[[B_CAST:.+]] = vector.shape_cast %[[B_VEC]] : vector<1x1x16x1xf16> to vector<16xf16>
//       CHECK:   %[[C_CAST:.+]] = vector.shape_cast %[[C_VEC]] : vector<8x1x1x1xf32> to vector<8xf32>
//       CHECK:   %[[WMMA:.+]] = amdgpu.wmma 16x16x16 %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]]
//       CHECK:   %[[R_CAST:.+]] = vector.shape_cast %[[WMMA]] : vector<8xf32> to vector<8x1x1x1xf32>
//       CHECK:   %[[B_OUT:.*]] = vector.broadcast %[[R_CAST]] : vector<8x1x1x1xf32> to vector<1x1x8x1x1x1xf32>
//       CHECK:   %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[B_OUT]] : vector<1x1x8x1x1x1xf32> -> vector<16x16xf32>
//       CHECK:   return %[[R_SIMD]]

// -----

// RDNA4 V_WMMA_F32_16X16X16_F16

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
//       CHECK:   %[[WMMA:.+]] = amdgpu.wmma 16x16x16 %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]]
//       CHECK:   %[[R_CAST:.+]] = vector.shape_cast %[[WMMA]] : vector<8xf32> to vector<1x1x8x1xf32>
//       CHECK:   %[[B_OUT:.*]] = vector.broadcast %[[R_CAST]] : vector<1x1x8x1xf32> to vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[B_OUT]] : vector<1x1x1x1x8x1xf32> -> vector<16x16xf32>
//       CHECK:   return %[[R_SIMD]]

// -----

// gfx1250 V_WMMA_F32_16X16X4_F32

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 16x4, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [16, 2],
  element_tile     = [1, 2],

  subgroup_strides = [1, 1],
  thread_strides   = [1, 16]
>

// B: shape = 4x16, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [2, 16],
  element_tile     = [2, 1],

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

func.func @contract_to_gfx1250_WMMA_16x16x4_mm(%a : vector<16x4xf32>, %b : vector<4x16xf32>, %c : vector<16x16xf32>) -> vector<16x16xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<16x4xf32>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<4x16xf32>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<16x16xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mma = #iree_gpu.mma_layout<WMMA_F32_16x16x4_F32>
  } %A, %B, %C : vector<16x4xf32>, vector<4x16xf32> into vector<16x16xf32>

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

// CHECK-LABEL: func.func @contract_to_gfx1250_WMMA_16x16x4_mm
//  CHECK-SAME: (%[[A:.+]]: vector<16x4xf32>, %[[B:.+]]: vector<4x16xf32>, %[[C:.+]]: vector<16x16xf32>)
//       CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %[[C]] : vector<16x16xf32> -> vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %[[A]] : vector<16x4xf32> -> vector<1x1x1x1x1x2xf32>
//       CHECK:   %[[B_SIMT:.+]] = iree_vector_ext.to_simt %[[B]] : vector<4x16xf32> -> vector<1x1x1x1x2x1xf32>
//       CHECK:   %[[C_VEC:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<1x1x8x1xf32> from vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[A_VEC:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x2xf32> from vector<1x1x1x1x1x2xf32>
//       CHECK:   %[[B_VEC:.+]] = vector.extract %[[B_SIMT]][0, 0] : vector<1x1x2x1xf32> from vector<1x1x1x1x2x1xf32>
//       CHECK:   %[[A_CAST:.+]] = vector.shape_cast %[[A_VEC]] : vector<1x1x1x2xf32> to vector<2xf32>
//       CHECK:   %[[B_CAST:.+]] = vector.shape_cast %[[B_VEC]] : vector<1x1x2x1xf32> to vector<2xf32>
//       CHECK:   %[[C_CAST:.+]] = vector.shape_cast %[[C_VEC]] : vector<1x1x8x1xf32> to vector<8xf32>
//       CHECK:   %[[WMMA:.+]] = amdgpu.wmma 16x16x4 %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]] : vector<2xf32>, vector<2xf32>, vector<8xf32>
//       CHECK:   %[[R_CAST:.+]] = vector.shape_cast %[[WMMA]] : vector<8xf32> to vector<1x1x8x1xf32>
//       CHECK:   %[[B_OUT:.*]] = vector.broadcast %[[R_CAST]] : vector<1x1x8x1xf32> to vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[B_OUT]] : vector<1x1x1x1x8x1xf32> -> vector<16x16xf32>
//       CHECK:   return %[[R_SIMD]]

// -----

// gfx1250 V_WMMA_F32_16X16X32_F16

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 16x32, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [16, 2],
  element_tile     = [1, 16],

  subgroup_strides = [1, 1],
  thread_strides   = [1, 16]
>

// B: shape = 32x16, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [2, 16],
  element_tile     = [16, 1],

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

func.func @contract_to_gfx1250_WMMA_16x16x32_mm(%a : vector<16x32xf16>, %b : vector<32x16xf16>, %c : vector<16x16xf32>) -> vector<16x16xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<16x32xf16>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<32x16xf16>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<16x16xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mma = #iree_gpu.mma_layout<WMMA_F32_16x16x32_F16>
  } %A, %B, %C : vector<16x32xf16>, vector<32x16xf16> into vector<16x16xf32>

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

// CHECK-LABEL: func.func @contract_to_gfx1250_WMMA_16x16x32_mm
//  CHECK-SAME: (%[[A:.+]]: vector<16x32xf16>, %[[B:.+]]: vector<32x16xf16>, %[[C:.+]]: vector<16x16xf32>)
//       CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %[[C]] : vector<16x16xf32> -> vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %[[A]] : vector<16x32xf16> -> vector<1x1x1x1x1x16xf16>
//       CHECK:   %[[B_SIMT:.+]] = iree_vector_ext.to_simt %[[B]] : vector<32x16xf16> -> vector<1x1x1x1x16x1xf16>
//       CHECK:   %[[C_VEC:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<1x1x8x1xf32> from vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[A_VEC:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x16xf16> from vector<1x1x1x1x1x16xf16>
//       CHECK:   %[[B_VEC:.+]] = vector.extract %[[B_SIMT]][0, 0] : vector<1x1x16x1xf16> from vector<1x1x1x1x16x1xf16>
//       CHECK:   %[[A_CAST:.+]] = vector.shape_cast %[[A_VEC]] : vector<1x1x1x16xf16> to vector<16xf16>
//       CHECK:   %[[B_CAST:.+]] = vector.shape_cast %[[B_VEC]] : vector<1x1x16x1xf16> to vector<16xf16>
//       CHECK:   %[[C_CAST:.+]] = vector.shape_cast %[[C_VEC]] : vector<1x1x8x1xf32> to vector<8xf32>
//       CHECK:   %[[WMMA:.+]] = amdgpu.wmma 16x16x32 %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]] : vector<16xf16>, vector<16xf16>, vector<8xf32>
//       CHECK:   %[[R_CAST:.+]] = vector.shape_cast %[[WMMA]] : vector<8xf32> to vector<1x1x8x1xf32>
//       CHECK:   %[[B_OUT:.*]] = vector.broadcast %[[R_CAST]] : vector<1x1x8x1xf32> to vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[B_OUT]] : vector<1x1x1x1x8x1xf32> -> vector<16x16xf32>
//       CHECK:   return %[[R_SIMD]]

// -----

// gfx1250 V_WMMA_F32_16X16X64_F8

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 16x64, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [16, 2],
  element_tile     = [1, 32],

  subgroup_strides = [1, 1],
  thread_strides   = [1, 16]
>

// B: shape = 64x16, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [2, 16],
  element_tile     = [32, 1],

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

func.func @contract_to_gfx1250_WMMA_16x16x64_mm(%a : vector<16x64xf8E4M3FN>, %b : vector<64x16xf8E4M3FN>, %c : vector<16x16xf32>) -> vector<16x16xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<16x64xf8E4M3FN>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<64x16xf8E4M3FN>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<16x16xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mma = #iree_gpu.mma_layout<WMMA_F32_16x16x64_F8E4M3FN>
  } %A, %B, %C : vector<16x64xf8E4M3FN>, vector<64x16xf8E4M3FN> into vector<16x16xf32>

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

// CHECK-LABEL: func.func @contract_to_gfx1250_WMMA_16x16x64_mm
//  CHECK-SAME: (%[[A:.+]]: vector<16x64xf8E4M3FN>, %[[B:.+]]: vector<64x16xf8E4M3FN>, %[[C:.+]]: vector<16x16xf32>)
//       CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %[[C]] : vector<16x16xf32> -> vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %[[A]] : vector<16x64xf8E4M3FN> -> vector<1x1x1x1x1x32xf8E4M3FN>
//       CHECK:   %[[B_SIMT:.+]] = iree_vector_ext.to_simt %[[B]] : vector<64x16xf8E4M3FN> -> vector<1x1x1x1x32x1xf8E4M3FN>
//       CHECK:   %[[C_VEC:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<1x1x8x1xf32> from vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[A_VEC:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x32xf8E4M3FN> from vector<1x1x1x1x1x32xf8E4M3FN>
//       CHECK:   %[[B_VEC:.+]] = vector.extract %[[B_SIMT]][0, 0] : vector<1x1x32x1xf8E4M3FN> from vector<1x1x1x1x32x1xf8E4M3FN>
//       CHECK:   %[[A_CAST:.+]] = vector.shape_cast %[[A_VEC]] : vector<1x1x1x32xf8E4M3FN> to vector<32xf8E4M3FN>
//       CHECK:   %[[B_CAST:.+]] = vector.shape_cast %[[B_VEC]] : vector<1x1x32x1xf8E4M3FN> to vector<32xf8E4M3FN>
//       CHECK:   %[[C_CAST:.+]] = vector.shape_cast %[[C_VEC]] : vector<1x1x8x1xf32> to vector<8xf32>
//       CHECK:   %[[WMMA:.+]] = amdgpu.wmma 16x16x64 %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]] : vector<32xf8E4M3FN>, vector<32xf8E4M3FN>, vector<8xf32>
//       CHECK:   %[[R_CAST:.+]] = vector.shape_cast %[[WMMA]] : vector<8xf32> to vector<1x1x8x1xf32>
//       CHECK:   %[[B_OUT:.*]] = vector.broadcast %[[R_CAST]] : vector<1x1x8x1xf32> to vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[B_OUT]] : vector<1x1x1x1x8x1xf32> -> vector<16x16xf32>
//       CHECK:   return %[[R_SIMD]]

// -----

// gfx1250 V_WMMA_F32_16X16X128_F8

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 16x128, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [16, 2],
  element_tile     = [1, 64],

  subgroup_strides = [1, 1],
  thread_strides   = [1, 16]
>

// B: shape = 128x16, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile    = [1, 1],
  batch_tile       = [1, 1],
  outer_tile       = [1, 1],
  thread_tile      = [2, 16],
  element_tile     = [64, 1],

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

func.func @contract_to_gfx1250_WMMA_16x16x128_mm(%a : vector<16x128xf8E4M3FN>, %b : vector<128x16xf8E4M3FN>, %c : vector<16x16xf32>) -> vector<16x16xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<16x128xf8E4M3FN>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<128x16xf8E4M3FN>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<16x16xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mma = #iree_gpu.mma_layout<WMMA_F32_16x16x128_F8E4M3FN>
  } %A, %B, %C : vector<16x128xf8E4M3FN>, vector<128x16xf8E4M3FN> into vector<16x16xf32>

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

// CHECK-LABEL: func.func @contract_to_gfx1250_WMMA_16x16x128_mm
//  CHECK-SAME: (%[[A:.+]]: vector<16x128xf8E4M3FN>, %[[B:.+]]: vector<128x16xf8E4M3FN>, %[[C:.+]]: vector<16x16xf32>)
//       CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %[[C]] : vector<16x16xf32> -> vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %[[A]] : vector<16x128xf8E4M3FN> -> vector<1x1x1x1x1x64xf8E4M3FN>
//       CHECK:   %[[B_SIMT:.+]] = iree_vector_ext.to_simt %[[B]] : vector<128x16xf8E4M3FN> -> vector<1x1x1x1x64x1xf8E4M3FN>
//       CHECK:   %[[C_VEC:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<1x1x8x1xf32> from vector<1x1x1x1x8x1xf32>
//       CHECK:   %[[A_VEC:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x64xf8E4M3FN> from vector<1x1x1x1x1x64xf8E4M3FN>
//       CHECK:   %[[B_VEC:.+]] = vector.extract %[[B_SIMT]][0, 0] : vector<1x1x64x1xf8E4M3FN> from vector<1x1x1x1x64x1xf8E4M3FN>
//       CHECK:   %[[A_CAST:.+]] = vector.shape_cast %[[A_VEC]] : vector<1x1x1x64xf8E4M3FN> to vector<64xf8E4M3FN>
//       CHECK:   %[[B_CAST:.+]] = vector.shape_cast %[[B_VEC]] : vector<1x1x64x1xf8E4M3FN> to vector<64xf8E4M3FN>
//       CHECK:   %[[C_CAST:.+]] = vector.shape_cast %[[C_VEC]] : vector<1x1x8x1xf32> to vector<8xf32>
//       CHECK:   %[[WMMA:.+]] = amdgpu.wmma 16x16x128 %[[A_CAST]] * %[[B_CAST]]

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
    iree.amdgpu.mma = #iree_gpu.virtual_mma_layout<VMFMA_F32_32x32x16_F16>
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
// CHECK:       %[[MFMA_0:.*]] = amdgpu.mfma 32x32x8 %[[A_SLICE_0]] * %[[B_SLICE_0]] + %[[C_CAST]] blgp =  none
// CHECK-SAME:     : vector<4xf16>, vector<4xf16>, vector<16xf32>
// CHECK:       %[[A_SLICE_1:.+]] = vector.extract_strided_slice %[[A_CAST]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:       %[[B_SLICE_1:.+]] = vector.extract_strided_slice %[[B_CAST]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// CHECK:       %[[MFMA_1:.+]] = amdgpu.mfma 32x32x8 %[[A_SLICE_1]] * %[[B_SLICE_1]] + %[[MFMA_0]] blgp =  none
// CHECK-SAME:     : vector<4xf16>, vector<4xf16>, vector<16xf32>
// CHECK:       %[[R_CAST:.+]] = vector.shape_cast %[[MFMA_1]] : vector<16xf32> to vector<4x1x4x1xf32>
// CHECK:       %[[B_OUT:.*]] = vector.broadcast %[[R_CAST]] : vector<4x1x4x1xf32> to vector<1x1x4x1x4x1xf32>
// CHECK:       %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[B_OUT]] : vector<1x1x4x1x4x1xf32> -> vector<32x32xf32>
// CHECK:       return %[[R_SIMD]]

// -----

#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 1, 1],
  batch_tile = [1, 2, 2],
  outer_tile = [1, 1, 1],
  thread_tile = [16, 1, 4],
  element_tile = [1, 1, 8],

  subgroup_strides = [1, 0, 0],
  thread_strides = [1, 0, 16]
>

#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1],
  batch_tile = [2, 2, 2],
  outer_tile = [1, 1, 1],
  thread_tile = [16, 1, 4],
  element_tile = [1, 1, 8],

  subgroup_strides = [0, 0, 0],
  thread_strides = [1, 0, 16]
>

#layout_c = #iree_vector_ext.nested_layout<
  subgroup_tile = [2, 1],
  batch_tile = [1, 2],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [4, 1],

  subgroup_strides = [1, 0],
  thread_strides = [16, 1]
>

func.func @contract_with_3d_in_and_2d_out(%a: vector<32x2x64xf8E4M3FNUZ>, %b: vector<32x2x64xf8E4M3FNUZ>, %c : vector<32x32xf32>) -> vector<32x32xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<32x2x64xf8E4M3FNUZ>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<32x2x64xf8E4M3FNUZ>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<32x32xf32>

  %output = vector.contract {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d3)>],
    iterator_types = ["parallel", "reduction", "reduction", "parallel"],
    kind = #vector.kind<add>
  } %A, %B, %C {iree.amdgpu.mma = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>} : vector<32x2x64xf8E4M3FNUZ>, vector<32x2x64xf8E4M3FNUZ> into vector<32x32xf32>

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

// CHECK-LABEL: func @contract_with_3d_in_and_2d_out
// CHECK:       %[[A_CAST:.+]] = vector.shape_cast %{{.+}} : vector<1x1x1x1x1x8xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
// CHECK:       %[[B_CAST:.+]] = vector.shape_cast %{{.+}} : vector<1x1x1x1x1x8xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
// CHECK:       %[[C_CAST:.+]] = vector.shape_cast %{{.+}} : vector<1x1x4x1xf32> to vector<4xf32>
// CHECK:       %[[MFMA_0:.*]] = amdgpu.mfma 16x16x32 %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]] blgp =  none
// CHECK:       %[[A_CAST_1:.+]] = vector.shape_cast %{{.+}} : vector<1x1x1x1x1x8xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
// CHECK:       %[[B_CAST_1:.+]] = vector.shape_cast %{{.+}} : vector<1x1x1x1x1x8xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
// CHECK:       %[[MFMA_1:.*]] = amdgpu.mfma 16x16x32 %[[A_CAST_1]] * %[[B_CAST_1]] + %[[MFMA_0]] blgp =  none
// CHECK:       %[[MFMA_1_CAST:.*]] = vector.shape_cast %[[MFMA_1]] : vector<4xf32> to vector<1x1x4x1xf32>
// CHECK:       %[[B_CAST_2:.+]] = vector.shape_cast %{{.+}} : vector<1x1x1x1x1x8xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
// CHECK:       %[[C_CAST_1:.+]] = vector.shape_cast %{{.+}} : vector<1x1x4x1xf32> to vector<4xf32>
// CHECK:       %[[MFMA_2:.*]] = amdgpu.mfma 16x16x32 %[[A_CAST]] * %[[B_CAST_2]] + %[[C_CAST_1]] blgp =  none
// CHECK:       %[[B_CAST_3:.+]] = vector.shape_cast %{{.+}} : vector<1x1x1x1x1x8xf8E4M3FNUZ> to vector<8xf8E4M3FNUZ>
// CHECK:       %[[MFMA_3:.*]] = amdgpu.mfma 16x16x32 %[[A_CAST_1]] * %[[B_CAST_3]] + %[[MFMA_2]] blgp =  none
// CHECK:       %[[R_CAST:.+]] = vector.shape_cast %[[MFMA_3]] : vector<4xf32> to vector<1x1x4x1xf32>
// CHECK:       %[[R0:.+]]:4 = vector.to_elements %[[MFMA_1_CAST]] : vector<1x1x4x1xf32>
// CHECK:       %[[R1:.+]]:4 = vector.to_elements %[[R_CAST]] : vector<1x1x4x1xf32>
// CHECK:       %[[B_OUT:.+]] = vector.from_elements
// CHECK-SAME:    %[[R0]]#0, %[[R0]]#1, %[[R0]]#2, %[[R0]]#3
// CHECK-SAME:    %[[R1]]#0, %[[R1]]#1, %[[R1]]#2, %[[R1]]#3
// CHECK:       %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[B_OUT]] : vector<1x2x1x1x4x1xf32> -> vector<32x32xf32>
// CHECK:       return %[[R_SIMD]]
