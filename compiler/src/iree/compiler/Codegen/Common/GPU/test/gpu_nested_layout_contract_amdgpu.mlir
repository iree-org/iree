// RUN: iree-opt --split-input-file --iree-transform-dialect-interpreter --cse %s | FileCheck %s

// CDNA3 V_MFMA_F32_32x32x8_F16

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 32x8, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [32, 2],
  elements_per_thread     = [1, 4],

  subgroup_order          = [0, 1],
  batch_order             = [0, 1],
  outer_order             = [0, 1],
  thread_order            = [1, 0],
  element_order           = [0, 1]
>

// B: shape = 8x32, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [2, 32],
  elements_per_thread     = [4, 1],

  subgroup_order          = [0, 1],
  batch_order             = [0, 1],
  outer_order             = [0, 1],
  thread_order            = [0, 1],
  element_order           = [1, 0]
>

// C: vector<32x32>, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [4, 1],
  threads_per_outer       = [2, 32],
  elements_per_thread     = [4, 1],

  subgroup_order          = [0, 1],
  batch_order             = [0, 1],
  outer_order             = [0, 1],
  thread_order            = [0, 1],
  element_order           = [1, 0]
>

func.func @contract_to_mfma_32x32x8_mm(%a : vector<32x8xf16>, %b : vector<8x32xf16>, %c : vector<32x32xf32>) -> vector<32x32xf32> {
  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mfma = #iree_gpu.mfma_layout<F16_32x32x8_F32>,
    "__vector_layout_test_anchor_operand_0" = #layout_a,
    "__vector_layout_test_anchor_operand_1" = #layout_b,
    "__vector_layout_test_anchor_operand_2" = #layout_c,
    "__vector_layout_test_anchor_result_0" = #layout_c
  } %a, %b, %c : vector<32x8xf16>, vector<8x32xf16> into vector<32x32xf32>
  return %output : vector<32x32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @contract_to_mfma_32x32x8_mm
//  CHECK-SAME: (%[[A:.+]]: vector<32x8xf16>, %[[B:.+]]: vector<8x32xf16>, %[[C:.+]]: vector<32x32xf32>)
//       CHECK:   %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x1x1x4xf32>
//       CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %[[C]] : vector<32x32xf32> -> vector<1x1x4x1x1x4xf32>
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %[[A]] : vector<32x8xf16> -> vector<1x1x1x1x1x4xf16>
//       CHECK:   %[[B_SIMT:.+]] = iree_vector_ext.to_simt %[[B]] : vector<8x32xf16> -> vector<1x1x1x1x1x4xf16>
//       CHECK:   %[[C_VEC:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<4x1x1x4xf32> from vector<1x1x4x1x1x4xf32>
//       CHECK:   %[[A_VEC:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x4xf16> from vector<1x1x1x1x1x4xf16>
//       CHECK:   %[[B_VEC:.+]] = vector.extract %[[B_SIMT]][0, 0] : vector<1x1x1x4xf16> from vector<1x1x1x1x1x4xf16>
//       CHECK:   %[[A_CAST:.+]] = vector.shape_cast %[[A_VEC]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[B_CAST:.+]] = vector.shape_cast %[[B_VEC]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[C_CAST:.+]] = vector.shape_cast %[[C_VEC]] : vector<4x1x1x4xf32> to vector<4x4xf32>
//       CHECK:   %[[MFMA:.+]] = amdgpu.mfma %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]]
//  CHECK-SAME:     {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16>, vector<4x4xf32>
//       CHECK:   %[[R_CAST:.+]] = vector.shape_cast %[[MFMA]] : vector<4x4xf32> to vector<4x1x1x4xf32>
//       CHECK:   %[[INSERT:.+]] = vector.insert %[[R_CAST]], %[[INIT]] [0, 0] : vector<4x1x1x4xf32> into vector<1x1x4x1x1x4xf32>
//       CHECK:   %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[INSERT]] : vector<1x1x4x1x1x4xf32> -> vector<32x32xf32>
//       CHECK:   return {{.*}} %[[R_SIMD]]

// -----

// CDNA3 V_MFMA_F32_16X16X16_F16

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 16x16, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [16, 4],
  elements_per_thread     = [1, 4],

  subgroup_order          = [0, 1],
  batch_order             = [0, 1],
  outer_order             = [0, 1],
  thread_order            = [1, 0],
  element_order           = [0, 1]
>

// B: shape = 16x16, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [4, 16],
  elements_per_thread     = [4, 1],

  subgroup_order          = [0, 1],
  batch_order             = [0, 1],
  outer_order             = [0, 1],
  thread_order            = [0, 1],
  element_order           = [1, 0]
>

// C: shape = 16x16, layout = layoutB

func.func @contract_to_mfma_16x16x16_mm(%a : vector<16x16xf16>, %b : vector<16x16xf16>, %c : vector<16x16xf32>) -> vector<16x16xf32> {
  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mfma = #iree_gpu.mfma_layout<F16_16x16x16_F32>,
    "__vector_layout_test_anchor_operand_0" = #layout_a,
    "__vector_layout_test_anchor_operand_1" = #layout_b,
    "__vector_layout_test_anchor_operand_2" = #layout_b,
    "__vector_layout_test_anchor_result_0" = #layout_b
  } %a, %b, %c : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>
  return %output : vector<16x16xf32>
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
//       CHECK:   %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x1x1x1x4xf32>
//       CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %[[C]] : vector<16x16xf32> -> vector<1x1x1x1x1x4xf32>
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %[[A]] : vector<16x16xf16> -> vector<1x1x1x1x1x4xf16>
//       CHECK:   %[[B_SIMT:.+]] = iree_vector_ext.to_simt %[[B]] : vector<16x16xf16> -> vector<1x1x1x1x1x4xf16>
//       CHECK:   %[[C_VEC:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<1x1x1x4xf32> from vector<1x1x1x1x1x4xf32>
//       CHECK:   %[[A_VEC:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x4xf16> from vector<1x1x1x1x1x4xf16>
//       CHECK:   %[[B_VEC:.+]] = vector.extract %[[B_SIMT]][0, 0] : vector<1x1x1x4xf16> from vector<1x1x1x1x1x4xf16>
//       CHECK:   %[[A_CAST:.+]] = vector.shape_cast %[[A_VEC]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[B_CAST:.+]] = vector.shape_cast %[[B_VEC]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[C_CAST:.+]] = vector.shape_cast %[[C_VEC]] : vector<1x1x1x4xf32> to vector<4xf32>
//       CHECK:   %[[MFMA:.+]] = amdgpu.mfma %[[A_CAST]] * %[[B_CAST]] + %[[C_CAST]]
//  CHECK-SAME:     {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16>, vector<4xf32>
//       CHECK:   %[[R_CAST:.+]] = vector.shape_cast %[[MFMA]] : vector<4xf32> to vector<1x1x1x4xf32>
//       CHECK:   %[[INSERT:.+]] = vector.insert %[[R_CAST]], %[[INIT]] [0, 0] : vector<1x1x1x4xf32> into vector<1x1x1x1x1x4xf32>
//       CHECK:   %[[R_SIMD:.+]] = iree_vector_ext.to_simd %[[INSERT]] : vector<1x1x1x1x1x4xf32> -> vector<16x16xf32>
//       CHECK:   return {{.*}} %[[R_SIMD]]

// -----

// None-one M/N batch

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 32x8, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 1],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [32, 2],
  elements_per_thread     = [1, 4],

  subgroup_order          = [0, 1],
  batch_order             = [0, 1],
  outer_order             = [0, 1],
  thread_order            = [1, 0],
  element_order           = [0, 1]
>

// B: shape = 8x32, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [2, 32],
  elements_per_thread     = [4, 1],

  subgroup_order          = [0, 1],
  batch_order             = [0, 1],
  outer_order             = [0, 1],
  thread_order            = [0, 1],
  element_order           = [1, 0]
>

// C: vector<32x32>, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 1],
  outers_per_batch        = [4, 1],
  threads_per_outer       = [2, 32],
  elements_per_thread     = [4, 1],

  subgroup_order          = [0, 1],
  batch_order             = [0, 1],
  outer_order             = [0, 1],
  thread_order            = [0, 1],
  element_order           = [1, 0]
>

func.func @contract_to_mfma_32x32x8_mm_mnbatch(%a : vector<64x8xf16>, %b : vector<8x32xf16>, %c : vector<64x32xf32>) -> vector<64x32xf32> {
  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mfma = #iree_gpu.mfma_layout<F16_32x32x8_F32>,
    "__vector_layout_test_anchor_operand_0" = #layout_a,
    "__vector_layout_test_anchor_operand_1" = #layout_b,
    "__vector_layout_test_anchor_operand_2" = #layout_c,
    "__vector_layout_test_anchor_result_0" = #layout_c
  } %a, %b, %c : vector<64x8xf16>, vector<8x32xf16> into vector<64x32xf32>
  return %output : vector<64x32xf32>
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
//       CHECK:   %[[C_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<64x32xf32> -> vector<2x1x4x1x1x4xf32>
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<64x8xf16> -> vector<2x1x1x1x1x4xf16>
//       CHECK:   %[[C_SLICE0:.+]] = vector.extract %[[C_SIMT]][0, 0] : vector<4x1x1x4xf32> from vector<2x1x4x1x1x4xf32>
//       CHECK:   %[[A_SLICE0:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x4xf16> from vector<2x1x1x1x1x4xf16>
//       CHECK:   %[[A0_CAST:.+]] = vector.shape_cast %[[A_SLICE0]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[C0_CAST:.+]] = vector.shape_cast %[[C_SLICE0]] : vector<4x1x1x4xf32> to vector<4x4xf32>
//       CHECK:   %[[MFMA0:.+]] = amdgpu.mfma %[[A0_CAST]] * %{{.+}} + %[[C0_CAST]]
//       CHECK:   %[[R0_CAST:.+]] = vector.shape_cast %[[MFMA0]] : vector<4x4xf32> to vector<4x1x1x4xf32>
//       CHECK:   %[[C0_INS:.+]] = vector.insert %[[R0_CAST]], %[[INIT]] [0, 0] : vector<4x1x1x4xf32> into vector<2x1x4x1x1x4xf32>
//       CHECK:   %[[C_SLICE1:.+]] = vector.extract %[[C_SIMT]][1, 0] : vector<4x1x1x4xf32> from vector<2x1x4x1x1x4xf32>
//       CHECK:   %[[A_SLICE1:.+]] = vector.extract %[[A_SIMT]][1, 0] : vector<1x1x1x4xf16> from vector<2x1x1x1x1x4xf16>
//       CHECK:   %[[A1_CAST:.+]] = vector.shape_cast %[[A_SLICE1]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[C1_CAST:.+]] = vector.shape_cast %[[C_SLICE1]] : vector<4x1x1x4xf32> to vector<4x4xf32>
//       CHECK:   %[[MFMA1:.+]] = amdgpu.mfma %[[A1_CAST]] * %{{.+}} + %[[C1_CAST]]
//       CHECK:   %[[R1_CAST:.+]] = vector.shape_cast %[[MFMA1]] : vector<4x4xf32> to vector<4x1x1x4xf32>
//       CHECK:   %[[C1_INS:.+]] = vector.insert %[[R1_CAST]], %[[C0_INS]] [1, 0] : vector<4x1x1x4xf32> into vector<2x1x4x1x1x4xf32>
//       CHECK:   %[[R:.+]] = iree_vector_ext.to_simd %[[C1_INS]] : vector<2x1x4x1x1x4xf32> -> vector<64x32xf32>
//       CHECK:   return {{.*}}} %[[R]]

// -----

// None-one K batch

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 32x8, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [32, 2],
  elements_per_thread     = [1, 4],

  subgroup_order          = [0, 1],
  batch_order             = [0, 1],
  outer_order             = [0, 1],
  thread_order            = [1, 0],
  element_order           = [0, 1]
>

// B: shape = 8x32, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 1],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [2, 32],
  elements_per_thread     = [4, 1],

  subgroup_order          = [0, 1],
  batch_order             = [0, 1],
  outer_order             = [0, 1],
  thread_order            = [0, 1],
  element_order           = [1, 0]
>

// C: vector<32x32>, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [4, 1],
  threads_per_outer       = [2, 32],
  elements_per_thread     = [4, 1],

  subgroup_order          = [0, 1],
  batch_order             = [0, 1],
  outer_order             = [0, 1],
  thread_order            = [0, 1],
  element_order           = [1, 0]
>

func.func @contract_to_mfma_32x32x8_mm_kbatch(%a : vector<32x16xf16>, %b : vector<16x32xf16>, %c : vector<32x32xf32>) -> vector<32x32xf32> {
  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.amdgpu.mfma = #iree_gpu.mfma_layout<F16_32x32x8_F32>,
    "__vector_layout_test_anchor_operand_0" = #layout_a,
    "__vector_layout_test_anchor_operand_1" = #layout_b,
    "__vector_layout_test_anchor_operand_2" = #layout_c,
    "__vector_layout_test_anchor_result_0" = #layout_c
  } %a, %b, %c : vector<32x16xf16>, vector<16x32xf16> into vector<32x32xf32>
  return %output : vector<32x32xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @contract_to_mfma_32x32x8_mm_kbatch(%arg0: vector<32x16xf16>, %arg1: vector<16x32xf16>, %arg2: vector<32x32xf32>) -> vector<32x32xf32> {
//       CHECK:   %[[A_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<32x16xf16> -> vector<1x2x1x1x1x4xf16>
//       CHECK:   %[[B_SIMT:.+]] = iree_vector_ext.to_simt %{{.+}} : vector<16x32xf16> -> vector<2x1x1x1x1x4xf16>
//       CHECK:   %[[A_SLICE0:.+]] = vector.extract %[[A_SIMT]][0, 0] : vector<1x1x1x4xf16> from vector<1x2x1x1x1x4xf16>
//       CHECK:   %[[B_SLICE0:.+]] = vector.extract %[[B_SIMT]][0, 0] : vector<1x1x1x4xf16> from vector<2x1x1x1x1x4xf16>
//       CHECK:   %[[A0_CAST:.+]] = vector.shape_cast %[[A_SLICE0]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[B0_CAST:.+]] = vector.shape_cast %[[B_SLICE0]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[MFMA0:.+]] = amdgpu.mfma %[[A0_CAST]] * %[[B0_CAST]] + %{{.+}}
//       CHECK:   %[[R0_CAST:.+]] = vector.shape_cast %[[MFMA0]] : vector<4x4xf32> to vector<4x1x1x4xf32>
//       CHECK:   %[[A_SLICE1:.+]] = vector.extract %[[A_SIMT]][0, 1] : vector<1x1x1x4xf16> from vector<1x2x1x1x1x4xf16>
//       CHECK:   %[[B_SLICE1:.+]] = vector.extract %[[B_SIMT]][1, 0] : vector<1x1x1x4xf16> from vector<2x1x1x1x1x4xf16>
//       CHECK:   %[[A1_CAST:.+]] = vector.shape_cast %[[A_SLICE1]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[B1_CAST:.+]] = vector.shape_cast %[[B_SLICE1]] : vector<1x1x1x4xf16> to vector<4xf16>
//       CHECK:   %[[CAST:.+]] = vector.shape_cast %[[R0_CAST]] : vector<4x1x1x4xf32> to vector<4x4xf32>
//       CHECK:   %[[MFMA1:.+]] = amdgpu.mfma %[[A1_CAST]] * %[[B1_CAST]] + %[[CAST]]
//       CHECK:   %[[R_CAST:.+]] = vector.shape_cast %[[MFMA1]] : vector<4x4xf32> to vector<4x1x1x4xf32>
//       CHECK:   %[[INSERT:.+]] = vector.insert %[[R_CAST]], %{{.+}} [0, 0] : vector<4x1x1x4xf32> into vector<1x1x4x1x1x4xf32>
//       CHECK:   %[[R:.+]] = iree_vector_ext.to_simd %[[INSERT]] : vector<1x1x4x1x1x4xf32> -> vector<32x32xf32>
//       CHECK:   return {{.*}} %[[R]] : vector<32x32xf32>
