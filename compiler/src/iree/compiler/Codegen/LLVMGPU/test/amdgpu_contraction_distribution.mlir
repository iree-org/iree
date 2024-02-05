// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --cse %s | FileCheck %s

#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0)>
#row_layout = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX], [1, 16]>
#col_layout = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [1, 4, 4]>
#row_layout2 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEY, VECTORX], [1, 4, 4]>
#col_layout2 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEX], [1, 16]>
#layout_a = #iree_vector_ext.layout<#row_layout, #col_layout>
#layout_c = #iree_vector_ext.layout<#row_layout2, #col_layout2>
builtin.module attributes { transform.with_named_sequence } {
  func.func @distribute_mfma_16x16x16_mmt(%a : vector<16x16xf16>, %b : vector<16x16xf16>, %c : vector<16x16xf32>) -> vector<16x16xf32> {
    // CHECK-LABEL: distribute_mfma_16x16x16_mmt
    // CHECK-SAME: %[[ARG0:.+]]: vector<16x16xf16>, %[[ARG1:.+]]: vector<16x16xf16>, %[[ARG2:.+]]: vector<16x16xf32>
    // CHECK-DAG: %[[C:.+]] = iree_vector_ext.to_simt %[[ARG2]] : vector<16x16xf32> -> vector<1x1x4xf32>
    // CHECK-DAG: %[[CV:.+]] = vector.extract %[[C]][0, 0] : vector<4xf32> from vector<1x1x4xf32>
    // CHECK-DAG: %[[A:.+]] = iree_vector_ext.to_simt %[[ARG0]] : vector<16x16xf16> -> vector<1x1x4xf16>
    // CHECK-DAG: %[[AV:.+]] = vector.extract %[[A]][0, 0] : vector<4xf16> from vector<1x1x4xf16>
    // CHECK-DAG: %[[B:.+]] = iree_vector_ext.to_simt %[[ARG1]] : vector<16x16xf16> -> vector<1x1x4xf16>
    // CHECK-DAG: %[[BV:.+]] = vector.extract %[[B]][0, 0] : vector<4xf16> from vector<1x1x4xf16>
    // CHECK-DAG: %[[OUT:.+]] = amdgpu.mfma %[[AV]] * %[[BV]] + %[[CV]] {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    %output = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"],
                               kind = #vector.kind<add>,
                               "__vector_layout_test_anchor_operand_0" = #layout_a,
                               "__vector_layout_test_anchor_operand_1" = #layout_c,
                               "__vector_layout_test_anchor_operand_2" = #layout_c,
                               "__vector_layout_test_anchor_result_0" = #layout_c
                               }
                                %a, %b, %c : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>
    return %output : vector<16x16xf32>
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_amdgpu_contraction_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#row_layout = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX], [1, 32]>
#col_layout = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [1, 2, 4]>
#row_layout1 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEY, VECTORX], [1, 2, 4]>
#col_layout1 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEX], [1, 32]>
#row_layout2 = #iree_vector_ext.per_dim_layout<[BATCHX, VECTORY, LANEY, VECTORX], [1, 4, 2, 4]>
#col_layout2 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEX], [1, 32]>
#layout_a = #iree_vector_ext.layout<#row_layout, #col_layout>
#layout_b = #iree_vector_ext.layout<#row_layout1, #col_layout1>
#layout_c = #iree_vector_ext.layout<#row_layout2, #col_layout2>
builtin.module attributes { transform.with_named_sequence } {
  func.func @distribute_mfma_32x32x8_mm(%a : vector<32x8xf16>, %b : vector<8x32xf16>, %c : vector<32x32xf32>) -> vector<32x32xf32> {
    // CHECK-LABEL: distribute_mfma_32x32x8_mm
    // CHECK-SAME: %[[ARG0:.+]]: vector<32x8xf16>, %[[ARG1:.+]]: vector<8x32xf16>, %[[ARG2:.+]]: vector<32x32xf32>
    // CHECK-DAG: %[[C:.+]] = iree_vector_ext.to_simt %[[ARG2]] : vector<32x32xf32> -> vector<1x1x16xf32>
    // CHECK-DAG: %[[CV:.+]] = vector.extract %[[C]][0, 0] : vector<16xf32> from vector<1x1x16xf32>
    // CHECK-DAG: %[[A:.+]] = iree_vector_ext.to_simt %[[ARG0]] : vector<32x8xf16> -> vector<1x1x4xf16>
    // CHECK-DAG: %[[AV:.+]] = vector.extract %[[A]][0, 0] : vector<4xf16> from vector<1x1x4xf16>
    // CHECK-DAG: %[[B:.+]] = iree_vector_ext.to_simt %[[ARG1]] : vector<8x32xf16> -> vector<1x1x4xf16>
    // CHECK-DAG: %[[BV:.+]] = vector.extract %[[B]][0, 0] : vector<4xf16> from vector<1x1x4xf16>
    // CHECK-DAG: %[[OUT:.+]] = amdgpu.mfma %[[AV]] * %[[BV]] + %[[CV]] {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
    %output = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"],
                               kind = #vector.kind<add>,
                               "__vector_layout_test_anchor_operand_0" = #layout_a,
                               "__vector_layout_test_anchor_operand_1" = #layout_b,
                               "__vector_layout_test_anchor_operand_2" = #layout_c,
                               "__vector_layout_test_anchor_result_0" = #layout_c
                               }
                                %a, %b, %c : vector<32x8xf16>, vector<8x32xf16> into vector<32x32xf32>
    return %output : vector<32x32xf32>
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_amdgpu_contraction_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0)>
#row_layout = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX], [2, 16]>
#col_layout = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [8, 4, 4]>
#row_layout2 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEY, VECTORX], [4, 4, 4]>
#col_layout2 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEX], [8, 16]>
#row_layout3 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEY, VECTORX], [2, 4, 4]>
#col_layout3 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEX], [4, 16]>
#layout_a = #iree_vector_ext.layout<#row_layout, #col_layout>
#layout_b = #iree_vector_ext.layout<#row_layout2, #col_layout2>
#layout_c = #iree_vector_ext.layout<#row_layout3, #col_layout3>
builtin.module attributes { transform.with_named_sequence } {
  func.func @distribute_mfma_16x16x16_mmt_batch(%a : vector<32x128xf16>, %b : vector<64x128xf16>, %c : vector<32x64xf32>) -> vector<32x64xf32> {
    // CHECK-LABEL: distribute_mfma_16x16x16_mmt_batch
    // CHECK-COUNT-64: amdgpu.mfma {{.*}}, vector<4xf32>
    %output = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"],
                               kind = #vector.kind<add>,
                               "__vector_layout_test_anchor_operand_0" = #layout_a,
                               "__vector_layout_test_anchor_operand_1" = #layout_b,
                               "__vector_layout_test_anchor_operand_2" = #layout_c,
                               "__vector_layout_test_anchor_result_0" = #layout_c
                               }
                                %a, %b, %c : vector<32x128xf16>, vector<64x128xf16> into vector<32x64xf32>
    return %output : vector<32x64xf32>
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_amdgpu_contraction_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
