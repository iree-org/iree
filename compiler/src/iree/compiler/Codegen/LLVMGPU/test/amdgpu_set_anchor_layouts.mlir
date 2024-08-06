// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --cse %s --verify-diagnostics

// This tests that the compiler is setting the correct layout anchors for various vectorOps and shapes.
// Currently only testing on contraction layoutV1, but can be expanded to others.

#layout = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0)>

builtin.module attributes { transform.with_named_sequence } {
  func.func @anchor_mfma_16x16x16_mmt(%a : memref<16x16xf16>, %b : memref<16x16xf16>, %init : vector<16x16xf32>) -> vector<16x16xf32> {
    // CHECK-LABEL: anchor_mfma_16x16x16_mmt
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %lhs = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHX,  LANEX], [1, 16]>, <[ BATCHY,  LANEY,  VECTORX], [1, 4, 4]>>}}
    %rhs = vector.transfer_read %b[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHX,  LANEX], [1, 16]>, <[ BATCHY,  LANEY,  VECTORX], [1, 4, 4]>>}}
    %output = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %init : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHX,  LANEY,  VECTORX], [1, 4, 4]>, <[ BATCHY,  LANEX], [1, 16]>>}}
    return %output : vector<16x16xf32>
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %contract = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %layout16x16x16 = transform.param.constant #layout -> !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract, %layout16x16x16 : !transform.any_op, !transform.any_param

    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0)>

builtin.module attributes { transform.with_named_sequence } {
  func.func @anchor_mfma_16x16x16_mmt_batch(%a : memref<32x128xf16>, %b : memref<64x128xf16>, %init : vector<32x64xf32>) -> vector<32x64xf32> {
    // CHECK-LABEL: anchor_mfma_16x16x16_mmt_batch
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %lhs = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x128xf16>, vector<32x128xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHX,  LANEX], [2, 16]>, <[ BATCHY,  LANEY,  VECTORX], [8, 4, 4]>>}}
    %rhs = vector.transfer_read %b[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<64x128xf16>, vector<64x128xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHX,  LANEX], [4, 16]>, <[ BATCHY,  LANEY,  VECTORX], [8, 4, 4]>>}}
    %output = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %init : vector<32x128xf16>, vector<64x128xf16> into vector<32x64xf32>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHX,  LANEY,  VECTORX], [2, 4, 4]>, <[ BATCHY,  LANEX], [4, 16]>>}}
    return %output : vector<32x64xf32>
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %contract = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %layout16x16x16 = transform.param.constant #layout -> !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract, %layout16x16x16 : !transform.any_op, !transform.any_param

    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F32>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0)>

builtin.module attributes { transform.with_named_sequence } {
  func.func @anchor_wmma_16x16x16_mmt(%a : memref<16x16xf16>, %b : memref<16x16xf16>, %init : vector<16x16xf32>) -> vector<16x16xf32> {
    // CHECK-LABEL: anchor_wmma_16x16x16_mmt
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %lhs = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHX,  LANEX], [1, 16]>, <[ BATCHY,  LANEY,  VECTORX], [1, 1, 16]>>}}
    %rhs = vector.transfer_read %b[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHX,  LANEX], [1, 16]>, <[ BATCHY,  LANEY,  VECTORX], [1, 1, 16]>>}}
    %output = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %init : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf32>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHX,  VECTORY,  LANEY,  VECTORX], [1, 8, 2, 1]>, <[ BATCHY,  LANEX], [1, 16]>>}}
    return %output : vector<16x16xf32>
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %contract = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %layout16x16x16 = transform.param.constant #layout -> !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract, %layout16x16x16 : !transform.any_op, !transform.any_param

    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}
