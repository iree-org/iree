// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-dispatch-creation-pipeline)" \
// RUN:          --iree-dispatch-creation-experimental-data-tiling %s | FileCheck %s

// Tests to make sure that the set encoding pass work as in the dispatch
// creation pipeline. For example, we expect dimension collapsing to happen
// before we set encoding.

util.func public @multi_k_dim_generic(%arg0: tensor<256x64x2xf32>, %arg1: tensor<64x2x512xf32>,
                                      %arg2: tensor<256x512xf32>) -> tensor<256x512xf32> {
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
        ins(%arg0, %arg1: tensor<256x64x2xf32>, tensor<64x2x512xf32>) outs(%arg2: tensor<256x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<256x512xf32>
  util.return %4 : tensor<256x512xf32>
}

// The reduction dimensions should be collapsed to a single one.

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[ENCODING_LHS:.+]] = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-DAG: #[[ENCODING_RHS:.+]] = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-DAG: #[[ENCODING_OUT:.+]] = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]

//      CHECK: util.func public @multi_k_dim_generic(
//      CHECK:   linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<256x128xf32, #[[ENCODING_LHS]]>, tensor<128x512xf32, #[[ENCODING_RHS]]>)
// CHECK-SAME:      outs(%{{.*}} : tensor<256x512xf32, #[[ENCODING_OUT]]>)

// -----

util.func public @multi_batch_dim_generic(%arg0: tensor<4x8x256x128xf32>, %arg1: tensor<4x8x128x512xf32>,
                                          %arg2: tensor<4x8x256x512xf32>) -> tensor<4x8x256x512xf32> {
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%arg0, %arg1: tensor<4x8x256x128xf32>, tensor<4x8x128x512xf32>) outs(%arg2: tensor<4x8x256x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<4x8x256x512xf32>
  util.return %4 : tensor<4x8x256x512xf32>
}

// The batch dimensions should be collapsed to a single one.

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[ENCODING_LHS:.+]] = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-DAG: #[[ENCODING_RHS:.+]] = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-DAG: #[[ENCODING_OUT:.+]] = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]

//      CHECK: util.func public @multi_batch_dim_generic(
//      CHECK:   linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<32x256x128xf32, #[[ENCODING_LHS]]>, tensor<32x128x512xf32, #[[ENCODING_RHS]]>)
// CHECK-SAME:      outs(%{{.*}} : tensor<32x256x512xf32, #[[ENCODING_OUT]]>)

// -----

util.func public @broadcast_rhs_batch_mmt(%arg0: tensor<16x1024x1280xi8>, %arg1: tensor<10240x1280xi8>,
                                          %arg2: tensor<16x1024x10240xi32>) -> tensor<16x1024x10240xi32> {
  %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                                         affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1: tensor<16x1024x1280xi8>, tensor<10240x1280xi8>) outs(%arg2 : tensor<16x1024x10240xi32>) {
  ^bb0(%in: i8, %in_0: i8, %acc: i32):
    %22 = arith.extsi %in : i8 to i32
    %23 = arith.extsi %in_0 : i8 to i32
    %24 = arith.muli %22, %23 : i32
    %25 = arith.addi %acc, %24 : i32
    linalg.yield %25 : i32
  } -> tensor<16x1024x10240xi32>
  util.return %20: tensor<16x1024x10240xi32>
}

// The batch and M dimension should be collapsed.

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[ENCODING_LHS:.+]] = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-DAG: #[[ENCODING_RHS:.+]] = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-DAG: #[[ENCODING_OUT:.+]] = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]

//      CHECK: util.func public @broadcast_rhs_batch_mmt(
//      CHECK:   linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<16384x1280xi8, #[[ENCODING_LHS]]>, tensor<10240x1280xi8, #[[ENCODING_RHS]]>)
// CHECK-SAME:      outs(%{{.*}} : tensor<16384x10240xi32, #[[ENCODING_OUT]]>)

// -----

// The below tests that the encoding ops are hoisted to util.initializer.

util.func public @foo(%arg0: tensor<255x513xf32>) -> tensor<255x1023xf32> {
  %cst = arith.constant dense<1.000000e+00> : tensor<513x1023xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<255x1023xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<255x1023xf32>) -> tensor<255x1023xf32>
  %2 = linalg.matmul ins(%arg0, %cst : tensor<255x513xf32>, tensor<513x1023xf32>) outs(%1 : tensor<255x1023xf32>) -> tensor<255x1023xf32>
  util.return %2 : tensor<255x1023xf32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:  #[[LHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-DAG:  #[[RHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-DAG:  #[[RES_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK:      util.global private @[[HOISTED:.+]] : tensor<513x1023xf32, #[[RHS_ENCODING]]>
// CHECK:      util.initializer {
// CHECK:        %[[CST:.+]] = arith.constant dense<1.000000e+00> : tensor<513x1023xf32>
// CHECK:        %[[ENCODED_CST:.+]] = flow.tensor.encode %[[CST]] : tensor<513x1023xf32> -> tensor<513x1023xf32, #[[RHS_ENCODING]]>
// CHECK:        util.global.store %[[ENCODED_CST]], @[[HOISTED]]
