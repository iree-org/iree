// RUN: iree-opt --iree-dispatch-creation-hoist-encoding-ops --split-input-file %s | FileCheck %s

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#lhs_encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3]>
#rhs_encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3]>
#result_encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map1, #map2, #map3]>
module {
  util.func public @hoist_matmul_encodings(%arg0: tensor<2x128x64xf32>, %arg1: tensor<2x11008x128xf32>) -> tensor<2x11008x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %2 = flow.dispatch.region -> (tensor<2x11008x64xf32>) {
      %3 = iree_encoding.set_encoding %arg0 : tensor<2x128x64xf32> -> tensor<2x128x64xf32, #lhs_encoding>
      %4 = iree_encoding.set_encoding %arg1 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #rhs_encoding>
      %5 = tensor.empty() : tensor<2x11008x64xf32, #result_encoding>
      %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x11008x64xf32, #result_encoding>) -> tensor<2x11008x64xf32, #result_encoding>
      %7 = linalg.generic {
          indexing_maps = [#map1, #map2, #map3],
          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
          ins(%3, %4 : tensor<2x128x64xf32, #lhs_encoding>, tensor<2x11008x128xf32, #rhs_encoding>)
          outs(%6 : tensor<2x11008x64xf32, #result_encoding>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %9 = arith.mulf %in, %in_0 : f32
        %10 = arith.addf %9, %out : f32
        linalg.yield %10 : f32
      } -> tensor<2x11008x64xf32, #result_encoding>
      %8 = iree_encoding.unset_encoding %7 : tensor<2x11008x64xf32, #result_encoding> -> tensor<2x11008x64xf32>
      flow.return %8 : tensor<2x11008x64xf32>
    }
    util.return %2 : tensor<2x11008x64xf32>
  }
}

// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG:   #[[$LHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>
// CHECK-DAG:   #[[$RHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>
// CHECK-DAG:   #[[$OUT_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]>
// CHECK-LABEL: @hoist_matmul_encodings
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<2x128x64xf32>, %[[ARG1:.+]]: tensor<2x11008x128xf32>)
// CHECK-DAG:   %[[SET_ENCODING0:.+]] = iree_encoding.set_encoding %[[ARG0]] : tensor<2x128x64xf32> -> tensor<2x128x64xf32, #[[$LHS_ENCODING]]>
// CHECK-DAG:   %[[SET_ENCODING1:.+]] = iree_encoding.set_encoding %[[ARG1]] : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #[[$RHS_ENCODING]]>
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<2x11008x64xf32>) {
// CHECK:         %[[MATMUL:.+]] = linalg.generic {{.*}} ins(%[[SET_ENCODING0]], %[[SET_ENCODING1]]
// CHECK:         %[[UNSET_ENCODING1:.+]] = iree_encoding.unset_encoding %[[MATMUL]] : tensor<2x11008x64xf32, #[[$OUT_ENCODING]]>
// CHECK:         flow.return %[[UNSET_ENCODING1]] : tensor<2x11008x64xf32>
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]] : tensor<2x11008x64xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map2, #map3, #map4]>
util.func public @bubble_through_dequant(
    %arg0: tensor<2x11008x128xi8>, %arg1: tensor<2x11008xf32>, %arg2: tensor<2x11008xf32>) -> tensor<2x11008x128xf32, #encoding> {
  %6 = flow.dispatch.region -> (tensor<2x11008x128xf32, #encoding>) {
    %8 = tensor.empty() : tensor<2x11008x128xf32>
    %11 = linalg.generic
        {indexing_maps = [#map, #map1, #map1, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg1, %arg2 : tensor<2x11008x128xi8>, tensor<2x11008xf32>, tensor<2x11008xf32>)
        outs(%8 : tensor<2x11008x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %18 = arith.extui %in : i8 to i32
      %19 = arith.uitofp %18 : i32 to f32
      %20 = arith.subf %19, %in_1 : f32
      %21 = arith.mulf %20, %in_0 : f32
      linalg.yield %21 : f32
    } -> tensor<2x11008x128xf32>
    %13 = iree_encoding.set_encoding %11 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
    flow.return %13 : tensor<2x11008x128xf32, #encoding>
  }
  util.return %6 : tensor<2x11008x128xf32, #encoding>
}

// CHECK-DAG:   #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:   #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]>
// CHECK-DAG:   #[[$ENCODING_IBMAP:.+]] = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP]], [#[[MAP1]], #[[MAP3]]], #[[MAP2]]]>
// CHECK-DAG:   #[[$ENCODING_BMAP:.+]] = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP]], [#[[MAP1]], #[[MAP4]]], #[[MAP2]]]>
// CHECK-LABEL: @bubble_through_dequant
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x11008x128xi8>,
// CHECK-SAME:    %[[ARG1:.+]]: tensor<2x11008xf32>, %[[ARG2:.+]]: tensor<2x11008xf32>
// CHECK-DAG:   %[[SET_ENCODING0:.+]] = iree_encoding.set_encoding %[[ARG0]] : tensor<2x11008x128xi8> -> tensor<2x11008x128xi8, #[[$ENCODING_IBMAP]]>
// CHECK-DAG:   %[[SET_ENCODING1:.+]] = iree_encoding.set_encoding %[[ARG1]] : tensor<2x11008xf32> -> tensor<2x11008xf32, #[[$ENCODING_BMAP]]>
// CHECK-DAG:   %[[SET_ENCODING2:.+]] = iree_encoding.set_encoding %[[ARG2]] : tensor<2x11008xf32> -> tensor<2x11008xf32, #[[$ENCODING_BMAP]]>
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<2x11008x128xf32, #[[$ENCODING]]>
// CHECK:         %[[DEQUANT:.+]] = linalg.generic {{.*}} ins(%[[SET_ENCODING0]], %[[SET_ENCODING1]], %[[SET_ENCODING2]] : {{.*}} outs(%[[INIT]] :
// CHECK:         flow.return %[[DEQUANT]]
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]>
util.func public @bubble_through_broadcast(
    %arg0: tensor<11008x128xf32>) -> tensor<2x11008x128xf32, #encoding> {
  %6 = flow.dispatch.region -> (tensor<2x11008x128xf32, #encoding>) {
    %8 = tensor.empty() : tensor<2x11008x128xf32>
    %11 = linalg.generic
        {indexing_maps = [#map1, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<11008x128xf32>)
        outs(%8 : tensor<2x11008x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x11008x128xf32>
    %13 = iree_encoding.set_encoding %11 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
    flow.return %13 : tensor<2x11008x128xf32, #encoding>
  }
  util.return %6 : tensor<2x11008x128xf32, #encoding>
}

// CHECK-DAG:   #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:   #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]>
// CHECK-DAG:   #[[$ENCODING_BMAP:.+]] = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP]], [#[[MAP1]], #[[MAP3]]], #[[MAP2]]]>
// CHECK-LABEL: @bubble_through_broadcast
// CHECK-SAME:    %[[ARG0:.+]]: tensor<11008x128xf32>
// CHECK-DAG:   %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[ARG0]] : tensor<11008x128xf32> -> tensor<11008x128xf32, #[[$ENCODING_BMAP]]>
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<2x11008x128xf32, #[[$ENCODING]]>
// CHECK:         %[[BROADCAST:.+]] = linalg.generic {{.*}} ins(%[[SET_ENCODING]] : {{.*}} outs(%[[INIT]] :
// CHECK:         flow.return %[[BROADCAST]]
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
module {
  util.func public @no_hoist_if_source_is_compute_op(%arg0: tensor<2x11008x128xf32>) -> tensor<2x11008x128xf32, #encoding> {
    %0 = flow.dispatch.region -> (tensor<2x11008x128xf32, #encoding>) {
      %1 = tensor.empty() : tensor<2x11008x128xf32>
      %2 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg0 : tensor<2x11008x128xf32>, tensor<2x11008x128xf32>) outs(%1 : tensor<2x11008x128xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %4 = arith.addf %in, %in_0 : f32
        linalg.yield %4 : f32
      } -> tensor<2x11008x128xf32>
      %3 = iree_encoding.set_encoding %2 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
      flow.return %3 : tensor<2x11008x128xf32, #encoding>
    }
    util.return %0 : tensor<2x11008x128xf32, #encoding>
  }
}

// CHECK-LABEL: @no_hoist_if_source_is_compute_op
// CHECK-SAME:    %[[ARG0:.+]]: tensor<2x11008x128xf32>
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<2x11008x128xf32>
// CHECK:         %[[ADD:.+]] = linalg.generic {{.*}} ins(%[[ARG0]], %[[ARG0]] : {{.*}} outs(%[[INIT]] :
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[ADD]]
// CHECK:         flow.return %[[SET_ENCODING]]
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]]

// -----

#encoding = #iree_encoding.testing<>
util.func private @get_tensor(tensor<640x320xf32, #encoding>) -> tensor<640x320xf32>
util.func public @hoist_both_src_and_encoding() -> tensor<640x320xf32> {
  %0 = flow.dispatch.region -> (tensor<640x320xf32>) {
    %cst = arith.constant dense<1.0> : tensor<640x320xf32>
    %1 = iree_encoding.set_encoding %cst : tensor<640x320xf32> -> tensor<640x320xf32, #encoding>
    %2 = util.call @get_tensor(%1) : (tensor<640x320xf32, #encoding>) -> tensor<640x320xf32>
    flow.return %2 : tensor<640x320xf32>
  }
  util.return %0 : tensor<640x320xf32>
}
// CHECK-LABEL: util.func public @hoist_both_src_and_encoding(
// CHECK:         %[[CST:.+]] = arith.constant
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[CST]]
// CHECK:         flow.dispatch.region

// -----

// It tests that the constant-like ops within the dispatch are hoisted out.

#encoding = #iree_encoding.testing<>
util.global private @weight : tensor<640x320xf32>
util.func private @get_tensor(tensor<640x320xf32, #encoding>) -> tensor<640x320xf32>
util.func public @hoist_encoding_const_exprs() -> tensor<640x320xf32> {
  %init = tensor.empty() : tensor<640x320xf32>
  %0 = flow.dispatch.region -> (tensor<640x320xf32>) {
    %input = util.global.load @weight : tensor<640x320xf32>
    %1 = linalg.elementwise kind=#linalg.elementwise_kind<exp>
      ins(%input: tensor<640x320xf32>)
      outs(%init: tensor<640x320xf32>)
      -> tensor<640x320xf32>
    %2 = iree_encoding.set_encoding %1 : tensor<640x320xf32> -> tensor<640x320xf32, #encoding>
    %3 = util.call @get_tensor(%2) : (tensor<640x320xf32, #encoding>) -> tensor<640x320xf32>
    flow.return %3 : tensor<640x320xf32>
  }
  util.return %0 : tensor<640x320xf32>
}

// CHECK-LABEL: util.func public @hoist_encoding_const_exprs(
// CHECK:         %[[LOAD:.+]] = util.global.load
// CHECK:         %[[SRC:.+]] = linalg.elementwise
// CHECK-SAME:      ins(%[[LOAD]]
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]]
// CHECK:         flow.dispatch.region

// -----

#encoding = #iree_encoding.testing<>
util.func private @get_tensor(tensor<640x320xf32, #encoding>) -> tensor<640x320xf32>
util.func public @hoist_convertable_slice_op(%input: tensor<1024x320xf32>) -> tensor<640x320xf32> {
  %0 = flow.dispatch.region -> (tensor<640x320xf32>) {
    %1 = tensor.extract_slice %input[7, 0] [640, 320] [1, 1] : tensor<1024x320xf32> to tensor<640x320xf32>
    %2 = iree_encoding.set_encoding %1 : tensor<640x320xf32> -> tensor<640x320xf32, #encoding>
    %3 = util.call @get_tensor(%2) : (tensor<640x320xf32, #encoding>) -> tensor<640x320xf32>
    flow.return %3 : tensor<640x320xf32>
  }
  util.return %0 : tensor<640x320xf32>
}
// CHECK-LABEL: util.func public @hoist_convertable_slice_op(
// CHECK:         %[[SRC:.+]] = tensor.extract_slice
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]]
// CHECK:         flow.dispatch.region

// -----

// Avoid hoisting `set_encoding` operations that have pad encodings

#encoding = #iree_encoding.padding<[0, ?]>
util.func public @dont_hoist_pad_encoding() -> tensor<640x320xf32, #encoding> {
  %0 = flow.dispatch.region -> (tensor<640x320xf32, #encoding>) {
    %1 = tensor.empty() : tensor<640x320xf32>
    %2 = iree_encoding.set_encoding %1 : tensor<640x320xf32> -> tensor<640x320xf32, #encoding>
    flow.return %2 : tensor<640x320xf32, #encoding>
  }
  util.return %0 : tensor<640x320xf32, #encoding>
}
// CHECK-LABEL: @dont_hoist_pad_encoding
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[SET_ENCODING:.+]] = iree_encoding.set_encoding
//       CHECK:     flow.return %[[SET_ENCODING]]
//       CHECK:   return %[[DISPATCH]]

// -----

// Avoid hoisting `set_encoding` operations on scalar tensors.

#encoding = #iree_encoding.testing<>
util.func private @get_tensor(tensor<f32, #encoding>) -> tensor<f32>
util.func public @dont_hoist_encoding_on_scalar_tensor(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = flow.dispatch.region -> (tensor<f32>) {
    %1 = iree_encoding.set_encoding %arg0 : tensor<f32> -> tensor<f32, #encoding>
    %2 = util.call @get_tensor(%1) : (tensor<f32, #encoding>) -> tensor<f32>
    flow.return %2 : tensor<f32>
  }
  util.return %0 : tensor<f32>
}
// CHECK-LABEL: @dont_hoist_encoding_on_scalar_tensor
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[SET_ENCODING:.+]] = iree_encoding.set_encoding
//       CHECK:     %[[CALL:.+]] = util.call @get_tensor(%[[SET_ENCODING]])
//       CHECK:     flow.return %[[CALL]]
//       CHECK:   return %[[DISPATCH]]

// -----

// Tests the propagation of the unset encoding op through a parallel generic op.
// Note the permuted target operand indexing map that needs to be added to the
// encoding attribute to map the encoding to the domain of the generic op.

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0, d1) -> ()>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]>
util.func public @propagate_unset_encoding_through_generic(%arg0: tensor<?x4096xf32, #encoding>, %arg1: tensor<f32>, %arg2: index) -> tensor<4096x?xbf16> {
  %0 = flow.dispatch.region -> (tensor<4096x?xbf16>{%arg2}) {
    %1 = iree_encoding.unset_encoding %arg0 : tensor<?x4096xf32, #encoding> -> tensor<?x4096xf32>{%arg2}
    %2 = tensor.empty(%arg2) : tensor<4096x?xbf16>
    %3 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel"]} ins(%1, %arg1 : tensor<?x4096xf32>, tensor<f32>) outs(%2 : tensor<4096x?xbf16>) {
    ^bb0(%in: f32, %in_0: f32, %out: bf16):
      %4 = arith.mulf %in, %in_0 : f32
      %5 = arith.truncf %4 : f32 to bf16
      linalg.yield %5 : bf16
    } -> tensor<4096x?xbf16>
    flow.return %3 : tensor<4096x?xbf16>
  }
  util.return %0 : tensor<4096x?xbf16>
}
// CHECK-DAG:   #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG:   #[[MAP4:.+]] = affine_map<(d0, d1) -> ()>
// CHECK-DAG:   #[[MAP5:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]>
// CHECK-DAG:   #[[$ENCODING1:.+]] = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#[[MAP]], #[[MAP1]], [#[[MAP2]], #[[MAP3]], #[[MAP4]]]]>
// CHECK-DAG:   #[[$ENCODING2:.+]] = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#[[MAP]], #[[MAP1]], [#[[MAP2]], #[[MAP3]]]]>
// CHECK-LABEL: @propagate_unset_encoding_through_generic
// CHECK-SAME:    %[[ARG0:.+]]: tensor<?x4096xf32, #[[$ENCODING]]>, %[[ARG1:.+]]: tensor<f32>, %[[ARG2:.+]]: index
// CHECK:         flow.dispatch.region -> (tensor<4096x?xbf16>{%[[ARG2]]}) {
// CHECK:           %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[ARG1]] : tensor<f32> -> tensor<f32, #[[$ENCODING1]]>
// CHECK:           %[[EMPTY:.+]] = tensor.empty(%[[ARG2]]) : tensor<4096x?xbf16, #[[$ENCODING2]]>
// CHECK:           %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[ARG0]], %[[SET_ENCODING]] :  tensor<?x4096xf32, #[[$ENCODING]]>, tensor<f32, #[[$ENCODING1]]>)
// CHECK-SAME:        outs(%[[EMPTY]] :  tensor<4096x?xbf16, #[[$ENCODING2]]>)
// CHECK:           %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[GENERIC]] : tensor<4096x?xbf16, #[[$ENCODING2]]> -> tensor<4096x?xbf16>{%[[ARG2]]}
// CHECK:           flow.return %[[UNSET_ENCODING:.+]] : tensor<4096x?xbf16>
// CHECK:         }

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> ()>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
util.func public @propagate_unset_encoding_through_generic_with_scalar(%arg0: tensor<4096x?xf32, #encoding>, %arg1: f32, %arg2: index) -> tensor<4096x?xf32> {
  %0 = flow.dispatch.region -> (tensor<4096x?xf32>{%arg2}) {
    %1 = iree_encoding.unset_encoding %arg0 : tensor<4096x?xf32, #encoding> -> tensor<4096x?xf32>{%arg2}
    %2 = tensor.empty(%arg2) : tensor<4096x?xf32>
    %3 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%1, %arg1 : tensor<4096x?xf32>, f32) outs(%2 : tensor<4096x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.mulf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<4096x?xf32>
    flow.return %3 : tensor<4096x?xf32>
  }
  util.return %0 : tensor<4096x?xf32>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>
// CHECK-LABEL: @propagate_unset_encoding_through_generic_with_scalar(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK:         %{{.+}} = flow.dispatch.region -> (tensor<4096x?xf32>{%[[ARG2]]}
// CHECK:           %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:        ins(%[[ARG0]], %[[ARG1]]
// CHECK:           %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[GENERIC]] : tensor<4096x?xf32, #[[$ENCODING]]> -> tensor<4096x?xf32>{%[[ARG2]]}
// CHECK:           return %[[UNSET_ENCODING]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> ()>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]>
util.func public @dont_propagate_unset_encoding_outside_region(%arg0: tensor<?x4096xf32, #encoding>, %arg1: tensor<f32>, %arg2: index) -> tensor<?x4096xbf16> {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<?x4096xf32, #encoding> -> tensor<?x4096xf32>{%arg2}
  %1 = tensor.empty(%arg2) : tensor<?x4096xbf16>
  %2 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%0, %arg1 : tensor<?x4096xf32>, tensor<f32>) outs(%1 : tensor<?x4096xbf16>) {
  ^bb0(%in: f32, %in_0: f32, %out: bf16):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.truncf %3 : f32 to bf16
    linalg.yield %4 : bf16
  } -> tensor<?x4096xbf16>
  util.return %2 : tensor<?x4096xbf16>
}
// CHECK-LABEL: @dont_propagate_unset_encoding_outside_region
// CHECK:         iree_encoding.unset_encoding
// CHECK:         tensor.empty
// CHECK:         linalg.generic

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> ()>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]>
util.func public @dont_propagate_unset_encoding_with_multiple_uses(%arg0: tensor<?x4096xf32, #encoding>, %arg1: tensor<f32>, %arg2: index) -> (tensor<?x4096xbf16>, tensor<?x4096xbf16>) {
  %0:2 = flow.dispatch.region -> (tensor<?x4096xbf16>{%arg2}, tensor<?x4096xbf16>{%arg2}) {
    %1 = iree_encoding.unset_encoding %arg0 : tensor<?x4096xf32, #encoding> -> tensor<?x4096xf32>{%arg2}
    %2 = tensor.empty(%arg2) : tensor<?x4096xbf16>
    %3 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%1, %arg1 : tensor<?x4096xf32>, tensor<f32>) outs(%2 : tensor<?x4096xbf16>) {
    ^bb0(%in: f32, %in_0: f32, %out: bf16):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.truncf %5 : f32 to bf16
      linalg.yield %6 : bf16
    } -> tensor<?x4096xbf16>
    %4 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%1, %arg1 : tensor<?x4096xf32>, tensor<f32>) outs(%2 : tensor<?x4096xbf16>) {
    ^bb0(%in: f32, %in_0: f32, %out: bf16):
      %5 = arith.addf %in, %in_0 : f32
      %6 = arith.truncf %5 : f32 to bf16
      linalg.yield %6 : bf16
    } -> tensor<?x4096xbf16>
    flow.return %3, %4 : tensor<?x4096xbf16>, tensor<?x4096xbf16>
  }
  util.return %0#0, %0#1 : tensor<?x4096xbf16>, tensor<?x4096xbf16>
}
// CHECK-LABEL: @dont_propagate_unset_encoding_with_multiple_uses
// CHECK:         iree_encoding.unset_encoding
// CHECK:         tensor.empty
// CHECK:         linalg.generic

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]>
util.func public @dont_propagate_unset_encoding_through_generic_with_reduction(%arg0: tensor<?x4096xf32, #encoding>, %arg1: index) -> tensor<?xf32> {
  %0 = flow.dispatch.region -> (tensor<?xf32>{%arg1}) {
    %1 = iree_encoding.unset_encoding %arg0 : tensor<?x4096xf32, #encoding> -> tensor<?x4096xf32>{%arg1}
    %2 = tensor.empty(%arg1) : tensor<?xf32>
    %3 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]} ins(%1 : tensor<?x4096xf32>) outs(%2 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.addf %in, %out : f32
      linalg.yield %4 : f32
    } -> tensor<?xf32>
    flow.return %3 : tensor<?xf32>
  }
  util.return %0 : tensor<?xf32>
}
// CHECK-LABEL: @dont_propagate_unset_encoding_through_generic_with_reduction
// CHECK:         iree_encoding.unset_encoding
// CHECK:         tensor.empty
// CHECK:         linalg.generic

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]>
util.func public @dont_propagate_unset_encoding_on_init(%arg0: tensor<?x4096xf32>, %arg1: index) -> tensor<?x4096xf32> {
  %0 = flow.dispatch.region -> (tensor<?x4096xf32>{%arg1}) {
    %1 = tensor.empty(%arg1) : tensor<?x4096xf32, #encoding>
    %2 = iree_encoding.unset_encoding %1 : tensor<?x4096xf32, #encoding> -> tensor<?x4096xf32>{%arg1}
    %3 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x4096xf32>) outs(%2 : tensor<?x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.addf %in, %out : f32
      linalg.yield %4 : f32
    } -> tensor<?x4096xf32>
    flow.return %3 : tensor<?x4096xf32>
  }
  util.return %0 : tensor<?x4096xf32>
}
// CHECK-LABEL: @dont_propagate_unset_encoding_on_init
// CHECK:         tensor.empty
// CHECK:         iree_encoding.unset_encoding
// CHECK:         linalg.generic

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1, d0)>
#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2]>
util.func public @dont_propagate_non_projected_permutation(%arg0: tensor<?x4096xf32, #encoding>, %arg1: tensor<?x4x4096xf32>, %arg2: index) -> tensor<?x4096xf32> {
  %0 = flow.dispatch.region -> (tensor<?x4096xf32>{%arg2}) {
    %1 = iree_encoding.unset_encoding %arg0 : tensor<?x4096xf32, #encoding> -> tensor<?x4096xf32>{%arg2}
    %2 = tensor.empty(%arg2) : tensor<?x4096xf32>
    %3 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%1, %arg1 : tensor<?x4096xf32>, tensor<?x4x4096xf32>) outs(%2 : tensor<?x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      %5 = arith.addf %4, %out : f32
      linalg.yield %5 : f32
    } -> tensor<?x4096xf32>
    flow.return %3 : tensor<?x4096xf32>
  }
  util.return %0 : tensor<?x4096xf32>
}
// CHECK-LABEL: @dont_propagate_non_projected_permutation
// CHECK:         iree_encoding.unset_encoding
// CHECK:         tensor.empty
// CHECK:         linalg.generic
