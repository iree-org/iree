// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-fuse-encoding-ops-into-dispatch-regions-pass))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#encoding = #iree_encoding.testing<>
util.func public @parallel_fusion(%arg0: tensor<2x11008x128xf32>) -> tensor<2x11008x128xf32, #encoding> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x11008x128xf32>
  %1 = flow.dispatch.region -> (tensor<2x11008x128xf32>) {
    %3 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg0 : tensor<2x11008x128xf32>, tensor<2x11008x128xf32>)
        outs(%0 : tensor<2x11008x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<2x11008x128xf32>
    flow.return %3 : tensor<2x11008x128xf32>
  }
  %2 = iree_encoding.set_encoding %1 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
  util.return %2 : tensor<2x11008x128xf32, #encoding>
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing<>
// CHECK-LABEL: @parallel_fusion
// CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region -> (tensor<2x11008x128xf32, #[[$ENCODING]]>)
// CHECK:         %[[ADD:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH0]] : tensor<2x11008x128xf32, #[[$ENCODING]]>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.testing<>
util.func public @reduction_fusion(%arg0: tensor<2x11008x128x16xf32>) -> tensor<2x11008x128xf32, #encoding> {
  %0 = tensor.empty() : tensor<2x11008x128xf32>
  %1 = flow.dispatch.region -> (tensor<2x11008x128xf32>) {
    %5 = linalg.generic {
        indexing_maps = [#map, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%arg0 : tensor<2x11008x128x16xf32>)
        outs(%0 : tensor<2x11008x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.addf %in, %out : f32
      linalg.yield %6 : f32
    } -> tensor<2x11008x128xf32>
    flow.return %5 : tensor<2x11008x128xf32>
  }
  %2 = iree_encoding.set_encoding %1 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
  util.return %2 : tensor<2x11008x128xf32, #encoding>
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing<>
// CHECK-LABEL: @reduction_fusion
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<2x11008x128xf32, #[[$ENCODING]]>)
// CHECK:         %[[REDUCTION:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[REDUCTION]]
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]] : tensor<2x11008x128xf32, #[[$ENCODING]]>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding = #iree_encoding.matmul_k<k_dims = [2]>
util.func public @matmul_k_reduction_fusion(%arg0: tensor<2x11008x128x16xf32>) -> tensor<2x11008x128xf32, #encoding> {
  %0 = tensor.empty() : tensor<2x11008x128xf32>
  %1 = flow.dispatch.region -> (tensor<2x11008x128xf32>) {
    %5 = linalg.generic {
        indexing_maps = [#map, #map1],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%arg0 : tensor<2x11008x128x16xf32>)
        outs(%0 : tensor<2x11008x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.addf %in, %out : f32
      linalg.yield %6 : f32
    } -> tensor<2x11008x128xf32>
    flow.return %5 : tensor<2x11008x128xf32>
  }
  %2 = iree_encoding.set_encoding %1 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
  util.return %2 : tensor<2x11008x128xf32, #encoding>
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.matmul_k<k_dims = [2]>
// CHECK-LABEL: @matmul_k_reduction_fusion
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region
// CHECK:         %[[REDUCTION:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[REDUCTION]]
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]] : tensor<2x11008x128xf32, #[[$ENCODING]]>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#encoding = #iree_encoding.testing<>
util.func public @transpose_fusion(%arg0: tensor<2x128x11008xf32>) -> tensor<2x11008x128xf32, #encoding> {
  %0 = tensor.empty() : tensor<2x11008x128xf32>
  %1 = flow.dispatch.region -> (tensor<2x11008x128xf32>) {
    %5 = linalg.generic {
        indexing_maps = [#map, #map1],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<2x128x11008xf32>)
        outs(%0 : tensor<2x11008x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x11008x128xf32>
    flow.return %5 : tensor<2x11008x128xf32>
  }
  %2 = iree_encoding.set_encoding %1 : tensor<2x11008x128xf32> -> tensor<2x11008x128xf32, #encoding>
  util.return %2 : tensor<2x11008x128xf32, #encoding>
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing<>
// CHECK-LABEL: @transpose_fusion
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<2x11008x128xf32, #[[$ENCODING]]>
// CHECK:         %[[TRANSPOSE:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[TRANSPOSE]]
// CHECK:         flow.return %[[SET_ENCODING]]
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]] : tensor<2x11008x128xf32, #[[$ENCODING]]>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#encoding = #iree_encoding.testing<>
util.func public @fusion_dynamic(%arg0: tensor<?x?x?xf32>, %d0: index, %d1: index, %d2: index) -> tensor<?x?x?xf32, #encoding> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %1 = flow.dispatch.region -> (tensor<?x?x?xf32>{%d0, %d1, %d2}) {
    %3 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg0 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%0 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<?x?x?xf32>
    flow.return %3 : tensor<?x?x?xf32>
  }
  %2 = iree_encoding.set_encoding %1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding>
  util.return %2 : tensor<?x?x?xf32, #encoding>
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing<>
// CHECK-LABEL: @fusion_dynamic
// CHECK-SAME:    {{.+}}: tensor<?x?x?xf32>, %[[D0:.+]]: index, %[[D1:.+]]: index, %[[D2:.+]]: index)
// CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region -> (tensor<?x?x?xf32, #[[$ENCODING]]>
// CHECK-SAME:      {%[[D0]], %[[D1]], %[[D2]]}
// CHECK:         %[[ADD:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH0]] : tensor<?x?x?xf32, #[[$ENCODING]]>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#encoding = #iree_encoding.testing<>
util.func public @multi_encoding_fusion_dynamic(%arg0: tensor<?x?x?xf32>, %d0: index, %d1: index, %d2: index) -> (tensor<?x?x?xf32, #encoding>, tensor<?x?x?xf32, #encoding>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %1 = flow.dispatch.region -> (tensor<?x?x?xf32>{%d0, %d1, %d2}) {
    %3 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0, %arg0 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%0 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<?x?x?xf32>
    flow.return %3 : tensor<?x?x?xf32>
  }
  %2 = iree_encoding.set_encoding %1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding>
  %3 = iree_encoding.set_encoding %1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding>
  util.return %2, %3 : tensor<?x?x?xf32, #encoding>, tensor<?x?x?xf32, #encoding>
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing<>
// CHECK-LABEL: @multi_encoding_fusion_dynamic
// CHECK-SAME:    {{.+}}: tensor<?x?x?xf32>, %[[D0:.+]]: index, %[[D1:.+]]: index, %[[D2:.+]]: index)
// CHECK:       %[[DISPATCH:.+]] = flow.dispatch.region -> (tensor<?x?x?xf32, #[[$ENCODING]]>
// CHECK-SAME:      {%[[D0]], %[[D1]], %[[D2]]}
// CHECK:         %[[ADD:.+]] = linalg.generic
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH]], %[[DISPATCH]]

// -----

#encoding = #iree_encoding.testing<>
util.func public @reshape_fusion(%arg0: tensor<32x32xf32>) -> tensor<16x64xf32, #encoding> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<32x32xf32>
  %1 = flow.dispatch.region -> (tensor<32x32xf32>) {
    %3 = linalg.add ins(%arg0, %arg0 : tensor<32x32xf32>, tensor<32x32xf32>)
        outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    flow.return %3 : tensor<32x32xf32>
  }
  %collapsed = tensor.collapse_shape %1 [[0, 1]] : tensor<32x32xf32> into tensor<1024xf32>
  %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [16, 64] : tensor<1024xf32> into tensor<16x64xf32>
  %2 = iree_encoding.set_encoding %expanded : tensor<16x64xf32> -> tensor<16x64xf32, #encoding>
  util.return %2 : tensor<16x64xf32, #encoding>
}
// CHECK:       #[[$ENCODING:.+]] = #iree_encoding.testing<>
// CHECK-LABEL: @reshape_fusion
// CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region -> (tensor<16x64xf32, #[[$ENCODING]]>)
// CHECK:         linalg.add
// CHECK:         tensor.collapse_shape
// CHECK:         tensor.expand_shape
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH0]] : tensor<16x64xf32, #[[$ENCODING]]>

// -----

// Tests that dependencies of the chain of ops before set_encoding are moved before the
// the dispatch region. In this case this would be the `arith.divsi` and `arith.constant`
// that the `tensor.expand_shape` depends on.

#map = affine_map<(d0) -> (d0)>
#encoding = #iree_encoding.testing<>
util.func public @move_dependencies_before_dispatch(%arg0: tensor<?xf32>, %arg1: index) -> tensor<?x1024xf32, #encoding> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty(%arg1) : tensor<?xf32>
  %1 = flow.dispatch.region -> (tensor<?xf32>{%arg1}) {
    %3 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]}
        ins(%arg0, %arg0 : tensor<?xf32>, tensor<?xf32>)
        outs(%0 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<?xf32>
    flow.return %3 : tensor<?xf32>
  }
  %c1024 = arith.constant 1024 : index
  %2 = arith.divsi %arg1, %c1024 : index
  %3 = tensor.expand_shape %1 [[0, 1]] output_shape [%2, 1024] : tensor<?xf32> into tensor<?x1024xf32>
  %4 = iree_encoding.set_encoding %3 : tensor<?x1024xf32> -> tensor<?x1024xf32, #encoding>
  util.return %4 : tensor<?x1024xf32, #encoding>
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing<>
// CHECK-LABEL: @move_dependencies_before_dispatch
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK:         %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:         %[[DIV:.+]] = arith.divsi %[[ARG1]], %[[C1024]] : index
// CHECK:         %[[DISPATCH0:.+]] = flow.dispatch.region -> (tensor<?x1024xf32, #[[$ENCODING]]>{%[[DIV]]})
// CHECK:           %[[ADD:.+]] = linalg.generic
// CHECK:           %[[EXPAND:.+]] = tensor.expand_shape %[[ADD]]
// CHECK-SAME:        output_shape [%[[DIV]], 1024]
// CHECK:           %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[EXPAND]]
// CHECK:           flow.return %[[SET_ENCODING]] : tensor<?x1024xf32, #[[$ENCODING]]>
// CHECK:         }
// CHECK:         util.return %[[DISPATCH0]] : tensor<?x1024xf32, #[[$ENCODING]]>

// -----

#encoding0 = #iree_encoding.testing<[#iree_encoding.specialized<0>]>
#encoding1 = #iree_encoding.testing<[#iree_encoding.specialized<1>]>
util.func public @encoding_fusion(%arg0: tensor<128xf32, #encoding0>) -> tensor<128xf32, #encoding1> {
  %1 = flow.dispatch.region -> (tensor<128xf32>) {
    %3 = iree_encoding.unset_encoding %arg0 : tensor<128xf32, #encoding0> -> tensor<128xf32>
    flow.return %3 : tensor<128xf32>
  }
  %2 = iree_encoding.set_encoding %1 : tensor<128xf32> -> tensor<128xf32, #encoding1>
  util.return %2 : tensor<128xf32, #encoding1>
}
// CHECK-LABEL: @encoding_fusion
// CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region
// CHECK:         iree_encoding.unset_encoding
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH0]]

// -----

#encoding = #iree_encoding.testing<>
util.func public @attention_fusion(
    %query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>,
    %value: tensor<192x1024x64xf32>, %scale: f32) -> tensor<192x1024x64xf32, #encoding> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %1 = flow.dispatch.region -> (tensor<192x1024x64xf32>) {
    %3 = iree_linalg_ext.attention {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> ()>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
        ins(%query, %key, %value, %scale : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, f32)
        outs(%0 : tensor<192x1024x64xf32>) {
           ^bb0(%arg0: f32):
           iree_linalg_ext.yield %arg0 : f32
        } -> tensor<192x1024x64xf32>
    flow.return %3 : tensor<192x1024x64xf32>
  }
  %2 = iree_encoding.set_encoding %1 : tensor<192x1024x64xf32> -> tensor<192x1024x64xf32, #encoding>
  util.return %2 : tensor<192x1024x64xf32, #encoding>
}
// CHECK-LABEL: @attention_fusion
// CHECK:       %[[DISPATCH0:.+]] = flow.dispatch.region
// CHECK:         iree_linalg_ext.attention
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding
// CHECK:         flow.return %[[SET_ENCODING]] :
// CHECK:       }
// CHECK:       util.return %[[DISPATCH0]]
