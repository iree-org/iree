// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-preprocessing-convert-conv-filter-to-channels-last{filter-layout=hwfc}))" %s | FileCheck --check-prefixes=CHECK-HWFC,CHECK-ALL %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-preprocessing-convert-conv-filter-to-channels-last))" %s | FileCheck --check-prefixes=CHECK-FHWC,CHECK-ALL %s

// CHECK-HWFC: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d3, d6)>
// CHECK-FHWC: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
// CHECK-ALL-LABEL: @conv_i8
util.func @conv_i8(%arg0: tensor<2x130x130x16xi8>, %arg1: tensor<3x3x16x320xi8>,
%arg2: tensor<2x128x128x320xi32>)
    -> tensor<2x128x128x320xi32> {

// CHECK-HWFC:      %[[EMPTY:.*]] = tensor.empty() : tensor<3x3x320x16xi8>
// CHECK-HWFC:      %[[TRANSPOSE:.*]] = linalg.transpose ins(%arg1 : tensor<3x3x16x320xi8>) outs(%[[EMPTY]] : tensor<3x3x320x16xi8>) permutation = [0, 1, 3, 2]
// CHECK-HWFC:      %[[GENERIC:.*]] = linalg.generic
// CHECK-HWFC-SAME: indexing_maps = [#map, #[[$MAP]], #map2],
// CHECK-HWFC-SAME: ins(%arg0, %[[TRANSPOSE]] : tensor<2x130x130x16xi8>, tensor<3x3x320x16xi8>)

// CHECK-FHWC:      %[[EMPTY:.*]] = tensor.empty() : tensor<320x3x3x16xi8>
// CHECK-FHWC:      %[[TRANSPOSE:.*]] = linalg.transpose ins(%arg1 : tensor<3x3x16x320xi8>) outs(%[[EMPTY]] : tensor<320x3x3x16xi8>) permutation = [3, 0, 1, 2]
// CHECK-FHWC:      %[[GENERIC:.*]] = linalg.generic
// CHECK-FHWC-SAME: indexing_maps = [#map, #[[$MAP]], #map2],
// CHECK-FHWC-SAME: ins(%arg0, %[[TRANSPOSE]] : tensor<2x130x130x16xi8>, tensor<320x3x3x16xi8>)
  %conv0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<2x130x130x16xi8>, tensor<3x3x16x320xi8>)
             outs(%arg2 : tensor<2x128x128x320xi32>) -> tensor<2x128x128x320xi32>
  util.return %conv0 : tensor<2x128x128x320xi32>
}

// -----

// CHECK-ALL-LABEL: @conv_dyn_input
util.func @conv_dyn_input(%arg0: tensor<?x?x?x16xf16>, %arg1: tensor<3x3x16x320xf16>,
%arg2: tensor<?x?x?x320xf32>)
    -> tensor<?x?x?x320xf32> {
// CHECK-ALL:       %[[GENERIC:.*]] = linalg.generic

// CHECK-HWFC-SAME: ins(%arg0, %[[TRANSPOSED:.*]]: tensor<?x?x?x16xf16>, tensor<3x3x320x16xf16>)
// CHECK-FHWC-SAME: ins(%arg0, %[[TRANSPOSED:.*]]: tensor<?x?x?x16xf16>, tensor<320x3x3x16xf16>)

// CHECK-ALL-SAME:  outs(%arg2 : tensor<?x?x?x320xf32>)
// CHECK-ALL:       util.return %[[GENERIC]] : tensor<?x?x?x320xf32>
  %conv0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<?x?x?x16xf16>, tensor<3x3x16x320xf16>)
             outs(%arg2 : tensor<?x?x?x320xf32>) -> tensor<?x?x?x320xf32>
  util.return %conv0 : tensor<?x?x?x320xf32>
}

// -----

// CHECK-ALL-LABEL: @conv_dyn_filter
util.func @conv_dyn_filter(%arg0: tensor<2x130x130x16xf16>, %arg1: tensor<?x?x16x320xf16>,
%arg2: tensor<2x128x128x320xf32>)
    -> tensor<2x128x128x320xf32> {

// CHECK-ALL-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-ALL-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-ALL-DAG:   %[[DIM0:.*]] = tensor.dim %arg1, %[[C0]]
// CHECK-ALL-DAG:   %[[DIM1:.*]] = tensor.dim %arg1, %[[C1]]
// CHECK-ALL:       %[[EMPTY:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]])

// CHECK-HWFC:      %[[TRANSPOSE:.*]] = linalg.transpose ins(%arg1 : tensor<?x?x16x320xf16>) outs(%[[EMPTY]] : tensor<?x?x320x16xf16>) permutation = [0, 1, 3, 2]
// CHECK-FHWC:      %[[TRANSPOSE:.*]] = linalg.transpose ins(%arg1 : tensor<?x?x16x320xf16>) outs(%[[EMPTY]] : tensor<320x?x?x16xf16>) permutation = [3, 0, 1, 2]

// CHECK-ALL:       linalg.generic

// CHECK-HWFC-SAME: ins(%arg0, %[[TRANSPOSE]] : tensor<2x130x130x16xf16>, tensor<?x?x320x16xf16>) outs(%arg2 : tensor<2x128x128x320xf32>)
// CHECK-FHWC-SAME: ins(%arg0, %[[TRANSPOSE]] : tensor<2x130x130x16xf16>, tensor<320x?x?x16xf16>) outs(%arg2 : tensor<2x128x128x320xf32>)

  %conv0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<2x130x130x16xf16>, tensor<?x?x16x320xf16>)
             outs(%arg2 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
  util.return %conv0 : tensor<2x128x128x320xf32>
}
