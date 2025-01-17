// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-preprocessing-convert-conv-filter-to-channels-last{filter-layout=hwfc}))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-preprocessing-convert-conv-filter-to-channels-last{filter-layout=fhwc}))" %s | FileCheck --check-prefix=FHWC %s

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d3, d6)>
// FHWC: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
// CHECK-LABEL: @conv_fp16
// FHWC-LABEL: @conv_fp16
util.func @conv_fp16(%arg0: tensor<2x130x130x16xf16>, %arg1: tensor<3x3x16x320xf16>,
%arg2: tensor<2x128x128x320xf32>)
    -> tensor<2x128x128x320xf32> {

// CHECK:      %[[EMPTY:.*]] = tensor.empty() : tensor<3x3x320x16xf16>
// CHECK:      %[[TRANSPOSE:.*]] = linalg.transpose ins(%arg1 : tensor<3x3x16x320xf16>) outs(%[[EMPTY]] : tensor<3x3x320x16xf16>) permutation = [0, 1, 3, 2]
// CHECK:      %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#map, #[[$MAP]], #map2],
// CHECK-SAME: ins(%arg0, %[[TRANSPOSE]] : tensor<2x130x130x16xf16>, tensor<3x3x320x16xf16>)

// FHWC:      %[[EMPTY:.*]] = tensor.empty() : tensor<320x3x3x16xf16>
// FHWC:      %[[TRANSPOSE:.*]] = linalg.transpose ins(%arg1 : tensor<3x3x16x320xf16>) outs(%[[EMPTY]] : tensor<320x3x3x16xf16>) permutation = [3, 0, 1, 2]
// FHWC:      %[[GENERIC:.*]] = linalg.generic
// FHWC-SAME: indexing_maps = [#map, #[[$MAP]], #map2],
// FHWC-SAME: ins(%arg0, %[[TRANSPOSE]] : tensor<2x130x130x16xf16>, tensor<320x3x3x16xf16>)
  %conv0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<2x130x130x16xf16>, tensor<3x3x16x320xf16>)
             outs(%arg2 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
  util.return %conv0 : tensor<2x128x128x320xf32>
}

// -----

// CHECK-LABEL: @conv_dyn_input
util.func @conv_dyn_input(%arg0: tensor<?x?x?x16xf16>, %arg1: tensor<3x3x16x320xf16>,
%arg2: tensor<?x?x?x320xf32>)
    -> tensor<?x?x?x320xf32> {
  %conv0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<?x?x?x16xf16>, tensor<3x3x16x320xf16>)
             outs(%arg2 : tensor<?x?x?x320xf32>) -> tensor<?x?x?x320xf32>
  util.return %conv0 : tensor<?x?x?x320xf32>
}

// -----

// CHECK-LABEL: @conv_dyn_filter
util.func @conv_dyn_filter(%arg0: tensor<2x130x130x16xf16>, %arg1: tensor<?x?x16x320xf16>,
%arg2: tensor<2x128x128x320xf32>)
    -> tensor<2x128x128x320xf32> {
  %conv0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<2x130x130x16xf16>, tensor<?x?x16x320xf16>)
             outs(%arg2 : tensor<2x128x128x320xf32>) -> tensor<2x128x128x320xf32>
  util.return %conv0 : tensor<2x128x128x320xf32>
}

// -----

// CHECK-LABEL: @conv_i8
util.func @conv_i8(%arg0: tensor<2x130x130x16xi8>, %arg1: tensor<3x3x16x320xi8>,
%arg2: tensor<2x128x128x320xi32>)
    -> tensor<2x128x128x320xi32> {
  %conv0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
             ins(%arg0, %arg1 : tensor<2x130x130x16xi8>, tensor<3x3x16x320xi8>)
             outs(%arg2 : tensor<2x128x128x320xi32>) -> tensor<2x128x128x320xi32>
  util.return %conv0 : tensor<2x128x128x320xi32>
}
