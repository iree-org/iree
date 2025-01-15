// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-preprocessing-convert-conv-filter-to-channels-last))" %s | FileCheck %s

// CHECK-LABEL: @conv
util.func @conv(%arg0: tensor<2x130x130x16xf16>, %arg1: tensor<3x3x16x320xf16>,
%arg2: tensor<2x128x128x320xf32>)
    -> tensor<2x128x128x320xf32> {
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


