// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-annotate-data-tiling-hints))" --split-input-file %s | FileCheck %s

util.func public @matmul(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
         ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
         outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: @matmul(
// CHECK:         linalg.matmul
// CHECK-SAME:      iree.opt.data_tiling

// -----

util.func public @matmul_with_preset_hints(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = linalg.matmul {"iree.opt.data_tiling"}
         ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
         outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul
         ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
         outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: @matmul_with_preset_hints(
// CHECK:         linalg.matmul
// CHECK-SAME:      iree.opt.data_tiling
// CHECK-NOT:       iree.opt.data_tiling

// -----

util.func public @conv_2d_nhwc_hwcf(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x?x?xf32>, %arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
         ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
         outs(%arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  util.return %0 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: @conv_2d_nhwc_hwcf(
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      iree.opt.data_tiling

// -----

// 1D convolutions are not yet supported for data tiling.
util.func public @conv_1d_ncw_fcw(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>, %arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.conv_1d_ncw_fcw
         ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
         outs(%arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  util.return %0 : tensor<?x?x?xf32>
}
// CHECK-LABEL: @conv_1d_ncw_fcw(
// CHECK:         linalg.conv_1d_ncw_fcw
// CHECK-NOT:       iree.opt.data_tiling

// -----

// 3D convolutions are not yet supported for data tiling.
util.func public @conv_3d_ndhwc_dhwcf(%arg0 : tensor<?x?x?x?x?xf32>, %arg1 : tensor<?x?x?x?x?xf32>, %arg2 : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg.conv_3d_ndhwc_dhwcf
         ins(%arg0, %arg1 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
         outs(%arg2 : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  util.return %0 : tensor<?x?x?x?x?xf32>
}
// CHECK-LABEL: @conv_3d_ndhwc_dhwcf(
// CHECK:         linalg.conv_3d_ndhwc_dhwcf
// CHECK-NOT:       iree.opt.data_tiling
