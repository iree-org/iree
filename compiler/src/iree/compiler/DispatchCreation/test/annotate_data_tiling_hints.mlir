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
