// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline %s | FileCheck %s

// CHECK-LABEL: @empty
util.func public @empty() {
  // CHECK-NEXT: util.return
  util.return
}

// -----

util.func public @elementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  %1 = arith.subf %0, %arg0 : tensor<4xf32>
  %2 = arith.mulf %1, %arg0 : tensor<4xf32>
  util.return %2 : tensor<4xf32>
}

// CHECK-LABEL: util.func public @elementwiseOps(%arg0: tensor<4xf32>) -> tensor<4xf32> {
//       CHECK:   %{{.+}} = linalg.generic
//       CHECK:     %{{.+}} = arith.addf %{{.+}}, %{{.+}} : f32
//       CHECK:   %{{.+}} = linalg.generic
//       CHECK:     %{{.+}} = arith.subf %{{.+}}, %{{.+}} : f32
//       CHECK:   %{{.+}} = linalg.generic
//       CHECK:     %{{.+}} = arith.mulf %{{.+}}, %{{.+}} : f32
