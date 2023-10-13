// RUN: iree-opt --split-input-file --iree-global-optimization-transformation-pipeline %s | FileCheck %s

// CHECK-LABEL: @empty
func.func @empty() {
  // CHECK-NEXT: return
  return
}

// -----

func.func @elementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  %1 = arith.subf %0, %arg0 : tensor<4xf32>
  %2 = arith.mulf %1, %arg0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: func.func @elementwiseOps(%arg0: tensor<4xf32>) -> tensor<4xf32> {
//       CHECK:   %{{.+}} = linalg.generic
//       CHECK:     %{{.+}} = arith.addf %{{.+}}, %{{.+}} : f32
//       CHECK:   %{{.+}} = linalg.generic
//       CHECK:     %{{.+}} = arith.subf %{{.+}}, %{{.+}} : f32
//       CHECK:   %{{.+}} = linalg.generic
//       CHECK:     %{{.+}} = arith.mulf %{{.+}}, %{{.+}} : f32
