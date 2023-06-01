// RUN: iree-opt --iree-convert-mhlo-to-stablehlo %s | FileCheck %s

// CHECK-LABEL: func.func @add
// CHECK-NEXT:    stablehlo.add
// CHECK-NEXT:    chlo.broadcast_add
// CHECK-NEXT:    stablehlo.add
// CHECK-NEXT:    return
func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = mhlo.add %arg0, %arg1 : tensor<4xf32>
  %1 = chlo.broadcast_add %0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %2 = stablehlo.add %1, %arg1 : tensor<4xf32>
  return %2 : tensor<4xf32>
}
