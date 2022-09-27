// RUN: iree-import-tflite --output-format=mlir-ir %S/add.tflite | FileCheck %s

//      CHECK: module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<1x8x8x3xf32> {iree.identifier = "input"}) -> (tensor<1x8x8x3xf32> {iree.identifier = "output"}) {
// CHECK-NEXT:     %0 = "tosa.add"(%arg0, %arg0) : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
// CHECK-NEXT:     %1 = "tosa.add"(%0, %arg0) : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
// CHECK-NEXT:     return %1 : tensor<1x8x8x3xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
