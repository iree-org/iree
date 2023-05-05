// RUN: iree-import-tflite --output-format=mlir-ir %S/multi_add.tflite | FileCheck %s

//      CHECK: module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<1x8x8x3xf32> {iree.identifier = "a"}, %arg1: tensor<1x8x8x3xf32> {iree.identifier = "b"}, %arg2: tensor<1x8x8x3xf32> {iree.identifier = "c"}, %arg3: tensor<1x8x8x3xf32> {iree.identifier = "d"}) -> (tensor<1x8x8x3xf32> {iree.identifier = "x"}, tensor<1x8x8x3xf32> {iree.identifier = "y"}) {
// CHECK-NEXT:     %0 = "tosa.add"(%arg1, %arg2) : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
// CHECK-NEXT:     %1 = "tosa.add"(%arg0, %0) : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
// CHECK-NEXT:     %2 = "tosa.add"(%arg3, %0) : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
// CHECK-NEXT:     return %1, %2 : tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
