// RUN: iree-import-tflite iree_tf_compiler/test/TFL/add.tflite | IreeFileCheck %s

//      CHECK: module {
// CHECK-NEXT:   func @main(%arg0: tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32> attributes {tf.entry_function = {inputs = "input", outputs = "output"}} {
// CHECK-NEXT:     %0 = "tosa.add"(%arg0, %arg0) : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
// CHECK-NEXT:     %1 = "tosa.add"(%0, %arg0) : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
// CHECK-NEXT:     return %1 : tensor<1x8x8x3xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
