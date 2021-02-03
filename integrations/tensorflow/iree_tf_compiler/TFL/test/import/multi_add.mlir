// RUN: iree-import-tflite iree_tf_compiler/test/TFL/multi_add.tflite | IreeFileCheck %s

//      CHECK: module {
// CHECK-NEXT:   func @main(%arg0: tensor<1x8x8x3xf32>, %arg1: tensor<1x8x8x3xf32>, %arg2: tensor<1x8x8x3xf32>, %arg3: tensor<1x8x8x3xf32>) -> (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) attributes {tf.entry_function = {inputs = "a,b,c,d", outputs = "x,y"}} {
// CHECK-NEXT:     %0 = "tosa.add"(%arg1, %arg2) : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
// CHECK-NEXT:     %1 = "tosa.add"(%arg0, %0) : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
// CHECK-NEXT:     %2 = "tosa.add"(%arg3, %0) : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
// CHECK-NEXT:     return %1, %2 : tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
