// RUN: iree-opt-tflite %s -iree-tflite-verify-fully-converted -split-input-file -verify-diagnostics

// CHECK-LABEL: func @main
func @main(%arg0: tensor<2xf32>) -> (tensor<2xf32>) {
  // CHECK: "tosa.add"
  %0 = "tosa.add"(%arg0, %arg0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// expected-error@+4 {{'tfl.add' op : unlegalized TFLite op still exists}}
// expected-error@+4 {{'tfl.sub' op : unlegalized TFLite op still exists}}
// expected-error@below {{The following TFLite operations still remain}}
func @main(%arg0: tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32> attributes {tf.entry_function = {inputs = "input", outputs = "output"}} {
  %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<1x8x8x3xf32>
  %1 = tfl.sub %0, %arg0 {fused_activation_function = "NONE"} : tensor<1x8x8x3xf32>
  return %1 : tensor<1x8x8x3xf32>
}
