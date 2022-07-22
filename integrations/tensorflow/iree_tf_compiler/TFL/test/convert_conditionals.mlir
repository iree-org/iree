// RUN: iree-opt-tflite --split-input-file --pass-pipeline='iree-tflite-convert-conditionals' %s | FileCheck %s

func.func @main(%arg0: tensor<i32>) -> (tensor<i32>) attributes {} {
  %0 = "tosa.const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tosa.const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %2 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: tosa.while_loop 
  %3:2 = "tfl.while_loop"(%0, %arg0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %4 = "tosa.greater"(%1, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    // CHECK-NEXT: tosa.yield
    "tfl.yield"(%4) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %5 = "tosa.add"(%arg1, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = "tosa.add"(%arg2, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // CHECK-NEXT: tosa.yield
    "tfl.yield"(%5, %6) : (tensor<i32>, tensor<i32>) -> ()
  }) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %3#1: tensor<i32>
}