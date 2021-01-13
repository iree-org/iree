// RUN: iree-tf-opt -split-input-file -verify-diagnostics -pass-pipeline='iree-tf-strip-module-metadata,func(iree-tf-strip-function-metadata)' %s | IreeFileCheck %s

// CHECK-LABEL: @tf_module
// CHECK-NOT: attributes
// CHECK-NOT: tf.versions
module @tf_module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 586 : i32}}  {
  // CHECK: func @multiply__2_2__i32__uniform
  // CHECK: iree.reflection
  // CHECK-NOT: tf._user_specified_name
  // CHECK-NOT: tf._user_specified_name
  // CHECK-NOT: tf._input_shapes
  func @multiply__2_2__i32__uniform(%arg0: tensor<2xi32> {tf._user_specified_name = "args_0"}, %arg1: tensor<2xi32> {tf._user_specified_name = "args_1"}) -> tensor<2xi32> attributes {iree.module.export, iree.reflection = {abi = "sip", abiv = 1 : i32, sip = "I12!S9!k0_0k1_1R3!_0"}, tf._input_shapes = [#tf.shape<2>, #tf.shape<2>]} {
    // CHECK-NEXT: mhlo.multiply
    %0 = mhlo.multiply %arg0, %arg1 : tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}
