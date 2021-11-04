// RUN: iree-opt-tflite -allow-unregistered-dialect -split-input-file -pass-pipeline='iree-tflite-lower-global-tensors' %s | IreeFileCheck %s

// CHECK-LABEL: module {
module {
  // CHECK: util.global private mutable @__iree_flow_Variable = dense<0.000000e+00> : tensor<16x16xf32>
  func @main(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
    // CHECK-NOT: tfl.call_once
    "tfl.call_once"() {session_init_function = "NoOp"} : () -> ()
    // CHECK: %[[ADDR:.+]] = util.global.address @__iree_flow_Variable : !util.ptr<tensor<16x16xf32>>
    // CHECK: %[[LOAD:.+]] = util.global.load.indirect %[[ADDR]] : !util.ptr<tensor<16x16xf32>> -> tensor<16x16xf32>
    // CHECK: %[[ADD:.+]] = tfl.add %[[LOAD]], %arg0
    // CHECK: util.global.store.indirect %[[ADD]], %[[ADDR]]
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
    %1 = "tfl.read_variable"(%0) : (tensor<*x!tf_type.resource>) -> tensor<16x16xf32>
    %2 = tfl.add %1, %arg0 {fused_activation_function = "NONE"} : tensor<16x16xf32>
    "tfl.assign_variable"(%0, %2) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return %2 : tensor<16x16xf32>
  }
  func private @NoOp() {
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
    %1 = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    "tfl.assign_variable"(%0, %1) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return
  }
}