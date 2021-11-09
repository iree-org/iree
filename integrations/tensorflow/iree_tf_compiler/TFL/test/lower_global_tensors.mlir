// RUN: iree-opt-tflite -split-input-file -allow-unregistered-dialect -pass-pipeline='iree-tflite-lower-global-tensors' %s | IreeFileCheck %s

// CHECK-LABEL: module {
module {
  // CHECK: util.global private mutable @__iree_flow_Variable = dense<1.000000e+00> : tensor<16x16xf32>
  // CHECK: func @state
  func @state(%arg0: tensor<16x16xf32>) -> () {
    "tfl.call_once"() {session_init_function = "StateInit"} : () -> ()
    return
  }

  func private @StateInit() {
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
    %1 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    "tfl.assign_variable"(%0, %1) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: module {
module {
  // CHECK: util.global private mutable @__iree_flow_Variable = dense<1.000000e+00> : tensor<16x16xf32>

  // CHECK: func @assign
  func @assign(%arg0: tensor<16x16xf32>) -> () {
    "tfl.call_once"() {session_init_function = "AssignInit"} : () -> ()
    // CHECK: %[[ADDR:.+]] = util.global.address @__iree_flow_Variable : !util.ptr<tensor<16x16xf32>>
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>

    // CHECK: util.global.store.indirect %arg0, %[[ADDR]]
    "tfl.assign_variable"(%0, %arg0) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return
  }

  func private @AssignInit() {
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
    %1 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    "tfl.assign_variable"(%0, %1) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: module {
module {
  // CHECK: util.global private mutable @__iree_flow_Variable = dense<1.000000e+00> : tensor<16x16xf32>

  // CHECK: func @read
  func @read(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
    "tfl.call_once"() {session_init_function = "ReadInit"} : () -> ()

    // CHECK: %[[ADDR:.+]] = util.global.address @__iree_flow_Variable : !util.ptr<tensor<16x16xf32>>
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>

    // CHECK: %[[LOAD:.+]] = util.global.load.indirect %[[ADDR]] : !util.ptr<tensor<16x16xf32>> -> tensor<16x16xf32>
    %1 = "tfl.read_variable"(%0) : (tensor<*x!tf_type.resource>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func private @ReadInit() {
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
    %1 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    "tfl.assign_variable"(%0, %1) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: module {
module {
  // CHECK: util.global private mutable @__iree_flow_Variable = dense<2.000000e+00> : tensor<16x16xf32>

  // func @readAssign
  func @readAssign(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
    "tfl.call_once"() {session_init_function = "ReadAssignInit"} : () -> ()
    // CHECK: %[[ADDR:.+]] = util.global.address @__iree_flow_Variable : !util.ptr<tensor<16x16xf32>>
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>

    // CHECK: %[[LOAD:.+]] = util.global.load.indirect %[[ADDR]] : !util.ptr<tensor<16x16xf32>> -> tensor<16x16xf32>
    %1 = "tfl.read_variable"(%0) : (tensor<*x!tf_type.resource>) -> tensor<16x16xf32>

    // CHECK: %[[ADD:.+]] = tfl.add %[[LOAD]], %arg0
    %2 = tfl.add %1, %arg0 {fused_activation_function = "NONE"} : tensor<16x16xf32>

    // CHECK: util.global.store.indirect %[[ADD]], %[[ADDR]]
    "tfl.assign_variable"(%0, %2) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return %2 : tensor<16x16xf32>
  }
  func private @ReadAssignInit() {
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
    %1 = "tfl.pseudo_const"() {value = dense<2.000000e+00> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    "tfl.assign_variable"(%0, %1) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return
  }
}

// -----

module {
  // CHECK-label: @nostate
  func @nostate(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
    "tfl.call_once"() {session_init_function = "NoStateInit"} : () -> ()
    // CHECK: tfl.var_handle
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>

    // CHECK: tfl.read_variable
    %1 = "tfl.read_variable"(%0) : (tensor<*x!tf_type.resource>) -> tensor<16x16xf32>

    %2 = tfl.add %1, %arg0 {fused_activation_function = "NONE"} : tensor<16x16xf32>

    // CHECK: tfl.assign_variable
    "tfl.assign_variable"(%0, %2) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return %2 : tensor<16x16xf32>
  }
  func private @NoStateInit() {
    return
  }
}
