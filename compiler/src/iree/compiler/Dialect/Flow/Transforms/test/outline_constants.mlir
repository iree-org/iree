// RUN: iree-opt --split-input-file --iree-flow-outline-constants %s | FileCheck %s

// Tests that we don't outline splats (as we want them to be transients).

// CHECK-LABEL: @splatConstant
util.func @splatConstant() {
  // CHECK-DAG: = arith.constant dense<1> : tensor<512x128xi32>
  %arith_cst = arith.constant dense<1> : tensor<512x128xi32>
  // CHECK-DAG: = flow.tensor.constant dense<1> : tensor<512x128xi32>
  %flow_cst = flow.tensor.constant dense<1> : tensor<512x128xi32>
  util.return
}

// -----

// Tests that constant parameters are outlined.

// CHECK: util.global private @__parameter_scope_key_tensor_4x2xi32 {inlining_policy = #util.inline.never} = #flow.parameter.named<"scope"::"key"> : tensor<4x2xi32>
// CHECK-LABEL: @parameterConstant
util.func @parameterConstant() {
  // CHECK: = util.global.load immutable @__parameter_scope_key_tensor_4x2xi32 : tensor<4x2xi32>
  %cst = flow.tensor.constant #flow.parameter.named<"scope"::"key"> : tensor<4x2xi32>
  util.return
}

// -----

// Tests that multiple constants will be hoisted and named uniquely.

//      CHECK: util.global private @__constant_tensor_2xf32 {inlining_policy = #util.inline.never} = dense<[0.0287729427, 0.0297581609]> : tensor<2xf32>
// CHECK-NEXT: util.global private @__constant_tensor_2xf32_0 {inlining_policy = #util.inline.never} = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>
// CHECK-NEXT: util.func private @denseConstants
util.func private @denseConstants() {
  // CHECK-NEXT: = util.global.load immutable @__constant_tensor_2xf32 : tensor<2xf32>
  %cst_0 = arith.constant dense<[0.0287729427, 0.0297581609]> : tensor<2xf32>
  // CHECK-NEXT: = util.global.load immutable @__constant_tensor_2xf32_0 : tensor<2xf32>
  %cst_1 = flow.tensor.constant dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>
  util.return
}

// -----

// Tests that constants are outlined to the module scope above their use to
// preserve ordering of existing functions/globals.

// CHECK: util.func private @external_func
util.func private @external_func()
// CHECK-NEXT: util.global private @__constant_tensor_2xi32
// CHECK-NEXT: util.func private @func_0()
util.func private @func_0() {
  // CHECK-NEXT: = util.global.load immutable @__constant_tensor_2xi32
  %cst_0 = arith.constant dense<[0, 1]> : tensor<2xi32>
  util.return
}

// CHECK: util.global private @existing_global
util.global private @existing_global : tensor<4xf32>
// CHECK-NEXT: util.global private @__constant_tensor_3xi32
// CHECK-NEXT: util.func private @func_1()
util.func private @func_1() {
  // CHECK-NEXT: = util.global.load immutable @__constant_tensor_3xi32
  %cst_1 = arith.constant dense<[2, 3, 4]> : tensor<3xi32>
  util.return
}

// -----

// Tests that any hoistable attrs are propagated to the outlined globals.

util.global private @device : !hal.device

//      CHECK: util.global private @__constant_tensor_2xi32
// CHECK-SAME:   stream.affinity = #hal.device.affinity<@device, [0]>
// CHECK-NEXT: util.func private @set_affinity
util.func private @set_affinity() attributes {
  stream.affinity = #hal.device.affinity<@device, [0]>
} {
  // CHECK-NEXT: = util.global.load immutable @__constant_tensor_2xi32
  %cst = arith.constant dense<[0, 1]> : tensor<2xi32>
  util.return
}
