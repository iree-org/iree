// RUN: iree-opt --split-input-file --iree-util-attribute-call-graph %s | FileCheck %s

// Tests propagation of opt-in attributes on functions to callers.

// CHECK-LABEL: util.func public @caller
util.func public @caller(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: util.call @callee
  // CHECK-SAME: {hal.abi.convention = #hal.abi.convention<coarse_fences>, nosideeffects}
  %0 = util.call @callee(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

util.func private @callee(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {
  hal.abi.convention = #hal.abi.convention<coarse_fences>,
  nosideeffects
} {
  util.return %arg0 : tensor<4xf32>
}

// -----

// Tests that existing attributes on call sites are not overwritten.

// CHECK-LABEL: util.func public @existing_attributes
util.func public @existing_attributes(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: util.call @side_effecting
  // CHECK-SAME: custom.attr = "preserved"
  // CHECK-SAME: hal.abi.convention = #hal.abi.convention<coarse_fences>
  // CHECK-NOT: nosideeffects
  %0 = util.call @side_effecting(%arg0) {custom.attr = "preserved"} : (tensor<4xf32>) -> tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

util.func private @side_effecting(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {
  hal.abi.convention = #hal.abi.convention<coarse_fences>
} {
  // Function without nosideeffects - has side effects.
  util.return %arg0 : tensor<4xf32>
}

// -----

// Tests external function declarations have their attributes propagated.

// CHECK-LABEL: util.func public @external_call
util.func public @external_call(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: util.call @external_func
  // CHECK-SAME: hal.abi.convention = #hal.abi.convention<coarse_fences>
  // CHECK-SAME: nosideeffects
  %0 = util.call @external_func(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// External declaration - no body.
util.func private @external_func(tensor<4xf32>) -> tensor<4xf32> attributes {
  hal.abi.convention = #hal.abi.convention<coarse_fences>,
  nosideeffects
}
