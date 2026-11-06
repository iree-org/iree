// RUN: iree-opt --split-input-file --iree-import-public %s | FileCheck %s

// CHECK-LABEL: util.func private @private_func
// CHECK: util.return
func.func private @private_func() -> () {
  return
}

// -----
// CHECK-LABEL: util.func public @noinline_func
// CHECK: inlining_policy = #util.inline.never
func.func @noinline_func() -> () attributes {noinline} {
  return
}

// -----
// The func dialect property must survive even when it is not printed as the
// legacy spelling.
// CHECK-LABEL: util.func public @no_inline_property
// CHECK-SAME: inlining_policy = #util.inline.never
func.func @no_inline_property() -> () attributes {no_inline} {
  return
}

// -----
// CHECK-LABEL: util.func public @nosideeffects_func
// CHECK: nosideeffects
func.func @nosideeffects_func() -> () attributes {nosideeffects} {
  return
}

// -----
// An omitted call tie mapping inherits the callee ABI when it has ties.
// CHECK-LABEL: util.func private @tied_callee
// CHECK-SAME: {iree.abi.tied = 0 : i64}
// CHECK-LABEL: util.func public @tied_caller
// CHECK: util.call @tied_callee{{.*}}-> %arg0
func.func private @tied_callee(%arg: tensor<4xi32>)
    -> (tensor<4xi32> {iree.abi.tied = 0 : i64}) {
  return %arg : tensor<4xi32>
}
func.func @tied_caller(%arg: tensor<4xi32>) -> tensor<4xi32> {
  %result = func.call @tied_callee(%arg) : (tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----
// An omitted mapping remains absent for a callee with no tied results.
// CHECK-LABEL: util.func private @untied_callee
// CHECK-NOT: tied_operands
// CHECK-LABEL: util.func public @untied_caller
// CHECK: util.call @untied_callee
// CHECK-NOT: tied_operands
func.func private @untied_callee(%arg: tensor<4xi32>) -> tensor<4xi32> {
  return %arg : tensor<4xi32>
}
func.func @untied_caller(%arg: tensor<4xi32>) -> tensor<4xi32> {
  %result = func.call @untied_callee(%arg) : (tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----
// A caller appearing before its callee exercises lookup after callee rewrite.
// CHECK-LABEL: util.func public @already_util_caller
// CHECK: util.call @already_util_callee{{.*}}-> %arg0
func.func @already_util_caller(%arg: tensor<4xi32>) -> tensor<4xi32> {
  %result = func.call @already_util_callee(%arg) : (tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}
func.func private @already_util_callee(%arg: tensor<4xi32>)
    -> (tensor<4xi32> {iree.abi.tied = 0 : i64}) {
  return %arg : tensor<4xi32>
}
