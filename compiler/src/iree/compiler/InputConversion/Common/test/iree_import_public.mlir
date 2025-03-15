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
// CHECK-LABEL: util.func public @nosideeffects_func
// CHECK: nosideeffects
func.func @nosideeffects_func() -> () attributes {nosideeffects} {
  return
}
