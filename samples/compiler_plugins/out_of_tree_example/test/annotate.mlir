// REQUIRES: iree_dynamic_plugins

// RUN: iree-compile --iree-load-plugin=%iree_ootex_plugin \
// RUN:   --iree-plugin=ootex --ootex-tag=from_out_of_tree \
// RUN:   --compile-to=preprocessing %s | FileCheck %s

// The ABI pass has moved the marked body into a private function.

// CHECK: util.func private @_marked
// CHECK-SAME: ootex.tag = "from_out_of_tree"
// CHECK-NOT: ootex.mark
func.func @marked() {
  "ootex.mark"() : () -> ()
  return
}
