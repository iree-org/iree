// REQUIRES: iree_dynamic_plugins

// The plugin is loaded by path, activated by the id it reports, and its pass
// option is an ordinary compiler flag.

// RUN: iree-compile --iree-load-plugin=$IREE_OOTEX_PLUGIN \
// RUN:   --iree-plugin=ootex --ootex-tag=from_out_of_tree \
// RUN:   --compile-to=preprocessing %s | FileCheck %s

// CHECK: util.func public @marked
// CHECK-SAME: ootex.tag = "from_out_of_tree"
// CHECK-NOT: ootex.mark
func.func @marked() {
  "ootex.mark"() : () -> ()
  return
}
