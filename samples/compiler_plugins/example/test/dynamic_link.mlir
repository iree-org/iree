// REQUIRES: iree_dynamic_plugins

// The loaded copy of the example plugin. The CHECK proves its option binds
// through the host and onActivate() runs, as static_link.mlir proves for the
// linked copy.

// Command line
// RUN: iree-compile --iree-load-plugin=%iree_example_dyn_plugin \
// RUN:   --iree-plugin=example_dyn --iree-example_dyn-flag \
// RUN:   --compile-to=input %s 2>&1 | FileCheck %s

// Environment
// RUN: env IREE_LOAD_PLUGINS=%iree_example_dyn_plugin \
// RUN:   iree-compile --iree-plugin=example_dyn --iree-example_dyn-flag \
// RUN:   --compile-to=input %s 2>&1 | FileCheck %s

// CHECK: remark: This remark is from the example plugin activation (flag=1)
func.func @main() {
  return
}
