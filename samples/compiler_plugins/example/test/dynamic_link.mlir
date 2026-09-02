// REQUIRES: iree_dynamic_plugins

// The example plugin's source again, loaded rather than linked. Reaching the
// CHECK proves the loaded copy binds its option through the host's option
// storage and that onActivate() runs, which is what static_link.mlir asserts
// for the linked copy. CMake passes the module path in IREE_EXAMPLE_DYN_PLUGIN.

// Loaded via the command line flag.
// RUN: iree-compile --iree-load-plugin=example_dyn=$IREE_EXAMPLE_DYN_PLUGIN \
// RUN:   --iree-plugin=example_dyn --iree-example_dyn-flag \
// RUN:   --compile-to=input %s 2>&1 | FileCheck %s

// Loaded via the environment variable.
// RUN: IREE_LOAD_PLUGINS=example_dyn=$IREE_EXAMPLE_DYN_PLUGIN \
// RUN:   iree-compile --iree-plugin=example_dyn --iree-example_dyn-flag \
// RUN:   --compile-to=input %s 2>&1 | FileCheck %s

// CHECK: remark: This remark is from the example plugin activation (flag=1)
func.func @main() {
  return
}
