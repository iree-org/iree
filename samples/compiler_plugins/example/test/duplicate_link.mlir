// REQUIRES: iree_dynamic_plugins

// Naming one library twice, and by both mechanisms at once, is ordinary. It
// used to abort inside the registrar.

// RUN: iree-compile --iree-load-plugin=$IREE_EXAMPLE_DYN_PLUGIN \
// RUN:   --iree-load-plugin=$IREE_EXAMPLE_DYN_PLUGIN \
// RUN:   --iree-plugin=example_dyn --compile-to=input %s 2>&1 | FileCheck %s

// RUN: IREE_LOAD_PLUGINS=$IREE_EXAMPLE_DYN_PLUGIN \
// RUN:   iree-compile --iree-load-plugin=$IREE_EXAMPLE_DYN_PLUGIN \
// RUN:   --iree-plugin=example_dyn --compile-to=input %s 2>&1 | FileCheck %s

// CHECK: remark: This remark is from the example plugin activation
func.func @main() {
  return
}
