// REQUIRES: iree_dynamic_plugins

// The same source as the statically linked example plugin, built as a loadable
// module and resolved against the compiler's renamed LLVM/MLIR ABI. Reaching
// the CHECK below proves the loaded copy binds its option through the host's
// global option storage and that onActivate() runs, exactly as static_link.mlir
// asserts for the linked-in copy. IREE_EXAMPLE_DYN_PLUGIN carries the module
// path from CMake; the test is unsupported when it is unset.

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
