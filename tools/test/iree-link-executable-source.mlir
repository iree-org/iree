// Tests linking modules containing hal.executable.source ops.
// hal.executable.source is used for hand-authored extern dispatch kernels
// (e.g., pre-compiled GPU kernel objects). The linker must copy these
// alongside the functions that reference them via flow.dispatch.

// RUN: iree-link \
// RUN:     --link-module=%p/iree-link-executable-source-module-d.mlir \
// RUN:     %s | FileCheck %s

// External reference to module D's transform function.
util.func private @module_d.transform(%arg0: tensor<4xf32>) -> tensor<4xf32>

// Main entry point that calls the linked function.
util.func public @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = util.call @module_d.transform(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  util.return %0 : tensor<4xf32>
}

// After linking:
// 1. The function body is filled in (not just a declaration).
// CHECK-DAG: util.func public @main
// CHECK-DAG: util.func private @module_d.transform

// 2. The flow.dispatch reference in the linked function body is intact.
// CHECK-DAG: flow.dispatch @extern_kernel::@entry

// 3. The hal.executable.source is copied with its body intact.
// CHECK-DAG: hal.executable.source private @extern_kernel
// CHECK-DAG: hal.executable.export public @entry ordinal(0)
// CHECK-DAG: workgroup_size = [64 : index, 1 : index, 1 : index]

// -----

// Test that hal.executable.source internal symbols are NOT treated as
// external dependencies. The nested hal.executable.export should not
// appear as a linkable symbol.

// RUN: iree-link --list-symbols \
// RUN:     %p/iree-link-executable-source-module-d.mlir \
// RUN:     | FileCheck %s --check-prefix=SYMBOLS

// Only the public function should be listed, not the executable or its exports.
// SYMBOLS: Public symbols in
// SYMBOLS: @transform
// SYMBOLS-NOT: @extern_kernel
// SYMBOLS-NOT: @entry
