// RUN: iree-link \
// RUN:     --link-module=%p/iree-link-module-a.mlir \
// RUN:     --link-module=%p/iree-link-module-b.mlir \
// RUN:     %s | FileCheck %s

// External references to module A.
util.func private @module_a.compute(%arg0: tensor<4xf32>) -> tensor<4xf32>

// External references to module B.
util.func private @module_b.helper(%arg0: i32) -> i32

// Main entry point that uses linked functions.
util.func public @main(%arg0: tensor<4xf32>, %arg1: i32) -> (tensor<4xf32>, i32) {
  // Use module A's compute function.
  %0 = util.call @module_a.compute(%arg0) : (tensor<4xf32>) -> tensor<4xf32>

  // Use module B's helper function.
  %1 = util.call @module_b.helper(%arg1) : (i32) -> i32

  util.return %0, %1 : tensor<4xf32>, i32
}

// After linking, check that:
// All operations are present (order may vary).
// CHECK-DAG: util.func public @main
// CHECK-DAG: util.func private @module_a.compute
// CHECK-DAG: util.func private @module_b.helper
// CHECK-DAG: util.func private @module_b.process
// CHECK-DAG: flow.executable private @compute_dispatch_0
// Verify the flow.executable has its nested content intact (ObjectLike trait).
// The nested func.func should NOT have been treated as an external dependency.
// CHECK-DAG: flow.executable.export public @compute_dispatch_0_elementwise_4_f32
// CHECK-DAG: func.func @compute_dispatch_0_elementwise_4_f32()

// -----

// RUN: iree-link --list-symbols %s | FileCheck %s --check-prefix=SYMBOLS

// Test --list-symbols mode (lists only public symbols without linking).
// SYMBOLS: Public symbols in
// SYMBOLS: main

// -----

// Tests linking from anonymous module to named and anonymous library modules.

// RUN: iree-link --link-module=%S/iree-link-module-a.mlir --link-module=%S/iree-link-module-b.mlir --link-module=%S/iree-link-module-c.mlir %S/iree-link-anonymous.mlir | FileCheck %s --check-prefix=ANONYMOUS

// After linking with anonymous module, check that:
// Functions from anonymous module are available without prefix.
// Functions from named modules have prefixes.
// Private function conflicts are resolved by renaming one to scale_factor_0.
// ANONYMOUS-DAG: util.func public @test_anonymous
// ANONYMOUS-DAG: util.func private @subtract
// ANONYMOUS-DAG: util.func private @double
// ANONYMOUS-DAG: util.func private @module_a.compute
// ANONYMOUS-DAG: util.func private @module_b.helper
// ANONYMOUS-DAG: util.func private @module_b.process
// ANONYMOUS-DAG: util.func private @scale_factor()
// ANONYMOUS-DAG: util.func private @scale_factor_0()
