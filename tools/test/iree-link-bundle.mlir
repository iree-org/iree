// Tests creating a bundle from multiple modules using nested module format.
// Named modules are wrapped in nested module containers.
// External declarations are preserved (not resolved in bundle mode).

// RUN: iree-link --bundle \
// RUN:     --link-module=%S/iree-link-module-a.mlir \
// RUN:     --link-module=%S/iree-link-module-b.mlir \
// RUN:     -o - | FileCheck %s

// CHECK: module {
// CHECK:   module @module_a {
// CHECK:     util.func private @module_b.process
// CHECK:     util.func public @compute
// CHECK:     flow.executable private @compute_dispatch_0
// CHECK:   }
// CHECK:   module @module_b {
// CHECK:     util.func public @helper
// CHECK:     util.func public @process
// CHECK:   }
// CHECK: }

// -----

// Tests that bytecode output works correctly (auto-detected from .mlirbc
// extension).

// RUN: iree-link --bundle \
// RUN:     --link-module=%S/iree-link-module-a.mlir \
// RUN:     --link-module=%S/iree-link-module-b.mlir \
// RUN:     -o %t.mlirbc
// RUN: iree-opt %t.mlirbc | FileCheck %s --check-prefix=BYTECODE

// BYTECODE: module {
// BYTECODE:   module @module_a {
// BYTECODE:     util.func public @compute
// BYTECODE:   }
// BYTECODE:   module @module_b {
// BYTECODE:     util.func public @helper
// BYTECODE:     util.func public @process
// BYTECODE:   }
// BYTECODE: }

// -----

// Tests that private function shadowing works correctly in bundles.
// Both anonymous modules have conflicting private @scale_factor() functions.
// mergeModuleInto renames the second module's symbols to avoid conflicts.

// RUN: iree-link --bundle \
// RUN:     --link-module=%S/iree-link-module-c.mlir \
// RUN:     --link-module=%S/iree-link-anonymous.mlir \
// RUN:     -o - | FileCheck %s --check-prefix=SHADOW

// SHADOW: module {
// SHADOW-DAG:   util.func private @scale_factor() -> i32
// SHADOW-DAG:   util.func public @subtract
// SHADOW-DAG:   util.func public @double
// SHADOW-DAG:   util.func private @scale_factor_0() -> i32
// SHADOW-DAG:   util.func public @test_anonymous
// SHADOW: }

// -----

// Tests creating a bundle with a mix of named and anonymous modules.
// Named modules (a, b) are wrapped in nested modules.
// Anonymous module (c) operations are placed at the top level.

// RUN: iree-link --bundle \
// RUN:     --link-module=%S/iree-link-module-a.mlir \
// RUN:     --link-module=%S/iree-link-module-b.mlir \
// RUN:     --link-module=%S/iree-link-module-c.mlir \
// RUN:     -o - | FileCheck %s --check-prefix=ANONYMOUS

// ANONYMOUS: module {
// ANONYMOUS:   module @module_a {
// ANONYMOUS:     util.func private @module_b.process
// ANONYMOUS:     util.func public @compute
// ANONYMOUS:   }
// ANONYMOUS:   module @module_b {
// ANONYMOUS:     util.func public @helper
// ANONYMOUS:     util.func public @process
// ANONYMOUS:   }
// ANONYMOUS:   util.func private @scale_factor
// ANONYMOUS:   util.func public @subtract
// ANONYMOUS:   util.func public @double
// ANONYMOUS: }

// -----

// Tests creating an empty bundle with no input modules.
// Should produce an empty module.

// RUN: iree-link --bundle -o - | FileCheck %s --check-prefix=EMPTY

// EMPTY: module {
// EMPTY: }

// -----

// Tests that globals with initializers are correctly handled during bundling.
// Two anonymous modules have conflicting @state globals with same initial value
// but different initializers. Both should be preserved with one renamed.

// RUN: iree-link --bundle \
// RUN:     --link-module=%S/iree-link-globals-a.mlir \
// RUN:     --link-module=%S/iree-link-globals-b.mlir \
// RUN:     -o - | FileCheck %s --check-prefix=GLOBALS

// First global, initializer, and function (keeps original names).
// GLOBALS: util.global private mutable @state = dense<0.000000e+00> : tensor<f32>
// GLOBALS-NEXT: util.initializer {
// GLOBALS-NEXT:   %[[C1:.+]] = arith.constant dense<1.000000e+00> : tensor<f32>
// GLOBALS-NEXT:   util.global.store %[[C1]], @state : tensor<f32>
// GLOBALS-NEXT:   util.return
// GLOBALS-NEXT: }
// GLOBALS-NEXT: util.func private @get_state() -> tensor<f32> {
// GLOBALS-NEXT:   %[[LOAD1:.+]] = util.global.load @state : tensor<f32>
// GLOBALS-NEXT:   util.return %[[LOAD1]] : tensor<f32>
// GLOBALS-NEXT: }

// Second global, initializer, and function (renamed to @state_0/@get_state_0).
// GLOBALS-NEXT: util.global private mutable @state_0 = dense<0.000000e+00> : tensor<f32>
// GLOBALS-NEXT: util.initializer {
// GLOBALS-NEXT:   %[[C2:.+]] = arith.constant dense<2.000000e+00> : tensor<f32>
// GLOBALS-NEXT:   util.global.store %[[C2]], @state_0 : tensor<f32>
// GLOBALS-NEXT:   util.return
// GLOBALS-NEXT: }
// GLOBALS-NEXT: util.func private @get_state_0() -> tensor<f32> {
// GLOBALS-NEXT:   %[[LOAD2:.+]] = util.global.load @state_0 : tensor<f32>
// GLOBALS-NEXT:   util.return %[[LOAD2]] : tensor<f32>
// GLOBALS-NEXT: }

// -----

// Tests three-way global conflict with different initial values.
// All three globals should be preserved with unique names.

// RUN: iree-link --bundle \
// RUN:     --link-module=%S/iree-link-globals-a.mlir \
// RUN:     --link-module=%S/iree-link-globals-b.mlir \
// RUN:     --link-module=%S/iree-link-globals-c.mlir \
// RUN:     -o - | FileCheck %s --check-prefix=THREEWAY

// First module: global, initializer, and function (keeps original names).
// THREEWAY: util.global private mutable @state = dense<0.000000e+00> : tensor<f32>
// THREEWAY-NEXT: util.initializer {
// THREEWAY-NEXT:   %[[C1:.+]] = arith.constant dense<1.000000e+00> : tensor<f32>
// THREEWAY-NEXT:   util.global.store %[[C1]], @state : tensor<f32>
// THREEWAY-NEXT:   util.return
// THREEWAY-NEXT: }
// THREEWAY-NEXT: util.func private @get_state() -> tensor<f32> {
// THREEWAY-NEXT:   %[[LOAD1:.+]] = util.global.load @state : tensor<f32>
// THREEWAY-NEXT:   util.return %[[LOAD1]] : tensor<f32>
// THREEWAY-NEXT: }

// Second module: global, initializer, and function (renamed to _0 suffix).
// THREEWAY-NEXT: util.global private mutable @state_0 = dense<0.000000e+00> : tensor<f32>
// THREEWAY-NEXT: util.initializer {
// THREEWAY-NEXT:   %[[C2:.+]] = arith.constant dense<2.000000e+00> : tensor<f32>
// THREEWAY-NEXT:   util.global.store %[[C2]], @state_0 : tensor<f32>
// THREEWAY-NEXT:   util.return
// THREEWAY-NEXT: }
// THREEWAY-NEXT: util.func private @get_state_0() -> tensor<f32> {
// THREEWAY-NEXT:   %[[LOAD2:.+]] = util.global.load @state_0 : tensor<f32>
// THREEWAY-NEXT:   util.return %[[LOAD2]] : tensor<f32>
// THREEWAY-NEXT: }

// Third module: global, initializer, and function (renamed to _1 suffix).
// THREEWAY-NEXT: util.global private mutable @state_1 = dense<5.000000e+00> : tensor<f32>
// THREEWAY-NEXT: util.initializer {
// THREEWAY-NEXT:   %[[C3:.+]] = arith.constant dense<3.000000e+00> : tensor<f32>
// THREEWAY-NEXT:   util.global.store %[[C3]], @state_1 : tensor<f32>
// THREEWAY-NEXT:   util.return
// THREEWAY-NEXT: }
// THREEWAY-NEXT: util.func private @get_state_1() -> tensor<f32> {
// THREEWAY-NEXT:   %[[LOAD3:.+]] = util.global.load @state_1 : tensor<f32>
// THREEWAY-NEXT:   util.return %[[LOAD3]] : tensor<f32>
// THREEWAY-NEXT: }
