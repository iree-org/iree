// RUN: iree-opt --split-input-file --iree-util-link-modules="link-module=%p/link_modules_bundle.mlir" %s | FileCheck %s

// Tests basic single function linking from a simple module.

// CHECK-LABEL: util.func private @simple.add
// CHECK: arith.addi
util.func private @simple.add(%arg0: i32, %arg1: i32) -> i32

// CHECK: util.func public @main
util.func public @main(%arg0: i32, %arg1: i32) -> i32 {
  %0 = util.call @simple.add(%arg0, %arg1) : (i32, i32) -> i32
  util.return %0 : i32
}

// -----

// Tests transitive dependency resolution: compute -> helper -> multiply_by_two.
// Dependencies are inserted immediately after the function that pulled them in.

// CHECK: util.func private @transitive.compute
// CHECK: util.func private @helper
// CHECK: util.func private @multiply_by_two
// CHECK: util.func public @main
util.func private @transitive.compute(%arg0: i32) -> i32

util.func public @main(%arg0: i32) -> i32 {
  %0 = util.call @transitive.compute(%arg0) : (i32) -> i32
  util.return %0 : i32
}

// -----

// Tests selective linking: only referenced symbols should be imported.
// func3 is not referenced and should NOT be linked.

// CHECK-LABEL: util.func private @multi.func1
util.func private @multi.func1(%arg0: i32) -> i32
// CHECK: util.func private @multi.func2
util.func private @multi.func2(%arg0: i32) -> i32
// CHECK-NOT: util.func private @multi.func3

util.func public @main(%arg0: i32) -> i32 {
  %0 = util.call @multi.func1(%arg0) : (i32) -> i32
  %1 = util.call @multi.func2(%0) : (i32) -> i32
  util.return %1 : i32
}

// -----

// Tests control flow (SCF) operations with external calls inside loops.

// CHECK: util.func private @with_scf.loop_compute
// CHECK: util.func private @process_element
// CHECK: util.func public @main
util.func private @with_scf.loop_compute(%arg0: i32, %arg1: index) -> i32

util.func public @main(%arg0: i32, %arg1: index) -> i32 {
  %0 = util.call @with_scf.loop_compute(%arg0, %arg1) : (i32, index) -> i32
  util.return %0 : i32
}

// -----

// Tests module name scoping: module_a and module_b export with dotted names.
// Internal helpers should also be linked.

// CHECK: util.func private @module_a.compute
util.func private @module_a.compute(%arg0: tensor<4xf32>) -> tensor<4xf32>
// CHECK: util.func private @internal_helper
// CHECK: util.func private @module_b.process
util.func private @module_b.process(%arg0: f32) -> f32
// CHECK: util.func public @main
// CHECK-NOT: util.func private @compute

util.func public @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = util.call @module_a.compute(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %c1 = arith.constant 1.0 : f32
  %1 = util.call @module_b.process(%c1) : (f32) -> f32
  util.return %0 : tensor<4xf32>
}

// -----

// Tests that linked symbols preserve the target's visibility.
// Target has private declaration, result should be private.

// CHECK: util.func private @simple.add
util.func private @simple.add(%arg0: i32, %arg1: i32) -> i32

util.func public @main(%arg0: i32, %arg1: i32) -> i32 {
  %0 = util.call @simple.add(%arg0, %arg1) : (i32, i32) -> i32
  util.return %0 : i32
}

// -----

// Tests that isolated-from-above regions are not traversed for symbol lookup.
// The builtin.module inside flow.executable contains a func.func, but we should
// not try to link it or scan it for dependencies (it's in an isolated region).

util.global private @config : i32

// CHECK: flow.executable private @executable
flow.executable private @executable {
  flow.executable.export public @dispatch workgroups(%arg0: index) -> (index, index, index) {
    flow.return %arg0, %arg0, %arg0 : index, index, index
  }
  builtin.module {
    // This func.func is inside an isolated region and should be ignored.
    func.func @dispatch() {
      return
    }
  }
}

// CHECK: util.func public @main
util.func public @main() {
  util.return
}

// -----

// Tests private function shadowing: module_a and module_b both have private
// scale_factor() functions with different implementations (returns 100 vs 200).
// When both modules are linked, one should be renamed to scale_factor_0.
// Each module's functions should call their own version of scale_factor.

// Verify module_a's compute_scaled calls scale_factor (returns 100) and multiplies.
// CHECK: util.func private @module_a.compute_scaled(%[[ARG0:.+]]: i32)
// CHECK:   %[[SCALE0:.+]] = util.call @scale_factor() : () -> i32
// CHECK:   %[[RESULT0:.+]] = arith.muli %[[ARG0]], %[[SCALE0]] : i32
// CHECK: util.func private @scale_factor() -> i32
// CHECK-NEXT: %{{.+}} = arith.constant 100 : i32
util.func private @module_a.compute_scaled(%arg0: i32) -> i32

// Verify module_b's process_scaled calls scale_factor_0 (returns 200) and adds.
// CHECK: util.func private @module_b.process_scaled(%[[ARG1:.+]]: i32)
// CHECK:   %[[SCALE1:.+]] = util.call @scale_factor_0() : () -> i32
// CHECK:   %[[RESULT1:.+]] = arith.addi %[[ARG1]], %[[SCALE1]] : i32
// CHECK: util.func private @scale_factor_0() -> i32
// CHECK-NEXT: %{{.+}} = arith.constant 200 : i32
util.func private @module_b.process_scaled(%arg0: i32) -> i32

// CHECK: util.func public @main
util.func public @main(%arg0: i32) -> (i32, i32) {
  %0 = util.call @module_a.compute_scaled(%arg0) : (i32) -> i32
  %1 = util.call @module_b.process_scaled(%arg0) : (i32) -> i32
  util.return %0, %1 : i32, i32
}

// -----

// Tests private function shadowing with same-module reference: when linking
// module_a.compute which calls internal_helper, and then separately linking
// module_a.compute_scaled which also calls scale_factor, both should work
// without conflicts because they're from the same source module.

// CHECK: util.func private @module_a.compute
util.func private @module_a.compute(%arg0: tensor<4xf32>) -> tensor<4xf32>
// CHECK: util.func private @internal_helper
// CHECK: util.func private @module_a.compute_scaled
util.func private @module_a.compute_scaled(%arg0: i32) -> i32
// CHECK: util.func private @scale_factor
// CHECK: util.func public @main
// Only one scale_factor from module_a should exist (no shadowing within same module).
// CHECK-NOT: scale_factor_0

util.func public @main(%arg0: tensor<4xf32>, %arg1: i32) -> (tensor<4xf32>, i32) {
  %0 = util.call @module_a.compute(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = util.call @module_a.compute_scaled(%arg1) : (i32) -> i32
  util.return %0, %1 : tensor<4xf32>, i32
}

// -----

// Tests three-way private function shadowing: module_a, module_b, and module_c
// all have private scale_factor() functions with different implementations
// (returns 100, 200, 300 respectively).
// When all three modules are linked, two should be renamed to scale_factor_0
// and scale_factor_1, ensuring unique name generation works correctly.

// Verify module_a's compute_scaled calls scale_factor (returns 100).
// CHECK: util.func private @module_a.compute_scaled
// CHECK:   %[[SCALE_A:.+]] = util.call @scale_factor() : () -> i32
// CHECK: util.func private @scale_factor() -> i32
// CHECK-NEXT: %{{.+}} = arith.constant 100 : i32
util.func private @module_a.compute_scaled(%arg0: i32) -> i32

// Verify module_b's process_scaled calls scale_factor_0 (returns 200).
// CHECK: util.func private @module_b.process_scaled
// CHECK:   %[[SCALE_B:.+]] = util.call @scale_factor_0() : () -> i32
// CHECK: util.func private @scale_factor_0() -> i32
// CHECK-NEXT: %{{.+}} = arith.constant 200 : i32
util.func private @module_b.process_scaled(%arg0: i32) -> i32

// Verify module_c's divide_scaled calls scale_factor_1 (returns 300).
// CHECK: util.func private @module_c.divide_scaled
// CHECK:   %[[SCALE_C:.+]] = util.call @scale_factor_1() : () -> i32
// CHECK: util.func private @scale_factor_1() -> i32
// CHECK-NEXT: %{{.+}} = arith.constant 300 : i32
util.func private @module_c.divide_scaled(%arg0: i32, %arg1: i32) -> i32

// CHECK: util.func public @main
util.func public @main(%arg0: i32, %arg1: i32) -> (i32, i32, i32) {
  %0 = util.call @module_a.compute_scaled(%arg0) : (i32) -> i32
  %1 = util.call @module_b.process_scaled(%arg0) : (i32) -> i32
  %2 = util.call @module_c.divide_scaled(%arg0, %arg1) : (i32, i32) -> i32
  util.return %0, %1, %2 : i32, i32, i32
}

// -----

// Tests target module conflict: the target module has a private definition
// of helper() that returns 42, and we're linking simple.add which should
// work fine (no conflict). This verifies that non-conflicting symbols
// from library modules can coexist with target symbols.

// CHECK: util.func private @helper() -> i32
// CHECK-NEXT: %[[C42:.+]] = arith.constant 42 : i32
// CHECK-NEXT: util.return %[[C42]]
util.func private @helper() -> i32 {
  %c42 = arith.constant 42 : i32
  util.return %c42 : i32
}

// CHECK: util.func private @simple.add
util.func private @simple.add(%arg0: i32, %arg1: i32) -> i32

// CHECK: util.func public @main
util.func public @main(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %0 = util.call @helper() : () -> i32
  %1 = util.call @simple.add(%arg0, %arg1) : (i32, i32) -> i32
  util.return %0, %1 : i32, i32
}

// -----

// Tests target module conflict with incoming symbol: the target module has
// a private definition of scale_factor() that returns 999, and we're linking
// module_a.compute_scaled which requires scale_factor() that returns 100.
// The incoming scale_factor should be renamed to avoid overwriting the
// target's definition.

// Target's scale_factor (should remain unchanged).
// CHECK: util.func private @scale_factor() -> i32
// CHECK-NEXT: %[[C999:.+]] = arith.constant 999 : i32
// CHECK-NEXT: util.return %[[C999]]
util.func private @scale_factor() -> i32 {
  %c999 = arith.constant 999 : i32
  util.return %c999 : i32
}

// Incoming module_a.compute_scaled should call scale_factor_0 (renamed).
// CHECK: util.func private @module_a.compute_scaled
// CHECK:   %[[SCALE_RENAMED:.+]] = util.call @scale_factor_0() : () -> i32
util.func private @module_a.compute_scaled(%arg0: i32) -> i32

// Renamed incoming scale_factor (returns 100).
// CHECK: util.func private @scale_factor_0() -> i32
// CHECK-NEXT: %[[C100:.+]] = arith.constant 100 : i32

// CHECK: util.func public @main
util.func public @main(%arg0: i32) -> (i32, i32) {
  // Call target's scale_factor (returns 999).
  %0 = util.call @scale_factor() : () -> i32
  // Call linked function which uses renamed scale_factor_0 (returns 100).
  %1 = util.call @module_a.compute_scaled(%arg0) : (i32) -> i32
  util.return %0, %1 : i32, i32
}
