// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-NOT: util.initializer
util.initializer {
  util.return
}

// -----

// Tests that util.scf.unreachable is converted to util.unreachable outside SCF.

// CHECK-LABEL: util.func public @scf_unreachable_to_terminator
util.func public @scf_unreachable_to_terminator() {
  // CHECK-NOT: util.scf.unreachable
  // CHECK: util.unreachable "not in scf"
  // CHECK-NOT: util.return
  util.scf.unreachable "not in scf"
  util.return
}

// -----

// Tests that ops after util.scf.unreachable are deleted when it is converted to
// a terminator.

// CHECK-LABEL: util.func public @delete_after_scf_unreachable
util.func public @delete_after_scf_unreachable() -> i32 {
  // CHECK-NOT: util.scf.unreachable
  // CHECK: util.unreachable "early exit"
  // CHECK-NOT: arith.constant
  // CHECK-NOT: util.return
  util.scf.unreachable "early exit"
  %c42 = arith.constant 42 : i32
  %c43 = arith.constant 43 : i32
  %sum = arith.addi %c42, %c43 : i32
  util.return %sum : i32
}

// -----

// Tests that even ops with side effects after util.scf.unreachable are deleted
// when it's converted to a terminator (outside SCF regions). This is a
// combination of turning util.scf.unreachable into util.unreachable and the
// util.unreachable patterns.

util.global private mutable @test_global : i32
util.func private @dropped_side_effect_func()

// CHECK-LABEL: util.func public @delete_side_effects_after_scf_unreachable_outside_scf
util.func public @delete_side_effects_after_scf_unreachable_outside_scf() {
  // CHECK-NOT: util.scf.unreachable
  // CHECK: util.unreachable "early exit"
  // CHECK-NOT: util.global.store
  // CHECK-NOT: util.call
  // CHECK-NOT: util.return
  util.scf.unreachable "early exit"
  %c42 = arith.constant 42 : i32
  util.global.store %c42, @test_global : i32
  util.call @dropped_side_effect_func() : () -> ()
  util.return
}

// -----

// Tests that util.scf.unreachable inside SCF regions is preserved and
// subsequent operations are erased and replaced with poison.

util.func private @preserved_side_effect_func()

// CHECK-LABEL: util.func public @scf_unreachable_in_scf
util.func public @scf_unreachable_in_scf(%cond: i1, %input: i32) -> i32 {
  // CHECK: %[[POISON:.+]] = ub.poison : i32
  // CHECK: scf.if
  %result = scf.if %cond -> i32 {
    // CHECK: util.call @preserved_side_effect_func
    // CHECK: util.scf.unreachable "in scf region"
    // CHECK-NOT: arith.constant 100
    // CHECK: scf.yield %[[POISON]]
    // Side effect to prevent elimination:
    util.call @preserved_side_effect_func() : () -> ()
    util.scf.unreachable "in scf region"
    %c100 = arith.constant 100 : i32
    %sum = arith.addi %input, %c100 : i32
    scf.yield %sum : i32
  } else {
    // Use input to create dependency
    %c42 = arith.constant 42 : i32
    %val = arith.addi %input, %c42 : i32
    scf.yield %val : i32
  }
  util.return %result : i32
}

// -----

// Tests that even ops with side effects after util.scf.unreachable are deleted
// inside SCF regions.

util.global private mutable @test_global : i32
util.func private @preserved_side_effect_func()
util.func private @dropped_side_effect_func()

// CHECK-LABEL: util.func public @delete_side_effects_after_scf_unreachable_in_scf
util.func public @delete_side_effects_after_scf_unreachable_in_scf(%cond: i1, %input: i32) -> i32 {
  // CHECK: %[[POISON:.+]] = ub.poison : i32
  // CHECK: scf.if
  %result = scf.if %cond -> i32 {
    // CHECK: util.call @preserved_side_effect_func
    // CHECK: util.scf.unreachable "early exit"
    // CHECK-NOT: arith.constant 42
    // CHECK-NOT: util.global.store
    // CHECK-NOT: util.call @dropped_side_effect_func
    // CHECK: scf.yield %[[POISON]]
    util.call @preserved_side_effect_func() : () -> ()  // Side effect to prevent elimination
    util.scf.unreachable "early exit"
    %c42 = arith.constant 42 : i32
    %result_val = arith.addi %input, %c42 : i32
    util.global.store %result_val, @test_global : i32
    util.call @dropped_side_effect_func() : () -> ()
    scf.yield %result_val : i32
  } else {
    %c100 = arith.constant 100 : i32
    %val = arith.addi %input, %c100 : i32
    scf.yield %val : i32
  }
  util.return %result : i32
}

// -----

// Tests that values escaping from unreachable blocks are replaced with poison
// when util.scf.unreachable is converted to util.unreachable. This test ensures
// the pattern handles cases where values defined after util.scf.unreachable are
// used in other blocks/regions.

util.func private @side_effect()

// CHECK-LABEL: util.func public @escaping_values_from_unreachable_block
util.func public @escaping_values_from_unreachable_block(%cond: i1, %cond2: i1, %input: i32) -> i32 {
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // When bb1 has util.scf.unreachable followed by ops that produce values
  // used in other blocks, those values should be replaced with poison.
  // The side effect keeps the block from being eliminated entirely.
  util.call @side_effect() : () -> ()  // Keep block alive so pattern runs.
  util.scf.unreachable "unreachable path"
  %c42 = arith.constant 42 : i32
  %c43 = arith.constant 43 : i32
  %val1 = arith.addi %c42, %c43 : i32
  %val2 = arith.addi %val1, %input : i32
  // These values escape to bb3 but should be replaced with poison.
  cf.cond_br %cond2, ^bb3(%val1 : i32), ^bb3(%val2 : i32)

^bb2:
  %c100 = arith.constant 100 : i32
  cf.br ^bb3(%c100 : i32)

^bb3(%arg: i32):
  // CHECK: util.return
  util.return %arg : i32
}

// -----

// Tests that util.scf.unreachable followed by operations with escaping values
// works correctly when converted to util.unreachable without side effects.

// CHECK-LABEL: util.func public @unreachable_with_escaping_no_side_effect
util.func public @unreachable_with_escaping_no_side_effect(%input: i32) -> i32 {
  // CHECK: util.unreachable "no side effect path"
  // CHECK-NOT: arith.constant
  // CHECK-NOT: arith.addi
  // CHECK-NOT: util.return
  util.scf.unreachable "no side effect path"
  %c42 = arith.constant 42 : i32
  %c43 = arith.constant 43 : i32
  %val = arith.addi %c42, %c43 : i32
  %result = arith.addi %val, %input : i32
  util.return %result : i32
}

// -----

// Tests that util.scf.unreachable inside a scf.execute_region with CFG-style
// control flow is preserved with operations after it deleted. Even though it's
// in a multi-block region we cannot convert to util.unreachable because the
// region still requires scf.yield as a terminator.

util.func private @side_effect_in_execute_region()

// CHECK-LABEL: util.func public @scf_unreachable_in_execute_region_with_cfg
util.func public @scf_unreachable_in_execute_region_with_cfg(%outer_cond: i1, %cond: i1, %input: i32) -> i32 {
  // CHECK: %[[POISON:.+]] = ub.poison : i32
  // Nest the execute_region inside an scf.if to prevent inlining into func.
  // CHECK: scf.if
  %result = scf.if %outer_cond -> i32 {
    // CHECK: scf.execute_region
    %inner_result = scf.execute_region -> i32 {
      // CHECK: cf.cond_br
      cf.cond_br %cond, ^bb1, ^bb2

    ^bb1:
      // Side effect to prevent the execute_region from being eliminated.
      // CHECK: util.call @side_effect_in_execute_region
      util.call @side_effect_in_execute_region() : () -> ()
      // The util.scf.unreachable is preserved because we're still in an SCF
      // region that requires scf.yield. Operations after it are deleted.
      // CHECK: util.scf.unreachable "unreachable in CFG"
      // CHECK-NOT: arith.constant 42
      // CHECK: scf.yield %[[POISON]]
      util.scf.unreachable "unreachable in CFG"
      %c42 = arith.constant 42 : i32
      scf.yield %c42 : i32

    ^bb2:
      %c100 = arith.constant 100 : i32
      %val = arith.addi %input, %c100 : i32
      // CHECK: scf.yield
      scf.yield %val : i32
    }
    scf.yield %inner_result : i32
  } else {
    %c200 = arith.constant 200 : i32
    scf.yield %c200 : i32
  }
  util.return %result : i32
}

// -----

// Tests that util.scf.unreachable in nested SCF regions properly erases
// operations and replaces results with poison values.

util.func private @nested_side_effect()
util.func private @side_effect_producer() -> i32

// CHECK-LABEL: util.func public @nested_scf_unreachable
util.func public @nested_scf_unreachable(%outer: i1, %inner: i1, %input: i32, %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: scf.for
  %result = scf.for %i = %c0 to %count step %c1 iter_args(%arg = %input) -> i32 {
    // CHECK: scf.if
    %inner_result = scf.if %inner -> i32 {
      // CHECK: util.call @nested_side_effect
      util.call @nested_side_effect() : () -> ()
      // CHECK: util.scf.unreachable "nested unreachable"
      // CHECK-NOT: util.call @side_effect_producer
      util.scf.unreachable "nested unreachable"
      %b0 = util.call @side_effect_producer() : () -> i32
      scf.yield %b0 : i32
    } else {
      %c100 = arith.constant 100 : i32
      scf.yield %c100 : i32
    }
    scf.yield %inner_result : i32
  }
  util.return %result : i32
}

// -----

// Tests that multiple util.scf.unreachable ops in the same function are
// handled correctly.

util.func private @nested_side_effect() -> i32

// CHECK-LABEL: util.func public @multiple_scf_unreachable
util.func public @multiple_scf_unreachable(%cond1: i1, %cond2: i1) -> (i32, i32) {
  // CHECK: %[[POISON:.+]] = ub.poison : i32
  %r1 = scf.if %cond1 -> i32 {
    // CHECK: util.scf.unreachable "first"
    // CHECK: scf.yield %[[POISON]]
    util.scf.unreachable "first"
    %b0 = util.call @nested_side_effect() : () -> i32
    scf.yield %b0 : i32
  } else {
    %b1 = util.call @nested_side_effect() : () -> i32
    scf.yield %b1 : i32
  }
  %r2 = scf.if %cond2 -> i32 {
    %b2 = util.call @nested_side_effect() : () -> i32
    scf.yield %b2 : i32
  } else {
    // CHECK: util.scf.unreachable "second"
    // CHECK: scf.yield %[[POISON]]
    util.scf.unreachable "second"
    %b3 = util.call @nested_side_effect() : () -> i32
    scf.yield %b3 : i32
  }
  util.return %r1, %r2 : i32, i32
}
