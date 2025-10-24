// RUN: iree-opt --iree-stream-replicate-globals-per-affinity --split-input-file %s | FileCheck %s

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[$DEVICE_B:.+]] : !hal.device
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device

// CHECK: util.global private @[[$GLOBAL:.+]] : tensor<10xf32>
// CHECK: util.global private @[[$GLOBAL_B:.+]] : tensor<10xf32>
// CHECK: util.initializer {
// CHECK:   %[[LOAD:.+]] = util.global.load @[[$GLOBAL]]
// CHECK:   %[[TRANSFER_B:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
// CHECK:   util.global.store %[[TRANSFER_B]], @[[$GLOBAL_B]]
// CHECK: }
// CHECK: util.global private @[[$GLOBAL_A:.+]] : tensor<10xf32>
// CHECK: util.initializer {
// CHECK:   %[[LOAD:.+]] = util.global.load @[[$GLOBAL]]
// CHECK:   %[[TRANSFER_A:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
// CHECK:   util.global.store %[[TRANSFER_A]], @[[$GLOBAL_A]]
// CHECK: }
util.global private @global : tensor<10xf32>

// CHECK-LABEL: @unknown_global_device(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
util.func private @unknown_global_device(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  // CHECK: %[[OPERAND_A:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %0 = flow.tensor.transfer %arg0 : tensor<10xf32> to #hal.device.affinity<@device_a>

  // CHECK: %[[LOAD_A:.+]] = util.global.load immutable @[[$GLOBAL_A]]
  %global = util.global.load immutable @global : tensor<10xf32>

  // CHECK: flow.dispatch @dispatch(%[[OPERAND_A]], %[[LOAD_A]])
  %1 = flow.dispatch @dispatch(%0, %global) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>

  // CHECK: %[[OPERAND_B:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
  %2 = flow.tensor.transfer %arg0 : tensor<10xf32> to #hal.device.affinity<@device_b>

  // CHECK: %[[LOAD_B:.+]] = util.global.load immutable @[[$GLOBAL_B]]
  // CHECK: flow.dispatch @dispatch(%[[OPERAND_B]], %[[LOAD_B]])
  %3 = flow.dispatch @dispatch(%2, %global) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  util.return %1, %3 : tensor<10xf32>, tensor<10xf32>
}

// -----

// Test case with a global that has an initializer.
// The new globals should be placed after the original global's initializer.

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[$DEVICE_B:.+]] : !hal.device
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device

// CHECK: util.global private @[[$GLOBAL:.+]] : tensor<10xf32>
// CHECK: util.initializer {
// CHECK:   %[[CST:.+]] = arith.constant dense<0.0{{.*}}> : tensor<10xf32>
// CHECK:   util.global.store %[[CST]], @[[$GLOBAL]]
// CHECK: }
// CHECK: util.global private @[[$GLOBAL_B:.+]] : tensor<10xf32>
// CHECK: util.initializer {
// CHECK:   %[[LOAD:.+]] = util.global.load @[[$GLOBAL]]
// CHECK:   %[[TRANSFER_B:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
// CHECK:   util.global.store %[[TRANSFER_B]], @[[$GLOBAL_B]]
// CHECK: }
// CHECK: util.global private @[[$GLOBAL_A:.+]] : tensor<10xf32>
// CHECK: util.initializer {
// CHECK:   %[[LOAD:.+]] = util.global.load @[[$GLOBAL]]
// CHECK:   %[[TRANSFER_A:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
// CHECK:   util.global.store %[[TRANSFER_A]], @[[$GLOBAL_A]]
// CHECK: }
util.global private @global : tensor<10xf32>
util.initializer {
  %0 = arith.constant dense<0.0> : tensor<10xf32>
  util.global.store %0, @global : tensor<10xf32>
  util.return
}

// CHECK-LABEL: @unknown_global_device(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
util.func private @unknown_global_device(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  // CHECK: %[[OPERAND_A:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %0 = flow.tensor.transfer %arg0 : tensor<10xf32> to #hal.device.affinity<@device_a>

  // CHECK: %[[LOAD_A:.+]] = util.global.load immutable @[[$GLOBAL_A]]
  %global = util.global.load immutable @global : tensor<10xf32>

  // CHECK: flow.dispatch @dispatch(%[[OPERAND_A]], %[[LOAD_A]])
  %1 = flow.dispatch @dispatch(%0, %global) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>

  // CHECK: %[[OPERAND_B:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
  %2 = flow.tensor.transfer %arg0 : tensor<10xf32> to #hal.device.affinity<@device_b>

  // CHECK: %[[LOAD_B:.+]] = util.global.load immutable @[[$GLOBAL_B]]
  // CHECK: flow.dispatch @dispatch(%[[OPERAND_B]], %[[LOAD_B]])
  %3 = flow.dispatch @dispatch(%2, %global) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  util.return %1, %3 : tensor<10xf32>, tensor<10xf32>
}

// -----

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[$DEVICE_B:.+]] : !hal.device
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device

// CHECK: util.global private @[[$GLOBAL:.+]] : tensor<f32>
// CHECK: util.global private @[[$GLOBAL_B:.+]] : tensor<f32>
// CHECK: util.initializer {
// CHECK:   %[[LOAD:.+]] = util.global.load @[[$GLOBAL]]
// CHECK:   %[[TRANSFER_B:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
// CHECK:   util.global.store %[[TRANSFER_B]], @[[$GLOBAL_B]]
// CHECK: }
// CHECK: util.global private @[[$GLOBAL_A:.+]] : tensor<f32>
// CHECK: util.initializer {
// CHECK:   %[[LOAD:.+]] = util.global.load @[[$GLOBAL]]
// CHECK:   %[[TRANSFER_A:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
// CHECK:   util.global.store %[[TRANSFER_A]], @[[$GLOBAL_A]]
// CHECK: }
util.global private @global : tensor<f32>

// CHECK-LABEL: @ambiguous_indirect_global_load(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
util.func private @ambiguous_indirect_global_load(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %global = util.global.load immutable @global : tensor<f32>

  // CHECK-DAG: %[[LOAD_A:.+]] = util.global.load immutable @[[$GLOBAL_A]]
  // CHECK-DAG: %[[LOAD_B:.+]] = util.global.load immutable @[[$GLOBAL_B]]
  // CHECK-DAG: %[[LHS_A:.+]] = flow.dispatch @dispatch0(%[[LOAD_A]])
  // CHECK-DAG: %[[LHS_B:.+]] = flow.dispatch @dispatch0(%[[LOAD_B]])
  %0 = flow.dispatch @dispatch0(%global) : (tensor<f32>) -> tensor<f32>

  // CHECK: %[[RHS_A:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %1 = flow.tensor.transfer %arg0 : tensor<f32> to #hal.device.affinity<@device_a>

  // CHECK: %[[RES_A:.+]] = flow.dispatch @dispatch1(%[[LHS_A]], %[[RHS_A]])
  %2 = flow.dispatch @dispatch1(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK: %[[RHS_B:.+]] = flow.tensor.transfer %[[ARG1]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
  %3 = flow.tensor.transfer %arg1 : tensor<f32> to #hal.device.affinity<@device_b>

  // CHECK: %[[RES_B:.+]] = flow.dispatch @dispatch2(%[[LHS_B]], %[[RHS_B]])
  %4 = flow.dispatch @dispatch2(%0, %3) : (tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK: util.return %[[RES_A]], %[[RES_B]]
  util.return %2, %4 : tensor<f32>, tensor<f32>
}

// -----

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[$DEVICE_B:.+]] : !hal.device
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device

// CHECK: util.global private @[[$FOO:.+]] : tensor<2x5xf32>
// CHECK: util.global private @[[$FOO_B:.+]] : tensor<2x5xf32>
// CHECK: util.initializer {
// CHECK:   %[[LOAD:.+]] = util.global.load @[[$FOO]]
// CHECK:   %[[TRANSFER_B:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
// CHECK:   util.global.store %[[TRANSFER_B]], @[[$FOO_B]]
// CHECK: }
// CHECK: util.global private @[[$FOO_A:.+]] : tensor<2x5xf32>
// CHECK: util.initializer {
// CHECK:   %[[LOAD:.+]] = util.global.load @[[$FOO]]
// CHECK:   %[[TRANSFER_A:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
// CHECK:   util.global.store %[[TRANSFER_A]], @[[$FOO_A]]
// CHECK: }
util.global private @foo : tensor<2x5xf32>

// CHECK:  util.global private @[[$BAR:.+]] : tensor<10xf32>
// CHECK: util.global private @[[$BAR_B:.+]] : tensor<10xf32>
// CHECK: util.initializer {
// CHECK:   %[[LOAD:.+]] = util.global.load @[[$BAR]]
// CHECK:   %[[TRANSFER_B:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
// CHECK:   util.global.store %[[TRANSFER_B]], @[[$BAR_B]]
// CHECK: }
// CHECK: util.global private @[[$BAR_A:.+]] : tensor<10xf32>
// CHECK: util.initializer {
// CHECK:   %[[LOAD:.+]] = util.global.load @[[$BAR]]
// CHECK:   %[[TRANSFER_A:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
// CHECK:   util.global.store %[[TRANSFER_A]], @[[$BAR_A]]
// CHECK: }
util.global private @bar : tensor<10xf32>

// CHECK-LABEL: @ambiguous_indirect_global_load_multi_levels(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
util.func private @ambiguous_indirect_global_load_multi_levels(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  %foo = util.global.load immutable @foo : tensor<2x5xf32>
  %bar = util.global.load immutable @bar : tensor<10xf32>

  // CHECK-DAG: %[[LOAD_FOO_A:.+]] = util.global.load immutable @[[$FOO_A]]
  // CHECK-DAG: %[[LOAD_FOO_B:.+]] = util.global.load immutable @[[$FOO_B]]
  // CHECK-DAG: %[[VAL_A:.+]] = flow.dispatch @dispatch0(%[[LOAD_FOO_A]])
  // CHECK-DAG: %[[RESHAPE_A:.+]] = flow.tensor.reshape %[[VAL_A]] : tensor<2x5xf32> -> tensor<10xf32>
  // CHECK-DAG: %[[VAL_B:.+]] = flow.dispatch @dispatch0(%[[LOAD_FOO_B]])
  // CHECK-DAG: %[[RESHAPE_B:.+]] = flow.tensor.reshape %[[VAL_B]] : tensor<2x5xf32> -> tensor<10xf32>
  %0 = flow.dispatch @dispatch0(%foo) : (tensor<2x5xf32>) -> tensor<2x5xf32>
  %1 = flow.tensor.reshape %0 : tensor<2x5xf32> -> tensor<10xf32>

  // CHECK-DAG: %[[LOAD_BAR_A:.+]] = util.global.load immutable @[[$BAR_A]]
  // CHECK-DAG: %[[LOAD_BAR_B:.+]] = util.global.load immutable @[[$BAR_B]]
  // CHECK-DAG: %[[LHS_A:.+]] = flow.dispatch @dispatch1(%[[RESHAPE_A]], %[[LOAD_BAR_A]])
  // CHECK-DAG: %[[LHS_B:.+]] = flow.dispatch @dispatch1(%[[RESHAPE_B]], %[[LOAD_BAR_B]])
  %2 = flow.dispatch @dispatch1(%1, %bar) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>

  // CHECK: %[[RHS_A:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %3 = flow.tensor.transfer %arg0 : tensor<10xf32> to #hal.device.affinity<@device_a>

  // CHECK: %[[RES_A:.+]] = flow.dispatch @dispatch2(%[[LHS_A]], %[[RHS_A]])
  %4 = flow.dispatch @dispatch2(%2, %3) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>

  // CHECK: %[[RHS_B:.+]] = flow.tensor.transfer %[[ARG1]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
  %5 = flow.tensor.transfer %arg1 : tensor<10xf32> to #hal.device.affinity<@device_b>

  // CHECK: %[[RES_B:.+]] = flow.dispatch @dispatch3(%[[LHS_B]], %[[RHS_B]])
  %6 = flow.dispatch @dispatch3(%2, %5) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>

  // CHECK: util.return %[[RES_A]], %[[RES_B]]
  util.return %4, %6 : tensor<10xf32>, tensor<10xf32>
}

// -----

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[$DEVICE_B:.+]] : !hal.device
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device
// CHECK: util.global private @[[$GLOBAL:.+]] : tensor<f32>
// CHECK: util.global private @[[$GLOBAL_B:.+]] : tensor<f32>
// CHECK: util.initializer {
// CHECK:   %[[LOAD:.+]] = util.global.load @[[$GLOBAL]]
// CHECK:   %[[TRANSFER_B:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
// CHECK:   util.global.store %[[TRANSFER_B]], @[[$GLOBAL_B]]
// CHECK: }
// CHECK: util.global private @[[$GLOBAL_A:.+]] : tensor<f32>
// CHECK: util.initializer {
// CHECK:   %[[LOAD:.+]] = util.global.load @[[$GLOBAL]]
// CHECK:   %[[TRANSFER_A:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
// CHECK:   util.global.store %[[TRANSFER_A]], @[[$GLOBAL_A]]
// CHECK: }
util.global private @global : tensor<f32>

// CHECK-LABEL: @launch_on_device_a(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
util.func private @launch_on_device_a(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: %[[OPERAND0:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %0 = flow.tensor.transfer %arg0 : tensor<f32> to #hal.device.affinity<@device_a>
  // CHECK: %[[OPERAND1:.+]] = util.global.load immutable @[[$GLOBAL_A]]
  // CHECK: flow.dispatch @dispatch(%[[OPERAND0]], %[[OPERAND1]])
  %1 = flow.dispatch @dispatch(%0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  util.return %1 : tensor<f32>
}

util.func private @launch_on_device_b(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: %[[OPERAND0:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
  %0 = flow.tensor.transfer %arg0 : tensor<f32> to #hal.device.affinity<@device_b>
  // CHECK: %[[OPERAND1:.+]] = util.global.load immutable @[[$GLOBAL_B]]
  // CHECK: flow.dispatch @dispatch(%[[OPERAND0]], %[[OPERAND1]])
  %1 = flow.dispatch @dispatch(%0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  util.return %1 : tensor<f32>
}

// CHECK-LABEL: @unknown_global_affinity_cross_functions(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
util.func private @unknown_global_affinity_cross_functions(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  // CHECK: %[[LOAD:.+]] = util.global.load immutable @[[$GLOBAL]]
  %global = util.global.load immutable @global : tensor<f32>

  // CHECK: util.call @launch_on_device_a(%[[ARG0]], %[[LOAD]])
  %1 = util.call @launch_on_device_a(%arg0, %global) : (tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK: util.call @launch_on_device_b(%[[ARG0]], %[[LOAD]])
  %2 = util.call @launch_on_device_b(%arg0, %global) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  util.return %1, %2 : tensor<f32>, tensor<f32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative tests. IR is not mutated after running the pass.
//===----------------------------------------------------------------------===//

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[DEVICE_B:.+]] : !hal.device
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device
util.global private @global {stream.affinity = #hal.device.affinity<@device_b>} : tensor<f32>

// CHECK-LABEL: @load
util.func private @load() -> tensor<f32> {
  // CHECK: %[[LOAD:.+]] = util.global.load immutable @[[$GLOBAL]]
  %global = util.global.load immutable @global : tensor<f32>
  // CHECK: util.return %[[LOAD]]
  util.return %global : tensor<f32>
}

// CHECK-LABEL: @negative_affinity_mismatch_cross_functions(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
util.func private @negative_affinity_mismatch_cross_functions(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: %[[OPERAND_A:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %0 = flow.tensor.transfer %arg0 : tensor<f32> to #hal.device.affinity<@device_a>

  // CHECK: %[[CALL:.+]] = util.call @load()
  %1 = util.call @load() : () -> tensor<f32>

  // CHECK: flow.dispatch @dispatch(%[[OPERAND_A]], %[[CALL]])
  %2 = flow.dispatch @dispatch(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  util.return %1 : tensor<f32>
}

// -----

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[DEVICE_B:.+]] : !hal.device
// CHECK: util.global private @[[$GLOBAL:.+]] {stream.affinity = #hal.device.affinity<@[[DEVICE_B]]>} : tensor<f32>
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device
util.global private @global {stream.affinity = #hal.device.affinity<@device_b>} : tensor<f32>

// CHECK-LABEL: @negative_invalid_program(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
util.func private @negative_invalid_program(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: %[[OPERAND0:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %0 = flow.tensor.transfer %arg0 : tensor<f32> to #hal.device.affinity<@device_a>
  // CHECK: %[[OPERAND1:.+]] = flow.tensor.transfer %[[ARG1]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
  %1 = flow.tensor.transfer %arg1 : tensor<f32> to #hal.device.affinity<@device_b>
  // CHECK: %[[LOAD:.+]] = util.global.load immutable @[[$GLOBAL]]
  %global = util.global.load immutable @global : tensor<f32>
  // CHECK: flow.dispatch @dispatch(%[[OPERAND0]], %[[OPERAND1]], %[[LOAD]])
  %2 = flow.dispatch @dispatch(%0, %1, %global) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  util.return %1 : tensor<f32>
}

// -----

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[DEVICE_B:.+]] : !hal.device
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device
util.global private @global {stream.affinity = #hal.device.affinity<@device_b>} : tensor<f32>

// CHECK-LABEL: @negative_affinity_mismatch_with_known_global_affinity(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
util.func private @negative_affinity_mismatch_with_known_global_affinity(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: %[[OPERAND_A:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %0 = flow.tensor.transfer %arg0 : tensor<f32> to #hal.device.affinity<@device_a>
  %global = util.global.load immutable @global : tensor<f32>

  // CHECK: %[[LOAD_B:.+]] = util.global.load immutable @[[$GLOBAL]]
  // CHECK: flow.dispatch @dispatch(%[[OPERAND_A]], %[[LOAD_B]])
  %1 = flow.dispatch @dispatch(%0, %global) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  util.return %1 : tensor<f32>
}

// -----

// Note: it is an invalid/unoptimized program because scalars are supposed to be
// used on host code.

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[$DEVICE_B:.+]] : !hal.device
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device
util.global private @global : f32

// CHECK-LABEL: @negative_scalar_global_with_unknown_global_affinity(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
util.func private @negative_scalar_global_with_unknown_global_affinity(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  // CHECK: %[[OPERAND_A:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %0 = flow.tensor.transfer %arg0 : tensor<f32> to #hal.device.affinity<@device_a>

  // CHECK: %[[LOAD:.+]] = util.global.load immutable @[[$GLOBAL]]
  %global = util.global.load immutable @global : f32

  // CHECK: flow.dispatch @dispatch(%[[OPERAND_A]], %[[LOAD]])
  %1 = flow.dispatch @dispatch(%0, %global) : (tensor<f32>, f32) -> tensor<f32>

  // CHECK: %[[OPERAND_B:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
  %2 = flow.tensor.transfer %arg0 : tensor<f32> to #hal.device.affinity<@device_b>

  // CHECK: flow.dispatch @dispatch(%[[OPERAND_B]], %[[LOAD]])
  %3 = flow.dispatch @dispatch(%2, %global) : (tensor<f32>, f32) -> tensor<f32>
  util.return %1, %3 : tensor<f32>, tensor<f32>
}
