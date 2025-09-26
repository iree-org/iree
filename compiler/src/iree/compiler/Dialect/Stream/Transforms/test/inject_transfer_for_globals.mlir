// RUN: iree-opt --iree-stream-inject-transfor-for-globals --split-input-file %s | FileCheck %s

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[$DEVICE_B:.+]] : !hal.device
// CHECK: util.global private @[[$GLOBAL:.+]] : tensor<f16>
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device
util.global private @global : tensor<f16>
// CHECK-LABEL: @unknown_global_device(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
util.func private @unknown_global_device(%arg0: tensor<f16>) -> (tensor<f16>, tensor<f16>) {
  // CHECK: %[[OPERAND_A:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %0 = flow.tensor.transfer %arg0 : tensor<f16> to #hal.device.affinity<@device_a>

  // CHECK: %[[LOAD:.+]] = util.global.load immutable @[[$GLOBAL]]
  // CHECK: %[[LOAD_A:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  // CHECK: %[[LOAD_B:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
  %global = util.global.load immutable @global : tensor<f16>

  // CHECK: flow.dispatch @dispatch(%[[OPERAND_A]], %[[LOAD_A]])
  %1 = flow.dispatch @dispatch(%0, %global) : (tensor<f16>, tensor<f16>) -> tensor<f16>

  // CHECK: %[[OPERAND_B:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
  %2 = flow.tensor.transfer %arg0 : tensor<f16> to #hal.device.affinity<@device_b>

  // CHECK: flow.dispatch @dispatch(%[[OPERAND_B]], %[[LOAD_B]])
  %3 = flow.dispatch @dispatch(%2, %global) : (tensor<f16>, tensor<f16>) -> tensor<f16>
  util.return %1, %3 : tensor<f16>, tensor<f16>
}

// -----

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[DEVICE_B:.+]] : !hal.device
// CHECK: util.global private @[[$GLOBAL:.+]] {stream.affinity = #hal.device.affinity<@[[DEVICE_B]]>} : tensor<f16>
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device
util.global private @global {stream.affinity = #hal.device.affinity<@device_b>} : tensor<f16>
// CHECK-LABEL: @known_global_device(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
util.func private @known_global_device(%arg0: tensor<f16>) -> tensor<f16> {
  // CHECK: %[[OPERAND_A:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %0 = flow.tensor.transfer %arg0 : tensor<f16> to #hal.device.affinity<@device_a>

  // CHECK: %[[LOAD:.+]] = util.global.load immutable @[[$GLOBAL]]
  // CHECK: %[[LOAD_A:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %global = util.global.load immutable @global : tensor<f16>

  // CHECK: flow.dispatch @dispatch(%[[OPERAND_A]], %[[LOAD_A]])
  %1 = flow.dispatch @dispatch(%0, %global) : (tensor<f16>, tensor<f16>) -> tensor<f16>
  util.return %1 : tensor<f16>
}

// -----

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[$DEVICE_B:.+]] : !hal.device
// CHECK: util.global private @[[$GLOBAL:.+]] : tensor<f16>
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device
util.global private @global : tensor<f16>

// CHECK-LABEL: @launch_on_device_a(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
util.func private @launch_on_device_a(%arg0: tensor<f16>, %arg1: tensor<f16>) -> tensor<f16> {
  // CHECK: %[[OPERAND1:.+]] = flow.tensor.transfer %[[ARG1]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  // CHECK: %[[OPERAND0:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %0 = flow.tensor.transfer %arg0 : tensor<f16> to #hal.device.affinity<@device_a>
  // CHECK: flow.dispatch @dispatch(%[[OPERAND0]], %[[OPERAND1]])
  %1 = flow.dispatch @dispatch(%0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
  util.return %1 : tensor<f16>
}

util.func private @launch_on_device_b(%arg0: tensor<f16>, %arg1: tensor<f16>) -> tensor<f16> {
  // CHECK: %[[OPERAND1:.+]] = flow.tensor.transfer %[[ARG1]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
  // CHECK: %[[OPERAND0:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
  %0 = flow.tensor.transfer %arg0 : tensor<f16> to #hal.device.affinity<@device_b>
  // CHECK: flow.dispatch @dispatch(%[[OPERAND0]], %[[OPERAND1]])
  %1 = flow.dispatch @dispatch(%0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
  util.return %1 : tensor<f16>
}

// CHECK-LABEL: @unknown_global_device_cross_function(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
util.func private @unknown_global_device_cross_function(%arg0: tensor<f16>) -> (tensor<f16>, tensor<f16>) {
  // CHECK: %[[LOAD:.+]] = util.global.load immutable @[[$GLOBAL]]
  %global = util.global.load immutable @global : tensor<f16>

  // CHECK: util.call @launch_on_device_a(%[[ARG0]], %[[LOAD]])
  %1 = util.call @launch_on_device_a(%arg0, %global) : (tensor<f16>, tensor<f16>) -> tensor<f16>

  // CHECK: util.call @launch_on_device_b(%[[ARG0]], %[[LOAD]])
  %2 = util.call @launch_on_device_b(%arg0, %global) : (tensor<f16>, tensor<f16>) -> tensor<f16>
  util.return %1, %2 : tensor<f16>, tensor<f16>
}

// -----

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[DEVICE_B:.+]] : !hal.device
// CHECK: util.global private @[[$GLOBAL:.+]] {stream.affinity = #hal.device.affinity<@[[DEVICE_B]]>} : tensor<f16>
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device
util.global private @global {stream.affinity = #hal.device.affinity<@device_b>} : tensor<f16>

// CHECK-LABEL: @load
util.func private @load() -> tensor<f16> {
  // CHECK: %[[LOAD:.+]] = util.global.load immutable @[[$GLOBAL]]
  %global = util.global.load immutable @global : tensor<f16>
  // CHECK: util.return %[[LOAD]]
  util.return %global : tensor<f16>
}

// CHECK-LABEL: @known_global_device_load_from_func_call(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
util.func private @known_global_device_load_from_func_call(%arg0: tensor<f16>) -> tensor<f16> {
  // CHECK: %[[OPERAND_A:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %0 = flow.tensor.transfer %arg0 : tensor<f16> to #hal.device.affinity<@device_a>

  // CHECK: %[[LOAD:.+]] = util.call @load
  // CHECK: %[[LOAD_A:.+]] = flow.tensor.transfer %[[LOAD]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %1 = util.call @load() : () -> tensor<f16>

  // CHECK: flow.dispatch @dispatch(%[[OPERAND_A]], %[[LOAD_A]])
  %2 = flow.dispatch @dispatch(%0, %1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
  util.return %1 : tensor<f16>
}

// -----

// CHECK: util.global private @[[$DEVICE_A:.+]] : !hal.device
// CHECK: util.global private @[[DEVICE_B:.+]] : !hal.device
// CHECK: util.global private @[[$GLOBAL:.+]] {stream.affinity = #hal.device.affinity<@[[DEVICE_B]]>} : tensor<f16>
util.global private @device_a : !hal.device
util.global private @device_b : !hal.device
util.global private @global {stream.affinity = #hal.device.affinity<@device_b>} : tensor<f16>

// CHECK-LABEL: @negative_invalid_program(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
util.func private @negative_invalid_program(%arg0: tensor<f16>, %arg1: tensor<f16>) -> tensor<f16> {
  // CHECK: %[[OPERAND0:.+]] = flow.tensor.transfer %[[ARG0]] {{.+}} to #hal.device.affinity<@[[$DEVICE_A]]>
  %0 = flow.tensor.transfer %arg0 : tensor<f16> to #hal.device.affinity<@device_a>
  // CHECK: %[[OPERAND1:.+]] = flow.tensor.transfer %[[ARG1]] {{.+}} to #hal.device.affinity<@[[$DEVICE_B]]>
  %1 = flow.tensor.transfer %arg1 : tensor<f16> to #hal.device.affinity<@device_b>
  // CHECK: %[[LOAD:.+]] = util.global.load immutable @[[$GLOBAL]]
  %global = util.global.load immutable @global : tensor<f16>
  // CHECK: flow.dispatch @dispatch(%[[OPERAND0]], %[[OPERAND1]], %[[LOAD]])
  %2 = flow.dispatch @dispatch(%0, %1, %global) : (tensor<f16>, tensor<f16>, tensor<f16>) -> tensor<f16>
  util.return %1 : tensor<f16>
}
