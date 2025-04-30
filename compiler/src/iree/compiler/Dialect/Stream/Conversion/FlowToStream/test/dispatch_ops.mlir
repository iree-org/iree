// RUN: iree-opt --split-input-file --iree-stream-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @dispatchNoWorkload
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index, %[[DIM1:.+]]: index, %[[DIM3:.+]]: index)
util.func public @dispatchNoWorkload(%input: tensor<7x?x24x?xf32>, %dim1: index, %dim3: index) -> tensor<?x?x1024xf32> {
  // CHECK: %[[RESULT_SIZE:.+]] = stream.tensor.sizeof tensor<?x?x1024xf32>{%[[DIM1]], %[[DIM3]]}
  // CHECK: %[[RESULT:.+]] = stream.tensor.dispatch @ex::@entry(%[[INPUT]]) :
  // CHECK-SAME: (tensor<7x?x24x?xf32>{%[[DIM1]], %[[DIM3]]} in !stream.resource<*>{%[[INPUT_SIZE]]}) -> tensor<?x?x1024xf32>{%[[DIM1]], %[[DIM3]]} in !stream.resource<*>{%[[RESULT_SIZE]]}
  %0 = flow.dispatch @ex::@entry(%input) : (tensor<7x?x24x?xf32>{%dim1, %dim3}) -> tensor<?x?x1024xf32>{%dim1, %dim3}
  // CHECK: util.return %[[RESULT]], %[[RESULT_SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<?x?x1024xf32>
}

// -----

// CHECK-LABEL: @dispatch
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index, %[[DIM1:.+]]: index, %[[DIM3:.+]]: index)
util.func public @dispatch(%input: tensor<7x?x24x?xf32>, %dim1: index, %dim3: index) -> (tensor<?x?x1024xf32>, tensor<1024x?x?xf32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK: %[[RESULT0_SIZE:.+]] = stream.tensor.sizeof tensor<?x?x1024xf32>{%[[DIM1]], %[[DIM3]]}
  // CHECK: %[[RESULT1_SIZE:.+]] = stream.tensor.sizeof tensor<1024x?x?xf32>{%[[DIM3]], %[[DIM1]]}
  // CHECK: %[[RESULTS:.+]]:2 = stream.tensor.dispatch @ex::@entry[%c1, %c2, %c3](%[[INPUT]]) :
  // CHECK-SAME: (tensor<7x?x24x?xf32>{%[[DIM1]], %[[DIM3]]} in !stream.resource<*>{%[[INPUT_SIZE]]}) -> (tensor<?x?x1024xf32>{%[[DIM1]], %[[DIM3]]} in !stream.resource<*>{%[[RESULT0_SIZE]]}, tensor<1024x?x?xf32>{%[[DIM3]], %[[DIM1]]} in !stream.resource<*>{%[[RESULT1_SIZE]]})
  %results:2 = flow.dispatch @ex::@entry[%c1, %c2, %c3](%input) : (tensor<7x?x24x?xf32>{%dim1, %dim3}) -> (tensor<?x?x1024xf32>{%dim1, %dim3}, tensor<1024x?x?xf32>{%dim3, %dim1})
  // CHECK: util.return %[[RESULTS]]#0, %[[RESULT0_SIZE]], %[[RESULTS]]#1, %[[RESULT1_SIZE]] : !stream.resource<*>, index, !stream.resource<*>, index
  util.return %results#0, %results#1 : tensor<?x?x1024xf32>, tensor<1024x?x?xf32>
}

// -----

// CHECK-LABEL: @tiedDispatch
//  CHECK-SAME: (%[[INPUT0:.+]]: !stream.resource<*>, %[[INPUT0_SIZE:.+]]: index, %[[INPUT1:.+]]: !stream.resource<*>, %[[INPUT1_SIZE:.+]]: index)
util.func public @tiedDispatch(%input0: tensor<i32>, %input1: tensor<2x3xi32>) -> tensor<3x9xi32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK: %[[T_SIZE:.+]] = stream.tensor.sizeof tensor<3x9xi32> : index
  // CHECK: %[[T:.+]] = stream.tensor.dispatch @ex::@entry0[%c1, %c2, %c3](%[[INPUT0]]) :
  // CHECK-SAME: (tensor<i32> in !stream.resource<*>{%[[INPUT0_SIZE]]}) -> tensor<3x9xi32> in !stream.resource<*>{%[[T_SIZE]]}
  %0 = flow.dispatch @ex::@entry0[%c1, %c2, %c3](%input0) : (tensor<i32>) -> tensor<3x9xi32>
  // CHECK: %[[RESULT:.+]] = stream.tensor.dispatch @ex::@entry1[%c1, %c2, %c3](%[[INPUT1]], %[[T]]) :
  // CHECK-SAME: (tensor<2x3xi32> in !stream.resource<*>{%[[INPUT1_SIZE]]}, tensor<3x9xi32> in !stream.resource<*>{%[[T_SIZE]]}) -> tensor<3x9xi32> in %[[T]]{%[[T_SIZE]]}
  %1 = flow.dispatch @ex::@entry1[%c1, %c2, %c3](%input1, %0) : (tensor<2x3xi32>, tensor<3x9xi32>) -> %0
  // CHECK: util.return %[[RESULT]], %[[T_SIZE]] : !stream.resource<*>, index
  util.return %1 : tensor<3x9xi32>
}

// -----

util.global private @device_a : !hal.device
util.global private @device_b : !hal.device

// CHECK-LABEL: @dispatchAffinity
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index, %[[DIM1:.+]]: index, %[[DIM3:.+]]: index)
util.func public @dispatchAffinity(%input: tensor<7x?x24x?xf32>, %dim1: index, %dim3: index) -> (tensor<?x?x1024xf32>, tensor<?x?x1024xf32>) {
  // CHECK: %[[INPUT_A:.+]] = stream.async.transfer %[[INPUT]] : !stream.resource<*>{%[[INPUT_SIZE]]} -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%[[INPUT_SIZE]]}
  // CHECK: %[[RESULT0_SIZE:.+]] = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<?x?x1024xf32>{%[[DIM1]], %[[DIM3]]}
  // CHECK: %[[RESULT0:.+]] = stream.tensor.dispatch on(#hal.device.affinity<@device_a>) @ex::@entry0(%[[INPUT_A]])
  // CHECK-SAME: (tensor<7x?x24x?xf32>{%[[DIM1]], %[[DIM3]]} in !stream.resource<*>{%[[INPUT_SIZE]]}) -> tensor<?x?x1024xf32>{%[[DIM1]], %[[DIM3]]} in !stream.resource<*>{%[[RESULT0_SIZE]]}
  %0 = flow.dispatch @ex::@entry0(%input) {
    stream.affinity = #hal.device.affinity<@device_a>
  } : (tensor<7x?x24x?xf32>{%dim1, %dim3}) -> tensor<?x?x1024xf32>{%dim1, %dim3}
  // CHECK: %[[INPUT_B:.+]] = stream.async.transfer %[[INPUT]] : !stream.resource<*>{%[[INPUT_SIZE]]} -> to(#hal.device.affinity<@device_b>) !stream.resource<*>{%[[INPUT_SIZE]]}
  // CHECK: %[[RESULT1_SIZE:.+]] = stream.tensor.sizeof on(#hal.device.affinity<@device_b>) tensor<?x?x1024xf32>{%[[DIM3]], %[[DIM1]]}
  // CHECK: %[[RESULT1:.+]] = stream.tensor.dispatch on(#hal.device.affinity<@device_b>) @ex::@entry1(%[[INPUT_B]])
  // CHECK-SAME: (tensor<7x?x24x?xf32>{%[[DIM1]], %[[DIM3]]} in !stream.resource<*>{%[[INPUT_SIZE]]}) -> tensor<?x?x1024xf32>{%[[DIM3]], %[[DIM1]]} in !stream.resource<*>{%[[RESULT1_SIZE]]}
  %1 = flow.dispatch @ex::@entry1(%input) {
    stream.affinity = #hal.device.affinity<@device_b>
  } : (tensor<7x?x24x?xf32>{%dim1, %dim3}) -> tensor<?x?x1024xf32>{%dim3, %dim1}
  // CHECK: return %[[RESULT0]], %[[RESULT0_SIZE]], %[[RESULT1]], %[[RESULT1_SIZE]]
  util.return %0, %1 : tensor<?x?x1024xf32>, tensor<?x?x1024xf32>
}
