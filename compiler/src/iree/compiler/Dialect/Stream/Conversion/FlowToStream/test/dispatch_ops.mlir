// RUN: iree-opt --split-input-file --iree-stream-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @dispatchNoWorkload
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index, %[[DIM1:.+]]: index, %[[DIM3:.+]]: index)
util.func public @dispatchNoWorkload(%input: tensor<7x?x24x?xf32>, %dim1: index, %dim3: index) -> tensor<?x?x1024xf32> {
  //      CHECK: %[[RESULT_SIZE:.+]] = stream.tensor.sizeof tensor<?x?x1024xf32>{%[[DIM1]], %[[DIM3]]}
  //      CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex::@entry(%[[INPUT]][%c0 to %[[INPUT_SIZE]] for %[[INPUT_SIZE]]]) :
  // CHECK-SAME:     (!stream.resource<*>{%[[INPUT_SIZE]]}) -> !stream.resource<*>{%[[RESULT_SIZE]]}
  %0 = flow.dispatch @ex::@entry(%input) : (tensor<7x?x24x?xf32>{%dim1, %dim3}) -> tensor<?x?x1024xf32>{%dim1, %dim3}
  // return %[[RESULT]], %[[RESULT_SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<?x?x1024xf32>
}

// -----

// CHECK-LABEL: @dispatch
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index, %[[DIM1:.+]]: index, %[[DIM3:.+]]: index)
util.func public @dispatch(%input: tensor<7x?x24x?xf32>, %dim1: index, %dim3: index) -> tensor<?x?x1024xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  //      CHECK: %[[RESULT_SIZE:.+]] = stream.tensor.sizeof tensor<?x?x1024xf32>{%[[DIM1]], %[[DIM3]]}
  //      CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex::@entry[%c1, %c2, %c3](%[[INPUT]][%c0 to %[[INPUT_SIZE]] for %[[INPUT_SIZE]]]) :
  // CHECK-SAME:     (!stream.resource<*>{%[[INPUT_SIZE]]}) -> !stream.resource<*>{%[[RESULT_SIZE]]}
  %0 = flow.dispatch @ex::@entry[%c1, %c2, %c3](%input) : (tensor<7x?x24x?xf32>{%dim1, %dim3}) -> tensor<?x?x1024xf32>{%dim1, %dim3}
  // return %[[RESULT]], %[[RESULT_SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<?x?x1024xf32>
}

// -----

// CHECK-LABEL: @tiedDispatch
//  CHECK-SAME: (%[[INPUT0:.+]]: !stream.resource<*>, %[[INPUT0_SIZE:.+]]: index, %[[INPUT1:.+]]: !stream.resource<*>, %[[INPUT1_SIZE:.+]]: index)
util.func public @tiedDispatch(%input0: tensor<i32>, %input1: tensor<2x3xi32>) -> tensor<3x9xi32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK: %[[T_SIZE:.+]] = stream.tensor.sizeof tensor<3x9xi32> : index
  // CHECK: %[[T:.+]] = stream.async.dispatch @ex::@entry0[%c1, %c2, %c3](%[[INPUT0]][%c0 to %[[INPUT0_SIZE]] for %[[INPUT0_SIZE]]]) : (!stream.resource<*>{%[[INPUT0_SIZE]]}) -> !stream.resource<*>{%[[T_SIZE]]}
  %0 = flow.dispatch @ex::@entry0[%c1, %c2, %c3](%input0) : (tensor<i32>) -> tensor<3x9xi32>
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex::@entry1[%c1, %c2, %c3](%[[INPUT1]][%c0 to %[[INPUT1_SIZE]] for %[[INPUT1_SIZE]]], %[[T]][%c0 to %[[T_SIZE]] for %[[T_SIZE]]]) : (!stream.resource<*>{%[[INPUT1_SIZE]]}, !stream.resource<*>{%[[T_SIZE]]}) -> %[[T]]{%[[T_SIZE]]}
  %1 = flow.dispatch @ex::@entry1[%c1, %c2, %c3](%input1, %0) : (tensor<2x3xi32>, tensor<3x9xi32>) -> %0
  // CHECK: util.return %[[RESULT]], %[[T_SIZE]] : !stream.resource<*>, index
  util.return %1 : tensor<3x9xi32>
}

// -----

// CHECK-LABEL: @dispatchAffinity
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index, %[[DIM1:.+]]: index, %[[DIM3:.+]]: index)
util.func public @dispatchAffinity(%input: tensor<7x?x24x?xf32>, %dim1: index, %dim3: index) -> (tensor<?x?x1024xf32>, tensor<?x?x1024xf32>) {
  //      CHECK: %[[RESULT0_SIZE:.+]] = stream.tensor.sizeof on(#hal.affinity.queue<[0]>) tensor<?x?x1024xf32>{%[[DIM1]], %[[DIM3]]}
  //      CHECK: %[[RESULT0:.+]] = stream.async.dispatch on(#hal.affinity.queue<[0]>) @ex::@entry0(%[[INPUT]][%c0 to %[[INPUT_SIZE]] for %[[INPUT_SIZE]]])
  %0 = flow.dispatch @ex::@entry0(%input) {
    stream.affinity = #hal.affinity.queue<[0]>
  } : (tensor<7x?x24x?xf32>{%dim1, %dim3}) -> tensor<?x?x1024xf32>{%dim1, %dim3}
  //      CHECK: %[[RESULT1_SIZE:.+]] = stream.tensor.sizeof on(#hal.affinity.queue<[1]>) tensor<?x?x1024xf32>{%[[DIM3]], %[[DIM1]]}
  //      CHECK: %[[RESULT1:.+]] = stream.async.dispatch on(#hal.affinity.queue<[1]>) @ex::@entry1(%[[INPUT]][%c0 to %[[INPUT_SIZE]] for %[[INPUT_SIZE]]])
  %1 = flow.dispatch @ex::@entry1(%input) {
    stream.affinity = #hal.affinity.queue<[1]>
  } : (tensor<7x?x24x?xf32>{%dim1, %dim3}) -> tensor<?x?x1024xf32>{%dim3, %dim1}
  // return %[[RESULT0]], %[[RESULT0_SIZE]], %[[RESULT1]], %[[RESULT1_SIZE]]
  util.return %0, %1 : tensor<?x?x1024xf32>, tensor<?x?x1024xf32>
}
