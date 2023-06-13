// RUN: iree-opt --split-input-file --iree-stream-conversion %s | FileCheck %s

// CHECK-LABEL: @tensorReshapePassThrough
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index)
func.func @tensorReshapePassThrough(%input: tensor<5x24x48xf32>) -> tensor<30x2x96xf32> {
  // CHECK: %[[RESULT_SIZE:.+]] = stream.tensor.sizeof tensor<30x2x96xf32> : index
  // CHECK: %[[RESULT:.+]] = stream.tensor.clone %[[INPUT]] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<30x2x96xf32> in !stream.resource<*>{%[[RESULT_SIZE]]}
  %0 = flow.tensor.reshape %input : tensor<5x24x48xf32> -> tensor<30x2x96xf32>
  // CHECK: return %[[RESULT]], %[[RESULT_SIZE]] : !stream.resource<*>, index
  return %0 : tensor<30x2x96xf32>
}

// -----

// CHECK-LABEL: @tensorReshapeWithSingleUse
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index)
func.func @tensorReshapeWithSingleUse(%input: tensor<5x24x48xf32>) -> tensor<30x2x96xf32> {
  // CHECK: %[[RESULT_SIZE:.+]] = stream.tensor.sizeof tensor<30x2x96xf32> : index
  // CHECK: %[[RESHAPE:.+]] = stream.tensor.clone %[[INPUT]] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<30x2x96xf32> in !stream.resource<*>{%[[RESULT_SIZE]]}
  %0 = flow.tensor.reshape %input : tensor<5x24x48xf32> -> tensor<30x2x96xf32>
  // CHECK: %[[RESULT:.+]] = stream.tensor.clone %[[RESHAPE]] : tensor<30x2x96xf32> in !stream.resource<*>{%[[RESULT_SIZE]]} -> tensor<30x2x96xf32> in !stream.resource<*>{%[[RESULT_SIZE]]}
  %1 = flow.tensor.clone %0 : tensor<30x2x96xf32>
  // CHECK: return %[[RESULT]], %[[RESULT_SIZE]] : !stream.resource<*>, index
  return %1 : tensor<30x2x96xf32>
}

// -----

// CHECK-LABEL: @tensorReshapeWithMultipleUses
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index)
func.func @tensorReshapeWithMultipleUses(%input: tensor<5x24x48xf32>)
    -> (tensor<60x2x48xf32>, tensor<30x2x96xf32>) {
  // CHECK: %[[T0:.+]] = stream.tensor.clone %[[INPUT]] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]}
  %1 = flow.tensor.clone %input : tensor<5x24x48xf32>
  // CHECK: %[[T1_SIZE:.+]] = stream.tensor.sizeof tensor<60x2x48xf32> : index
  // CHECK: %[[T1:.+]] = stream.tensor.clone %[[INPUT]] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<60x2x48xf32> in !stream.resource<*>{%[[T1_SIZE]]}
  %2 = flow.tensor.reshape %input : tensor<5x24x48xf32> -> tensor<60x2x48xf32>
  // CHECK: %[[T2:.+]] = stream.tensor.clone %[[T1]] : tensor<60x2x48xf32> in !stream.resource<*>{%[[T1_SIZE]]} -> tensor<60x2x48xf32> in !stream.resource<*>{%[[T1_SIZE]]}
  %3 = flow.tensor.clone %2 : tensor<60x2x48xf32>
  // CHECK: %[[T3_SIZE:.+]] = stream.tensor.sizeof tensor<30x2x96xf32> : index
  // CHECK: %[[T3:.+]] = stream.tensor.clone %[[T0]] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<30x2x96xf32> in !stream.resource<*>{%[[T3_SIZE]]}
  %4 = flow.tensor.reshape %1 : tensor<5x24x48xf32> -> tensor<30x2x96xf32>
  // CHECK: return %[[T2]], %[[T1_SIZE]], %[[T3]], %[[T3_SIZE]] : !stream.resource<*>, index, !stream.resource<*>, index
  return %3, %4 : tensor<60x2x48xf32>, tensor<30x2x96xf32>
}

// -----

// CHECK-LABEL: @tensorAlloca
//  CHECK-SAME: (%[[DIM0:.+]]: index)
func.func @tensorAlloca(%dim0: index) -> tensor<?x0xf32> {
  // CHECK: %[[ALLOCA_SIZE:.+]] = stream.tensor.sizeof tensor<?x0xf32>{%[[DIM0]]}
  // CHECK: %[[ALLOCA:.+]] = stream.async.alloca : !stream.resource<*>{%[[ALLOCA_SIZE]]}
  %0 = flow.tensor.alloca : tensor<?x0xf32>{%dim0}
  // CHECK: return %[[ALLOCA]]
  return %0 : tensor<?x0xf32>
}

// -----

// CHECK-LABEL: @tensorEmpty
//  CHECK-SAME: (%[[DIM0:.+]]: index)
func.func @tensorEmpty(%dim0: index) -> tensor<?x0xf32> {
  // CHECK: %[[EMPTY_SIZE:.+]] = stream.tensor.sizeof tensor<?x0xf32>{%[[DIM0]]}
  // CHECK: %[[EMPTY:.+]] = stream.tensor.empty : tensor<?x0xf32>{%[[DIM0]]} in !stream.resource<*>{%[[EMPTY_SIZE]]}
  %0 = flow.tensor.empty : tensor<?x0xf32>{%dim0}
  // CHECK: return %[[EMPTY]]
  return %0 : tensor<?x0xf32>
}

// -----

// CHECK-LABEL: @tensorSplat
//  CHECK-SAME: (%[[VALUE:.+]]: i8, %[[DIM0:.+]]: index)
func.func @tensorSplat(%value: i8, %dim0: index) -> tensor<?x128xi8> {
  // CHECK: %[[T_SIZE:.+]] = stream.tensor.sizeof tensor<?x128xi8>{%[[DIM0]]} : index
  // CHECK: %[[T:.+]] = stream.tensor.splat %[[VALUE]] : i8 -> tensor<?x128xi8>{%[[DIM0]]} in !stream.resource<*>{%[[T_SIZE]]}
  %0 = flow.tensor.splat %value : tensor<?x128xi8>{%dim0}
  // CHECK: return %[[T]], %[[T_SIZE]]
  return %0 : tensor<?x128xi8>
}

// -----

// CHECK-LABEL: @tensorSlice
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index)
func.func @tensorSlice(%input : tensor<5x24x48xf32>) -> tensor<3x24x48xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c24 = arith.constant 24 : index
  %c48 = arith.constant 48 : index
  // CHECK: %[[T_SIZE:.+]] = stream.tensor.sizeof tensor<3x24x48xf32> : index
  // CHECK: %[[T:.+]] = stream.tensor.slice %[[INPUT]][%c2, %c0, %c0 for %c3, %c24, %c48] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<3x24x48xf32> in !stream.resource<*>{%[[T_SIZE]]}
  %0 = flow.tensor.slice %input[%c2, %c0, %c0 for %c3, %c24, %c48] : tensor<5x24x48xf32> -> tensor<3x24x48xf32>
  // CHECK: return %[[T]], %[[T_SIZE]] : !stream.resource<*>, index
  return %0 : tensor<3x24x48xf32>
}

// -----

// CHECK-LABEL: @tensorUpdate
//  CHECK-SAME: (%[[UPDATE:.+]]: !stream.resource<*>, %[[UPDATE_SIZE:.+]]: index, %[[TARGET:.+]]: !stream.resource<*>, %[[TARGET_SIZE:.+]]: index)
func.func @tensorUpdate(%update : tensor<1x1x10xf32>, %target : tensor<5x1x10xf32>) -> tensor<5x1x10xf32> {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[T:.+]] = stream.tensor.update %[[UPDATE]], %[[TARGET]][%c4, %c1, %c1] : tensor<1x1x10xf32> in !stream.resource<*>{%[[UPDATE_SIZE]]} -> tensor<5x1x10xf32> in %[[TARGET]] as !stream.resource<*>{%[[TARGET_SIZE]]}
  %0 = flow.tensor.update %update, %target[%c4, %c1, %c1] : tensor<1x1x10xf32> -> %target as tensor<5x1x10xf32>
  // CHECK: return %[[T]], %[[TARGET_SIZE]] : !stream.resource<*>, index
  return %0 : tensor<5x1x10xf32>
}

// -----

// CHECK-LABEL: @tensorLoad
//  CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<*>, %[[SOURCE_SIZE:.+]]: index)
func.func @tensorLoad(%source : tensor<2x3xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[T0:.+]] = stream.async.transfer from(#hal.affinity.queue<[0, 1]>) %[[SOURCE]] :
  // CHECK-SAME:           !stream.resource<*>{%[[SOURCE_SIZE]]} -> !stream.resource<staging>{%[[SOURCE_SIZE]]}
  // CHECK: %[[T1:.+]] = stream.tensor.load %[[T0]][%c0, %c1] : tensor<2x3xi32> in !stream.resource<staging>{%[[SOURCE_SIZE]]} -> i32
  %0 = flow.tensor.load %source[%c0, %c1] : tensor<2x3xi32> attributes {
    stream.affinity = #hal.affinity.queue<[0, 1]>
  }
  // CHECK: return %[[T1]]
  return %0 : i32
}

// -----

// CHECK-LABEL: @tensorStore
//  CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<*>, %[[TARGET_SIZE:.+]]: index)
func.func @tensorStore(%target : tensor<2x3xi32>) -> tensor<2x3xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : i32
  // CHECK: %[[T0:.+]] = stream.async.transfer from(#hal.affinity.queue<[0, 1]>) %[[TARGET]] :
  // CHECK-SAME:           !stream.resource<*>{%[[TARGET_SIZE]]} -> !stream.resource<staging>{%[[TARGET_SIZE]]}
  // CHECK: %[[T1:.+]] = stream.tensor.store %c9_i32, %[[T0]][%c0, %c1] :
  // CHECK-SAME:           i32 -> tensor<2x3xi32> in %[[T0]] as !stream.resource<staging>{%[[TARGET_SIZE]]}
  // CHECK: %[[T2:.+]] = stream.async.transfer %[[T1]] :
  // CHECK-SAME:           !stream.resource<staging>{%[[TARGET_SIZE]]} -> to(#hal.affinity.queue<[0, 1]>) !stream.resource<*>{%[[TARGET_SIZE]]}
  %0 = flow.tensor.store %c9, %target[%c0, %c1] : tensor<2x3xi32> attributes {
    stream.affinity = #hal.affinity.queue<[0, 1]>
  }
  // CHECK: return %[[T2]]
  return %0 : tensor<2x3xi32>
}
