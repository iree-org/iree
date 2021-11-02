// RUN: iree-opt -split-input-file -iree-stream-conversion %s | IreeFileCheck %s

// CHECK-LABEL: @tensorReshapePassThrough
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index)
func @tensorReshapePassThrough(%input: tensor<5x24x48xf32>) -> tensor<30x2x96xf32> {
  // CHECK: %[[T:.+]] = stream.async.transfer %arg0 : !stream.resource<*>{%[[INPUT_SIZE]]} -> !stream.resource<*>{%[[INPUT_SIZE]]}
  // CHECK: %[[RESULT_SIZE:.+]] = stream.tensor.sizeof tensor<30x2x96xf32> : index
  // CHECK: %[[RESULT:.+]] = stream.tensor.clone %[[T]] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<30x2x96xf32> in !stream.resource<*>{%[[RESULT_SIZE]]}
  %0 = flow.tensor.reshape %input : tensor<5x24x48xf32> -> tensor<30x2x96xf32>
  // CHECK: return %[[RESULT]], %[[RESULT_SIZE]] : !stream.resource<*>, index
  return %0 : tensor<30x2x96xf32>
}

// -----

// CHECK-LABEL: @tensorReshapeWithSingleUse
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index)
func @tensorReshapeWithSingleUse(%input: tensor<5x24x48xf32>) -> tensor<30x2x96xf32> {
  // CHECK: %[[T:.+]] = stream.async.transfer %arg0 : !stream.resource<*>{%[[INPUT_SIZE]]} -> !stream.resource<*>{%[[INPUT_SIZE]]}
  // CHECK: %[[RESULT_SIZE:.+]] = stream.tensor.sizeof tensor<30x2x96xf32> : index
  // CHECK: %[[RESHAPE:.+]] = stream.tensor.clone %[[T]] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<30x2x96xf32> in !stream.resource<*>{%[[RESULT_SIZE]]}
  %0 = flow.tensor.reshape %input : tensor<5x24x48xf32> -> tensor<30x2x96xf32>
  // CHECK: %[[RESULT:.+]] = stream.tensor.clone %[[RESHAPE]] : tensor<30x2x96xf32> in !stream.resource<*>{%[[RESULT_SIZE]]} -> tensor<30x2x96xf32> in !stream.resource<*>{%[[RESULT_SIZE]]}
  %1 = flow.tensor.clone %0 : tensor<30x2x96xf32>
  // CHECK: return %[[RESULT]], %[[RESULT_SIZE]] : !stream.resource<*>, index
  return %1 : tensor<30x2x96xf32>
}

// -----

// CHECK-LABEL: @tensorReshapeWithMultipleUses
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index)
func @tensorReshapeWithMultipleUses(%input: tensor<5x24x48xf32>)
    -> (tensor<60x2x48xf32>, tensor<30x2x96xf32>) {
  // CHECK: %[[T_INPUT:.+]] = stream.async.transfer %[[INPUT]] : !stream.resource<*>{%[[INPUT_SIZE]]} -> !stream.resource<*>{%[[INPUT_SIZE]]}
  // CHECK: %[[T0:.+]] = stream.tensor.clone %[[T_INPUT]] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]}
  %1 = flow.tensor.clone %input : tensor<5x24x48xf32>
  // CHECK: %[[T1_SIZE:.+]] = stream.tensor.sizeof tensor<60x2x48xf32> : index
  // CHECK: %[[T1:.+]] = stream.tensor.clone %[[T_INPUT]] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<60x2x48xf32> in !stream.resource<*>{%[[T1_SIZE]]}
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

// CHECK-LABEL: @tensorSplat
//  CHECK-SAME: (%[[VALUE:.+]]: i8, %[[DIM0:.+]]: index)
func @tensorSplat(%value: i8, %dim0: index) -> tensor<?x128xi8> {
  // CHECK: %[[T_SIZE:.+]] = stream.tensor.sizeof tensor<?x128xi8>{%[[DIM0]]} : index
  // CHECK: %[[T:.+]] = stream.tensor.splat %[[VALUE]] : i8 -> tensor<?x128xi8>{%[[DIM0]]} in !stream.resource<*>{%[[T_SIZE]]}
  %0 = flow.tensor.splat %value : tensor<?x128xi8>{%dim0}
  // CHECK: return %[[T]], %[[T_SIZE]]
  return %0 : tensor<?x128xi8>
}

// -----

// CHECK-LABEL: @tensorSlice
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index)
func @tensorSlice(%input : tensor<5x24x48xf32>) -> tensor<3x24x48xf32> {
  // CHECK: %[[T0:.+]] = stream.async.transfer %[[INPUT]] : !stream.resource<*>{%[[INPUT_SIZE]]} -> !stream.resource<*>{%[[INPUT_SIZE]]}
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c24 = arith.constant 24 : index
  %c48 = arith.constant 48 : index
  // CHECK: %[[T1_SIZE:.+]] = stream.tensor.sizeof tensor<3x24x48xf32> : index
  // CHECK: %[[T1:.+]] = stream.tensor.slice %[[T0]][%c2, %c0, %c0 for %c3, %c24, %c48] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<3x24x48xf32> in !stream.resource<*>{%[[T1_SIZE]]}
  %0 = flow.tensor.slice %input[%c2, %c0, %c0 for %c3, %c24, %c48] : tensor<5x24x48xf32> -> tensor<3x24x48xf32>
  // CHECK: return %[[T1]], %[[T1_SIZE]] : !stream.resource<*>, index
  return %0 : tensor<3x24x48xf32>
}

// -----

// CHECK-LABEL: @tensorUpdate
//  CHECK-SAME: (%[[UPDATE:.+]]: !stream.resource<*>, %[[UPDATE_SIZE:.+]]: index, %[[TARGET:.+]]: !stream.resource<*>, %[[TARGET_SIZE:.+]]: index)
func @tensorUpdate(%update : tensor<1x1x10xf32>, %target : tensor<5x1x10xf32>) -> tensor<5x1x10xf32> {
  // CHECK: %[[T0:.+]] = stream.async.transfer %[[UPDATE]] : !stream.resource<*>{%[[UPDATE_SIZE]]} -> !stream.resource<*>{%[[UPDATE_SIZE]]}
  // CHECK: %[[T1:.+]] = stream.async.transfer %[[TARGET]] : !stream.resource<*>{%[[TARGET_SIZE]]} -> !stream.resource<*>{%[[TARGET_SIZE]]}
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[T2:.+]] = stream.tensor.update %[[T0]], %[[T1]][%c4, %c1, %c1] : tensor<1x1x10xf32> in !stream.resource<*>{%[[UPDATE_SIZE]]} -> tensor<5x1x10xf32> in %1 as !stream.resource<*>{%[[TARGET_SIZE]]}
  %0 = flow.tensor.update %update, %target[%c4, %c1, %c1] : tensor<1x1x10xf32> -> %target as tensor<5x1x10xf32>
  // CHECK: return %[[T2]], %[[TARGET_SIZE]] : !stream.resource<*>, index
  return %0 : tensor<5x1x10xf32>
}

// -----

// CHECK-LABEL: @tensorLoad
//  CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<*>, %[[SOURCE_SIZE:.+]]: index)
func @tensorLoad(%source : tensor<2x3xi32>) -> i32 {
  // CHECK: %[[T0:.+]] = stream.async.transfer %[[SOURCE]] : !stream.resource<*>{%[[SOURCE_SIZE]]} -> !stream.resource<*>{%[[SOURCE_SIZE]]}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[T1:.+]] = stream.async.transfer %[[T0]] : !stream.resource<*>{%[[SOURCE_SIZE]]} -> !stream.resource<staging>{%[[SOURCE_SIZE]]}
  // CHECK: %[[T2:.+]] = stream.tensor.load %[[T1]][%c0, %c1] : tensor<2x3xi32> in !stream.resource<staging>{%[[SOURCE_SIZE]]} -> i32
  %0 = flow.tensor.load %source[%c0, %c1] : tensor<2x3xi32>
  // CHECK: return %[[T2]]
  return %0 : i32
}

// -----

// CHECK-LABEL: @tensorStore
//  CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<*>, %[[TARGET_SIZE:.+]]: index)
func @tensorStore(%target : tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK: %[[T0:.+]] = stream.async.transfer %[[TARGET]] : !stream.resource<*>{%[[TARGET_SIZE]]} -> !stream.resource<*>{%[[TARGET_SIZE]]}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : i32
  // CHECK: %[[T1:.+]] = stream.async.transfer %[[T0]] : !stream.resource<*>{%[[TARGET_SIZE]]} -> !stream.resource<staging>{%[[TARGET_SIZE]]}
  // CHECK: %[[T2:.+]] = stream.tensor.store %c9_i32, %[[T1]][%c0, %c1] : i32 -> tensor<2x3xi32> in %1 as !stream.resource<staging>{%[[TARGET_SIZE]]}
  // CHECK: %[[T3:.+]] = stream.async.transfer %[[T2]] : !stream.resource<staging>{%[[TARGET_SIZE]]} -> !stream.resource<*>{%[[TARGET_SIZE]]}
  %0 = flow.tensor.store %c9, %target[%c0, %c1] : tensor<2x3xi32>
  // CHECK: return %[[T3]]
  return %0 : tensor<2x3xi32>
}
