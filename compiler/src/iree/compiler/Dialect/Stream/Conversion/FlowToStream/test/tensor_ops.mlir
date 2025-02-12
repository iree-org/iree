// RUN: iree-opt --split-input-file --iree-stream-conversion %s | FileCheck %s

// CHECK-LABEL: @tensorConstantStatic
util.func public @tensorConstantStatic() -> tensor<4x2xi32> {
  // CHECK-DAG: %[[CST:.+]] = stream.tensor.constant : tensor<4x2xi32> in !stream.resource<constant> = dense<2> : tensor<4x2xi32>
  // CHECK-DAG: %[[SIZE:.+]] = stream.resource.size %[[CST]] : !stream.resource<constant>
  // CHECK-DAG: %[[TRANSFER:.+]] = stream.async.transfer %[[CST]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %cst = flow.tensor.constant dense<2> : tensor<4x2xi32>
  // CHECK: util.return %[[TRANSFER]], %[[SIZE]]
  util.return %cst : tensor<4x2xi32>
}

// -----

// CHECK-LABEL: @tensorConstantParameter
util.func public @tensorConstantParameter() -> tensor<4x2xi32> {
  // CHECK-DAG: %[[CST:.+]] = stream.tensor.constant : tensor<4x2xi32> in !stream.resource<constant> = #stream.parameter.named<"scope"::"key"> : tensor<4x2xi32>
  // CHECK-DAG: %[[SIZE:.+]] = stream.resource.size %[[CST]] : !stream.resource<constant>
  // CHECK-DAG: %[[TRANSFER:.+]] = stream.async.transfer %[[CST]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %cst = flow.tensor.constant #flow.parameter.named<"scope"::"key"> : tensor<4x2xi32>
  // CHECK: util.return %[[TRANSFER]], %[[SIZE]]
  util.return %cst : tensor<4x2xi32>
}

// -----

// CHECK-LABEL: @tensorConstantDynamic
util.func public @tensorConstantDynamic() -> tensor<?x?xi32> {
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[D0:.+]] = util.optimization_barrier %[[C2]] : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[D1:.+]] = util.optimization_barrier %[[C4]] : index
  // CHECK-DAG: %[[CST:.+]] = stream.tensor.constant : tensor<?x?xi32>{%[[D0]], %[[D1]]} in !stream.resource<constant> = dense<2> : tensor<2x4xi32>
  // CHECK-DAG: %[[SIZE:.+]] = stream.resource.size %[[CST]] : !stream.resource<constant>
  // CHECK-DAG: %[[TRANSFER:.+]] = stream.async.transfer %[[CST]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %cst = flow.tensor.dynamic_constant dense<2> : tensor<2x4xi32> -> tensor<?x?xi32>
  // CHECK: util.return %[[TRANSFER]], %[[SIZE]]
  util.return %cst : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @tensorReshapePassThrough
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index)
util.func public @tensorReshapePassThrough(%input: tensor<5x24x48xf32>) -> tensor<30x2x96xf32> {
  // CHECK: %[[RESULT_SIZE:.+]] = stream.tensor.sizeof tensor<30x2x96xf32> : index
  // CHECK: %[[RESULT:.+]] = stream.tensor.clone %[[INPUT]] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<30x2x96xf32> in !stream.resource<*>{%[[RESULT_SIZE]]}
  %0 = flow.tensor.reshape %input : tensor<5x24x48xf32> -> tensor<30x2x96xf32>
  // CHECK: util.return %[[RESULT]], %[[RESULT_SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<30x2x96xf32>
}

// -----

// CHECK-LABEL: @tensorReshapeWithSingleUse
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index)
util.func public @tensorReshapeWithSingleUse(%input: tensor<5x24x48xf32>) -> tensor<30x2x96xf32> {
  // CHECK: %[[RESULT_SIZE:.+]] = stream.tensor.sizeof tensor<30x2x96xf32> : index
  // CHECK: %[[RESHAPE:.+]] = stream.tensor.clone %[[INPUT]] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<30x2x96xf32> in !stream.resource<*>{%[[RESULT_SIZE]]}
  %0 = flow.tensor.reshape %input : tensor<5x24x48xf32> -> tensor<30x2x96xf32>
  // CHECK: %[[RESULT:.+]] = stream.tensor.clone %[[RESHAPE]] : tensor<30x2x96xf32> in !stream.resource<*>{%[[RESULT_SIZE]]} -> tensor<30x2x96xf32> in !stream.resource<*>{%[[RESULT_SIZE]]}
  %1 = flow.tensor.clone %0 : tensor<30x2x96xf32>
  // CHECK: util.return %[[RESULT]], %[[RESULT_SIZE]] : !stream.resource<*>, index
  util.return %1 : tensor<30x2x96xf32>
}

// -----

// CHECK-LABEL: @tensorReshapeWithMultipleUses
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index)
util.func public @tensorReshapeWithMultipleUses(%input: tensor<5x24x48xf32>)
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
  // CHECK: util.return %[[T2]], %[[T1_SIZE]], %[[T3]], %[[T3_SIZE]] : !stream.resource<*>, index, !stream.resource<*>, index
  util.return %3, %4 : tensor<60x2x48xf32>, tensor<30x2x96xf32>
}

// -----

// CHECK-LABEL: @tensorBitCastWithSingleUse
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index)
util.func public @tensorBitCastWithSingleUse(%input: tensor<5x24x48xi8>) -> tensor<30x2x192xi4> {
  // CHECK: %[[RESULT_SIZE:.+]] = stream.tensor.sizeof tensor<30x2x192xi4> : index
  // CHECK: %[[BITCAST:.+]] = stream.tensor.clone %[[INPUT]] : tensor<5x24x48xi8> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<30x2x192xi4> in !stream.resource<*>{%[[RESULT_SIZE]]}
  %0 = flow.tensor.bitcast %input : tensor<5x24x48xi8> -> tensor<30x2x192xi4>
  // CHECK: %[[RESULT:.+]] = stream.tensor.clone %[[BITCAST]] : tensor<30x2x192xi4> in !stream.resource<*>{%[[RESULT_SIZE]]} -> tensor<30x2x192xi4> in !stream.resource<*>{%[[RESULT_SIZE]]}
  %1 = flow.tensor.clone %0 : tensor<30x2x192xi4>
  // CHECK: util.return %[[RESULT]], %[[RESULT_SIZE]] : !stream.resource<*>, index
  util.return %1 : tensor<30x2x192xi4>
}

// -----

// CHECK-LABEL: @tensorAlloca
//  CHECK-SAME: (%[[DIM0:.+]]: index)
util.func public @tensorAlloca(%dim0: index) -> tensor<?x0xf32> {
  // CHECK: %[[ALLOCA_SIZE:.+]] = stream.tensor.sizeof tensor<?x0xf32>{%[[DIM0]]}
  // CHECK: %[[ALLOCA:.+]] = stream.async.alloca : !stream.resource<*>{%[[ALLOCA_SIZE]]}
  %0 = flow.tensor.alloca : tensor<?x0xf32>{%dim0}
  // CHECK: util.return %[[ALLOCA]]
  util.return %0 : tensor<?x0xf32>
}

// -----

// CHECK-LABEL: @tensorEmpty
//  CHECK-SAME: (%[[DIM0:.+]]: index)
util.func public @tensorEmpty(%dim0: index) -> tensor<?x0xf32> {
  // CHECK: %[[EMPTY_SIZE:.+]] = stream.tensor.sizeof tensor<?x0xf32>{%[[DIM0]]}
  // CHECK: %[[EMPTY:.+]] = stream.tensor.empty : tensor<?x0xf32>{%[[DIM0]]} in !stream.resource<*>{%[[EMPTY_SIZE]]}
  %0 = flow.tensor.empty : tensor<?x0xf32>{%dim0}
  // CHECK: util.return %[[EMPTY]]
  util.return %0 : tensor<?x0xf32>
}

// -----

// CHECK-LABEL: @tensorSplat
//  CHECK-SAME: (%[[VALUE:.+]]: i8, %[[DIM0:.+]]: index)
util.func public @tensorSplat(%value: i8, %dim0: index) -> tensor<?x128xi8> {
  // CHECK: %[[T_SIZE:.+]] = stream.tensor.sizeof tensor<?x128xi8>{%[[DIM0]]} : index
  // CHECK: %[[T:.+]] = stream.tensor.splat %[[VALUE]] : i8 -> tensor<?x128xi8>{%[[DIM0]]} in !stream.resource<*>{%[[T_SIZE]]}
  %0 = flow.tensor.splat %value : tensor<?x128xi8>{%dim0}
  // CHECK: util.return %[[T]], %[[T_SIZE]]
  util.return %0 : tensor<?x128xi8>
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @tensorBarrierDispatch
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index, %[[DIM0:.+]]: index)
util.func public @tensorBarrierDispatch(%input: tensor<?x128xi8>, %dim0: index) -> tensor<?x128xi8> {
  // CHECK: %[[BARRIER:.+]] = stream.async.barrier on(#hal.device.affinity<@device>) %[[INPUT]] : !stream.resource<*>{%[[INPUT_SIZE]]}
  %barrier = flow.tensor.barrier %input : tensor<?x128xi8>{%dim0} on #hal.device.affinity<@device>
  // CHECK: %[[RESULT_SIZE:.+]] = stream.tensor.sizeof tensor<?x128xi8>{%[[DIM0]]} : index
  // CHECK: %[[RESULT:.+]] = stream.tensor.dispatch @ex::@entry(%[[BARRIER]])
  %0 = flow.dispatch @ex::@entry(%barrier) : (tensor<?x128xi8>{%dim0}) -> tensor<?x128xi8>{%dim0}
  // CHECK: util.return %[[RESULT]], %[[RESULT_SIZE]]
  util.return %0 : tensor<?x128xi8>
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @tensorTransfer
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index, %[[DIM0:.+]]: index)
util.func public @tensorTransfer(%input: tensor<?x128xi8>, %dim0: index) -> tensor<?x128xi8> {
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[INPUT]] : !stream.resource<*>{%[[INPUT_SIZE]]} -> to(#hal.device.affinity<@device>) !stream.resource<*>{%[[INPUT_SIZE]]}
  %transfer = flow.tensor.transfer %input : tensor<?x128xi8>{%dim0} to #hal.device.affinity<@device>
  // CHECK: util.return %[[TRANSFER]], %[[INPUT_SIZE]]
  util.return %transfer : tensor<?x128xi8>
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @tensorBarrier
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index, %[[DIM0:.+]]: index)
util.func public @tensorBarrier(%input: tensor<?x128xi8>, %dim0: index) -> tensor<?x128xi8> {
  // CHECK: %[[TRANSFER:.+]] = stream.async.barrier on(#hal.device.affinity<@device>) %[[INPUT]] : !stream.resource<*>{%[[INPUT_SIZE]]}
  %transfer = flow.tensor.barrier %input : tensor<?x128xi8>{%dim0} on #hal.device.affinity<@device>
  // CHECK: util.return %[[TRANSFER]], %[[INPUT_SIZE]]
  util.return %transfer : tensor<?x128xi8>
}

// -----

// CHECK-LABEL: @tensorSlice
//  CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[INPUT_SIZE:.+]]: index)
util.func public @tensorSlice(%input : tensor<5x24x48xf32>) -> tensor<3x24x48xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c24 = arith.constant 24 : index
  %c48 = arith.constant 48 : index
  // CHECK: %[[T_SIZE:.+]] = stream.tensor.sizeof tensor<3x24x48xf32> : index
  // CHECK: %[[T:.+]] = stream.tensor.slice %[[INPUT]][%c2, %c0, %c0 for %c3, %c24, %c48] : tensor<5x24x48xf32> in !stream.resource<*>{%[[INPUT_SIZE]]} -> tensor<3x24x48xf32> in !stream.resource<*>{%[[T_SIZE]]}
  %0 = flow.tensor.slice %input[%c2, %c0, %c0 for %c3, %c24, %c48] : tensor<5x24x48xf32> -> tensor<3x24x48xf32>
  // CHECK: util.return %[[T]], %[[T_SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<3x24x48xf32>
}

// -----

// CHECK-LABEL: @tensorUpdate
//  CHECK-SAME: (%[[UPDATE:.+]]: !stream.resource<*>, %[[UPDATE_SIZE:.+]]: index, %[[TARGET:.+]]: !stream.resource<*>, %[[TARGET_SIZE:.+]]: index)
util.func public @tensorUpdate(%update : tensor<1x1x10xf32>, %target : tensor<5x1x10xf32>) -> tensor<5x1x10xf32> {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[T:.+]] = stream.tensor.update %[[UPDATE]], %[[TARGET]][%c4, %c1, %c1] : tensor<1x1x10xf32> in !stream.resource<*>{%[[UPDATE_SIZE]]} -> tensor<5x1x10xf32> in %[[TARGET]] as !stream.resource<*>{%[[TARGET_SIZE]]}
  %0 = flow.tensor.update %update, %target[%c4, %c1, %c1] : tensor<1x1x10xf32> -> %target as tensor<5x1x10xf32>
  // CHECK: util.return %[[T]], %[[TARGET_SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<5x1x10xf32>
}

// -----

// CHECK-LABEL: @tensorLoad
//  CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<*>, %[[SOURCE_SIZE:.+]]: index)
util.func public @tensorLoad(%source : tensor<2x3xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[SLICE_SIZE:.+]] = stream.tensor.sizeof tensor<1x1xi32>
  // CHECK: %[[SLICE:.+]] = stream.tensor.slice %[[SOURCE]][%c0, %c1 for %c1, %c1] : tensor<2x3xi32> in !stream.resource<*>{%[[SOURCE_SIZE]]} -> tensor<1x1xi32> in !stream.resource<*>{%[[SLICE_SIZE]]}
  // CHECK: %[[STAGING:.+]] = stream.async.transfer
  // CHECK-SAME: %[[SLICE]] : !stream.resource<*>{%[[SLICE_SIZE]]}
  // CHECK-SAME: !stream.resource<staging>{%[[SLICE_SIZE]]}
  // CHECK: %[[VALUE:.+]] = stream.tensor.load %[[STAGING]][%c0, %c0] : tensor<1x1xi32> in !stream.resource<staging>{%[[SLICE_SIZE]]} -> i32
  %0 = flow.tensor.load %source[%c0, %c1] : tensor<2x3xi32>
  // CHECK: util.return %[[VALUE]]
  util.return %0 : i32
}

// -----

// CHECK-LABEL: @tensorLoadScalar
//  CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<*>, %[[SOURCE_SIZE:.+]]: index)
util.func public @tensorLoadScalar(%source : tensor<i32>) -> i32 {
  // CHECK: %[[STAGING:.+]] = stream.async.transfer
  // CHECK-SAME: %[[SOURCE]] : !stream.resource<*>{%[[SOURCE_SIZE]]}
  // CHECK-SAME: !stream.resource<staging>{%[[SOURCE_SIZE]]}
  // CHECK: %[[VALUE:.+]] = stream.tensor.load %[[STAGING]] : tensor<i32> in !stream.resource<staging>{%[[SOURCE_SIZE]]} -> i32
  %0 = flow.tensor.load %source : tensor<i32>
  // CHECK: util.return %[[VALUE]]
  util.return %0 : i32
}

// -----

// CHECK-LABEL: @tensorStore
//  CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<*>, %[[TARGET_SIZE:.+]]: index)
util.func public @tensorStore(%target : tensor<2x3xi32>) -> tensor<2x3xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[VALUE:.+]] = arith.constant 9
  %value = arith.constant 9 : i32
  // CHECK: %[[FILL:.+]] = stream.tensor.fill %[[VALUE]], %[[TARGET]][%c0, %c1 for %c1, %c1] : i32 -> tensor<2x3xi32> in %[[TARGET]] as !stream.resource<*>{%[[TARGET_SIZE]]}
  %0 = flow.tensor.store %value, %target[%c0, %c1] : tensor<2x3xi32>
  // CHECK: util.return %[[FILL]]
  util.return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @tensorStoreScalar
//  CHECK-SAME: (%[[TARGET:.+]]: !stream.resource<*>, %[[TARGET_SIZE:.+]]: index)
util.func public @tensorStoreScalar(%target : tensor<i32>) -> tensor<i32> {
  // CHECK: %[[VALUE:.+]] = arith.constant 9
  %value = arith.constant 9 : i32
  // CHECK: %[[FILL:.+]] = stream.tensor.fill %[[VALUE]], %[[TARGET]] : i32 -> tensor<i32> in %[[TARGET]] as !stream.resource<*>{%[[TARGET_SIZE]]}
  %0 = flow.tensor.store %value, %target : tensor<i32>
  // CHECK: util.return %[[FILL]]
  util.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: @tensorTrace
//  CHECK-SAME: (%[[TENSOR0:.+]]: !stream.resource<*>, %[[TENSOR0_SIZE:.+]]: index, %[[TENSOR1:.+]]: !stream.resource<*>, %[[TENSOR1_SIZE:.+]]: index, %[[TENSOR1_DIM0:.+]]: index, %[[TENSOR1_DIM2:.+]]: index)
util.func public @tensorTrace(%tensor0: tensor<5xf32>, %tensor1: tensor<?x3x?xi32>, %tensor1_dim0: index, %tensor1_dim2: index) {
  // CHECK-DAG: %[[TENSOR0_STAGED:.+]] = stream.async.transfer %[[TENSOR0]] : !stream.resource<*>{%[[TENSOR0_SIZE]]} -> !stream.resource<staging>{%[[TENSOR0_SIZE]]}
  // CHECK-DAG: %[[TENSOR1_STAGED:.+]] = stream.async.transfer %[[TENSOR1]] : !stream.resource<*>{%[[TENSOR1_SIZE]]} -> !stream.resource<staging>{%[[TENSOR1_SIZE]]}
  //      CHECK: stream.tensor.trace "FOOBAR" = [
  // CHECK-NEXT:   %[[TENSOR0_STAGED]] : tensor<5xf32> in !stream.resource<staging>{%[[TENSOR0_SIZE]]},
  // CHECK-NEXT:   %[[TENSOR1_STAGED]] : tensor<?x3x?xi32>{%[[TENSOR1_DIM0]], %[[TENSOR1_DIM2]]} in !stream.resource<staging>{%[[TENSOR1_SIZE]]}
  // CHECK-NEXT: ]
  flow.tensor.trace "FOOBAR" = [
    %tensor0 : tensor<5xf32>,
    %tensor1 : tensor<?x3x?xi32>{%tensor1_dim0, %tensor1_dim2}
  ]
  util.return
}
