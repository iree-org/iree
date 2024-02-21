// RUN: iree-opt --split-input-file --canonicalize %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @foldSwitchI32Nop
util.func public @foldSwitchI32Nop(%arg0 : index) -> i32 {
  // CHECK: %[[DEFAULT:.+]] = arith.constant 5
  %c5 = arith.constant 5 : i32
  %0 = util.switch i32 from [] at %arg0 else %c5 : i32
  // CHECK: util.return %[[DEFAULT]] : i32
  util.return %0 : i32
}

// -----

// CHECK-LABEL: @foldSwitchI32Identical
util.func public @foldSwitchI32Identical(%arg0 : index) -> i32 {
  // CHECK: %[[C100:.+]] = arith.constant 100
  %c100 = arith.constant 100 : i32
  %0 = util.switch i32 from [%c100, %c100, %c100] at %arg0 else %c100 : i32
  // CHECK: util.return %[[C100]] : i32
  util.return %0 : i32
}

// -----

// CHECK-LABEL: @foldSwitchI32ConstantIndex
util.func public @foldSwitchI32ConstantIndex() -> (i32, i32, i32, i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK-DAG: %[[C100:.+]] = arith.constant 100
  %c100 = arith.constant 100 : i32
  // CHECK-DAG: %[[C200:.+]] = arith.constant 200
  %c200 = arith.constant 200 : i32
  // CHECK-DAG: %[[C300:.+]] = arith.constant 300
  %c300 = arith.constant 300 : i32
  // CHECK-DAG: %[[C400:.+]] = arith.constant 400
  %c400 = arith.constant 400 : i32
  %0 = util.switch i32 from [%c100, %c200, %c300] at %c0 else %c400 : i32
  %1 = util.switch i32 from [%c100, %c200, %c300] at %c1 else %c400 : i32
  %2 = util.switch i32 from [%c100, %c200, %c300] at %c2 else %c400 : i32
  %3 = util.switch i32 from [%c100, %c200, %c300] at %c3 else %c400 : i32
  // CHECK: util.return %[[C100]], %[[C200]], %[[C300]], %[[C400]] : i32, i32, i32, i32
  util.return %0, %1, %2, %3 : i32, i32, i32, i32
}

// -----

// CHECK-LABEL: @foldCastSameType
// CHECK-SAME: (%[[SOURCE:.+]]: !util.buffer)
util.func public @foldCastSameType(%source: !util.buffer) -> !util.buffer {
  // CHECK-NOT: util.cast
  %0 = util.cast %source : !util.buffer to !util.buffer
  // CHECK: util.return %[[SOURCE]]
  util.return %0 : !util.buffer
}

// -----

// CHECK-LABEL: @foldChainedCast
// CHECK-SAME: (%[[SOURCE:.+]]: !util.buffer)
util.func public @foldChainedCast(%source: !util.buffer) -> !util.buffer {
  // CHECK-NOT: util.cast
  %0 = util.cast %source : !util.buffer to !util.object
  // CHECK-NOT: util.cast
  %1 = util.cast %0 : !util.object to !util.buffer
  // CHECK: util.return %[[SOURCE]]
  util.return %1 : !util.buffer
}

// -----

// CHECK-LABEL: @foldCastIntoNullOp
util.func public @foldCastIntoNullOp() -> !util.buffer {
  // CHECK: %[[NULL:.+]] = util.null : !util.buffer
  %0 = util.null : !util.object
  // CHECK-NOT: util.cast
  %1 = util.cast %0 : !util.object to !util.buffer
  // CHECK: util.return %[[NULL]]
  util.return %1 : !util.buffer
}
