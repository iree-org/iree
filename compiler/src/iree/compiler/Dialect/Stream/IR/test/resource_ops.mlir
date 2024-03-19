// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @resourceAlloc
util.func private @resourceAlloc(%arg0: index) -> !stream.resource<*> {
  // CHECK: = stream.resource.alloc uninitialized : !stream.resource<*>{%arg0}
  %0 = stream.resource.alloc uninitialized : !stream.resource<*>{%arg0}
  util.return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @resourceAlloca
util.func private @resourceAlloca(%arg0: index, %await_timepoint: !stream.timepoint) -> (!stream.resource<staging>, !stream.timepoint, !stream.resource<staging>, !stream.timepoint) {
  // CHECK: = stream.resource.alloca uninitialized : !stream.resource<staging>{%arg0} => !stream.timepoint
  %0:2 = stream.resource.alloca uninitialized : !stream.resource<staging>{%arg0} => !stream.timepoint
  // CHECK: = stream.resource.alloca uninitialized await(%arg1) => !stream.resource<staging>{%arg0} => !stream.timepoint
  %1:2 = stream.resource.alloca uninitialized await(%await_timepoint) => !stream.resource<staging>{%arg0} => !stream.timepoint
  util.return %0#0, %0#1, %1#0, %1#1 : !stream.resource<staging>, !stream.timepoint, !stream.resource<staging>, !stream.timepoint
}

// -----

// CHECK-LABEL: @resourceDealloca
util.func private @resourceDealloca(%arg0: index, %arg1: !stream.resource<staging>, %arg2: !stream.timepoint) {
  // CHECK: = stream.resource.dealloca %arg1 : !stream.resource<staging>{%arg0} => !stream.timepoint
  stream.resource.dealloca %arg1 : !stream.resource<staging>{%arg0} => !stream.timepoint
  // CHECK: = stream.resource.dealloca await(%arg2) => %arg1 : !stream.resource<staging>{%arg0} => !stream.timepoint
  stream.resource.dealloca await(%arg2) => %arg1 : !stream.resource<staging>{%arg0} => !stream.timepoint
  util.return
}

// -----

// CHECK-LABEL: @resourceSize
util.func private @resourceSize(%arg0: !stream.resource<*>) -> index {
  // CHECK: = stream.resource.size %arg0 : !stream.resource<*>
  %0 = stream.resource.size %arg0 : !stream.resource<*>
  util.return %0 : index
}

// -----

// CHECK-LABEL: @resourceTryMap
util.func private @resourceTryMap(%arg0: !util.buffer) -> (i1, !stream.resource<constant>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: = stream.resource.try_map %arg0[%c0] : !util.buffer -> i1, !stream.resource<constant>{%c128}
  %0:2 = stream.resource.try_map %arg0[%c0] : !util.buffer -> i1, !stream.resource<constant>{%c128}
  util.return %0#0, %0#1 : i1, !stream.resource<constant>
}

// -----

// CHECK-LABEL: @resourceLoad
util.func private @resourceLoad(%arg0: !stream.resource<staging>, %arg1: index) -> i32 {
  %c0 = arith.constant 0 : index
  // CHECK: = stream.resource.load %arg0[%c0] : !stream.resource<staging>{%arg1} -> i32
  %0 = stream.resource.load %arg0[%c0] : !stream.resource<staging>{%arg1} -> i32
  util.return %0 : i32
}

// -----

// CHECK-LABEL: @resourceStore
util.func private @resourceStore(%arg0: !stream.resource<staging>, %arg1: index) {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.resource.store %c123_i32, %arg0[%c0] : i32 -> !stream.resource<staging>{%arg1}
  stream.resource.store %c123_i32, %arg0[%c0] : i32 -> !stream.resource<staging>{%arg1}
  util.return
}

// -----

// CHECK-LABEL: @resourcePack
util.func private @resourcePack(%arg0: index, %arg1: index) -> (index, index, index) {
  %c128 = arith.constant 128 : index
  //      CHECK: stream.resource.pack offset(%c128) slices({
  // CHECK-NEXT:   [0, 9] = %arg0,
  // CHECK-NEXT:   [3, 8] = %arg1
  // CHECK-NEXT:  })
  %0:3 = stream.resource.pack offset(%c128) slices({
    [0, 9] = %arg0,
    [3, 8] = %arg1,
  }) : index
  util.return %0#0, %0#1, %0#2 : index, index, index
}

// -----

// CHECK-LABEL: @resourceConstants
util.func private @resourceConstants() -> (!stream.resource<constant>, !stream.resource<constant>, !stream.timepoint) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  //      CHECK: = stream.resource.constants :
  // CHECK-NEXT:   !stream.resource<constant>{%c4} = dense<100> : tensor<1xi32>,
  // CHECK-NEXT:   !stream.resource<constant>{%c8} = dense<[101, 102]> : tensor<2xi32>
  // CHECK-NEXT:   => !stream.timepoint
  %0:3 = stream.resource.constants :
    !stream.resource<constant>{%c4} = dense<100> : tensor<1xi32>,
    !stream.resource<constant>{%c8} = dense<[101, 102]> : tensor<2xi32>
    => !stream.timepoint
  util.return %0#0, %0#1, %0#2 : !stream.resource<constant>, !stream.resource<constant>, !stream.timepoint
}

// -----

// CHECK-LABEL: @resourceSubview
util.func private @resourceSubview(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK: = stream.resource.subview %arg0[%c128] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c256}
  %0 = stream.resource.subview %arg0[%c128] : !stream.resource<*>{%arg1} -> !stream.resource<*>{%c256}
  util.return %0 : !stream.resource<*>
}
