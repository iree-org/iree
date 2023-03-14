// RUN: iree-opt --split-input-file --iree-vm-conversion --cse --iree-vm-target-index-bits=32 %s | FileCheck %s

// CHECK-LABEL: @utilAlign
func.func @utilAlign(%arg0 : index, %arg1: index) ->  (index) {
  %result = util.align %arg0, %arg1 : index
  // CHECK-DAG: %c1 = vm.const.i32 1
  // CHECK-DAG: %0 = vm.sub.i32 %arg1, %c1 : i32
  // CHECK-DAG: %1 = vm.add.i32 %arg0, %0 : i32
  // CHECK-DAG: %2 = vm.not.i32 %0 : i32
  // CHECK-DAG: %3 = vm.and.i32 %1, %2 : i32
  // CHECK: vm.return %3 : i32
  return %result : index
}

// -----

// CHECK-LABEL: @utilAlignInt32
func.func @utilAlignInt32(%arg0 : i32, %arg1: i32) ->  (i32) {
  %result = util.align %arg0, %arg1 : i32
  // CHECK-DAG: %c1 = vm.const.i32 1
  // CHECK-DAG: %0 = vm.sub.i32 %arg1, %c1 : i32
  // CHECK-DAG: %1 = vm.add.i32 %arg0, %0 : i32
  // CHECK-DAG: %2 = vm.not.i32 %0 : i32
  // CHECK-DAG: %3 = vm.and.i32 %1, %2 : i32
  // CHECK: vm.return %3 : i32
  return %result : i32
}

// -----

// CHECK-LABEL: @utilAlignInt64
func.func @utilAlignInt64(%arg0 : i64, %arg1: i64) ->  (i64) {
  %result = util.align %arg0, %arg1 : i64
  // CHECK-DAG: %c1 = vm.const.i64 1
  // CHECK-DAG: %0 = vm.sub.i64 %arg1, %c1 : i64
  // CHECK-DAG: %1 = vm.add.i64 %arg0, %0 : i64
  // CHECK-DAG: %2 = vm.not.i64 %0 : i64
  // CHECK-DAG: %3 = vm.and.i64 %1, %2 : i64
  // CHECK: vm.return %3 : i64
  return %result : i64
}

// -----

// CHECK-LABEL: @utilSizeOfIndex
func.func @utilSizeOfIndex() ->  (index) {
  // CHECK: %[[SIZEOF:.*]] = vm.const.i32 4
  %0 = util.sizeof index
  // CHECK: vm.return %[[SIZEOF]]
  return %0 : index
}
