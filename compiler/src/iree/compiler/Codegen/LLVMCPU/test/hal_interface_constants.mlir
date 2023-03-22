// RUN: iree-opt --iree-convert-to-llvm --split-input-file %s | FileCheck %s

// CHECK-LABEL: llvm.func @constant_values
func.func @constant_values() {
  // CHECK: %[[STATE:.+]] = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t"
  // CHECK: %[[PTR_BASE:.+]] = llvm.extractvalue %[[STATE]][9]
  // CHECK: %[[VPTR:.+]] = llvm.getelementptr %[[PTR_BASE]][1] : (!llvm.ptr) -> !llvm.ptr, i32
  // CHECK: %[[V32:.+]] = llvm.load %[[VPTR]] : !llvm.ptr -> i32
  // CHECK: %[[V64:.+]] = llvm.zext %[[V32]] : i32 to i64
  %v1 = hal.interface.constant.load[1] : index
  // CHECK-NOT: unrealized_conversion_cast
  %v2 = arith.index_cast %v1 : index to i64
  // CHECK: llvm.call @sink
  llvm.call @sink(%v2) : (i64) -> ()
  return
}
llvm.func @sink(%arg0: i64) {
  llvm.return
}

