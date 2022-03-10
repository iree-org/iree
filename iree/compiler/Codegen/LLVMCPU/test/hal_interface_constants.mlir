// RUN: iree-opt -allow-unregistered-dialect -iree-convert-to-llvm -split-input-file %s | FileCheck %s

llvm.func @sink(i64)

// CHECK-LABEL: llvm.func internal @constant_values
func @constant_values() {
  // CHECK: %[[STATE:.+]] = llvm.load %arg1 : !llvm.ptr<struct<"iree_hal_executable_dispatch_state_v0_t"
  // CHECK: %[[PTR_BASE:.+]] = llvm.extractvalue %[[STATE]][8]
  // CHECK: %[[C1:.+]] = llvm.mlir.constant(1
  // CHECK: %[[VPTR:.+]] = llvm.getelementptr %[[PTR_BASE]][%[[C1]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
  // CHECK: %[[V32:.+]] = llvm.load %[[VPTR]] : !llvm.ptr<i32>
  // CHECK: %[[V64:.+]] = llvm.zext %[[V32]] : i32 to i64
  %v1 = hal.interface.constant.load[1] : index
  // CHECK-NOT: unrealized_conversion_cast
  %v2 = arith.index_cast %v1 : index to i64
  // CHECK: llvm.call @sink
  llvm.call @sink(%v2) : (i64) -> ()
  return
}
