// RUN: iree-opt -allow-unregistered-dialect -iree-codegen-convert-to-llvm -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: llvm.func @constant_values
func @constant_values() {
  // CHECK: %[[STATE:.+]] = llvm.load %arg0 : !llvm.ptr<struct<"iree_hal_executable_dispatch_state_v0_t"
  // CHECK: %[[PTR_BASE:.+]] = llvm.extractvalue %[[STATE]][3]
  // CHECK: %[[C1:.+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[VPTR:.+]] = llvm.getelementptr %[[PTR_BASE]][%[[C1]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
  // CHECK: %[[V32:.+]] = llvm.load %[[VPTR]] : !llvm.ptr<i32>
  // CHECK: %[[V64:.+]] = llvm.zext %[[V32]] : i32 to i64
  %v1 = hal.interface.load.constant offset = 1 : index
  // CHECK-NEXT: "test.sink"(%[[V64]])
  "test.sink"(%v1) : (index) -> ()
  return
}
