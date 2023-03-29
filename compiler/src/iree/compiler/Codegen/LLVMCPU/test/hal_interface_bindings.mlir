// RUN: iree-opt --iree-convert-to-llvm --split-input-file %s | FileCheck %s --dump-input=always

// CHECK-LABEL: llvm.func @binding_ptrs
func.func @binding_ptrs() {
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2
  // CHECK-DAG: %[[C5:.+]] = llvm.mlir.constant(5

  // CHECK: %[[STATE:.+]] = llvm.load %arg1
  // CHECK: %[[BINDING_PTRS:.+]] = llvm.extractvalue %[[STATE]][10]
  // CHECK: %[[ARRAY_PTR:.+]] = llvm.getelementptr %[[BINDING_PTRS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
  // CHECK: %[[BASE_PTR:.+]] = llvm.load %[[ARRAY_PTR]] : !llvm.ptr -> !llvm.ptr
  // CHECK: %[[BUFFER:.+]] = llvm.getelementptr %[[BASE_PTR]][72] : (!llvm.ptr) -> !llvm.ptr, i8
  %c72 = arith.constant 72 : index
  %c128 = arith.constant 128 : index
  %memref = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c72) : memref<?x2xf32>{%c128}

  // CHECK: %[[OFFSET_D0:.+]] = llvm.mul %[[C5]], %[[C2]]
  // CHECK: %[[OFFSET_D1:.+]] = llvm.add %[[OFFSET_D0]], %[[C1]]
  // CHECK: %[[OFFSET_PTR:.+]] = llvm.getelementptr %[[BUFFER]][%[[OFFSET_D1]]]
  // CHECK: %[[VALUE:.+]] = llvm.load %[[OFFSET_PTR]]
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %value = memref.load %memref[%c5, %c1] : memref<?x2xf32>

  // CHECK: llvm.call @sink(%[[VALUE]])
  llvm.call @sink(%value) : (f32) -> ()
  return
}
llvm.func @sink(%arg0: f32) {
  llvm.return
}
