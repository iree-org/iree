// RUN: iree-opt --iree-convert-to-llvm --split-input-file %s | FileCheck %s --dump-input=always

// CHECK-LABEL: llvm.func @binding_ptrs(
func.func @binding_ptrs() {
  // CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2
  // CHECK-DAG: %[[C5:.+]] = llvm.mlir.constant(5
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1

  // CHECK: %[[STATE:.+]] = llvm.load %arg1
  // CHECK: %[[BINDING_PTRS:.+]] = llvm.extractvalue %[[STATE]][10]
  // CHECK: %[[ARRAY_PTR:.+]] = llvm.getelementptr %[[BINDING_PTRS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
  // CHECK: %[[BASE_PTR:.+]] = llvm.load %[[ARRAY_PTR]] : !llvm.ptr -> !llvm.ptr
  %c72 = arith.constant 72 : index
  %c128 = arith.constant 128 : index
  %memref = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c72) : memref<?x2xf32, strided<[2, 1], offset: 18>>{%c128}

  // CHECK: %[[OFFSET_PTR0:.+]] = llvm.getelementptr %[[BASE_PTR]][18]
  // CHECK: %[[OFFSET_D0:.+]] = llvm.mul %[[C5]], %[[C2]]
  // CHECK: %[[INDEX1:.+]] = llvm.add %[[OFFSET_D0]], %[[C1]]
  // CHECK: %[[OFFSET_PTR1:.+]] = llvm.getelementptr %[[OFFSET_PTR0]][%[[INDEX1]]] 
  // CHECK: %[[VALUE:.+]] = llvm.load %[[OFFSET_PTR1]]
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %value = memref.load %memref[%c5, %c1] : memref<?x2xf32, strided<[2, 1], offset: 18>>

  // CHECK: llvm.call @sink(%[[VALUE]])
  llvm.call @sink(%value) : (f32) -> ()
  return
}
llvm.func @sink(%arg0: f32) {
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @binding_ptrs_dynamic(
func.func @binding_ptrs_dynamic() {
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1
  // CHECK-DAG: %[[C4:.+]] = llvm.mlir.constant(4
  // CHECK-DAG: %[[C7:.+]] = llvm.mlir.constant(7
  // CHECK-DAG: %[[C5:.+]] = llvm.mlir.constant(5
  // CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(3

  // CHECK: %[[STATE:.+]] = llvm.load %arg1
  // CHECK: %[[CONSTANT_BASEPTR:.+]] = llvm.extractvalue %[[STATE]][9]
  // CHECK: %[[OFFSET:.+]] = llvm.load %[[CONSTANT_BASEPTR]]
  // CHECK: %[[OFFSET_ZEXT:.+]] = llvm.zext %[[OFFSET]]
  %offset = hal.interface.constant.load[0] : index

  // CHECK: %[[STATE0:.+]] = llvm.load %arg1
  // CHECK: %[[CONSTANT_BASEPTR:.+]] = llvm.extractvalue %[[STATE0]][9]
  // CHECK: %[[DIM1_PTR:.+]] = llvm.getelementptr %[[CONSTANT_BASEPTR]][2]
  // CHECK: %[[DIM1:.+]] = llvm.load %[[DIM1_PTR]]
  // CHECK: %[[DIM1_ZEXT:.+]] = llvm.zext %[[DIM1]]
  // CHECK: %[[STATE1:.+]] = llvm.load %arg1
  // CHECK: %[[CONSTANT_BASEPTR0:.+]] = llvm.extractvalue %[[STATE1]][9]
  // CHECK: %[[DIM2_PTR:.+]] = llvm.getelementptr %[[CONSTANT_BASEPTR0]][3]
  // CHECK: %[[DIM2:.+]] = llvm.load %[[DIM2_PTR]]
  // CHECK: %[[DIM2_ZEXT:.+]] = llvm.zext %[[DIM2]]
  %dim0 = hal.interface.constant.load[1]: index
  %dim1 = hal.interface.constant.load[2] : index
  %dim2 = hal.interface.constant.load[3] : index

  // CHECK: %[[STATE3:.+]] = llvm.load %arg1
  // CHECK: %[[BINDING_PTRS:.+]] = llvm.extractvalue %[[STATE3]][10]
  // CHECK: %[[ARRAY_PTR:.+]] = llvm.getelementptr %[[BINDING_PTRS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
  // CHECK: %[[BASE_PTR:.+]] = llvm.load %[[ARRAY_PTR]] : !llvm.ptr -> !llvm.ptr
  %memref = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%offset) : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>{%dim0, %dim1, %dim2}

  // CHECK: %[[BASE_OFFSET:.+]] = llvm.udiv %[[OFFSET_ZEXT]], %[[C4]]
  // CHECK: %[[STRIDE1:.+]] = llvm.mul %[[DIM2_ZEXT]], %[[C1]]
  // CHECK: %[[STRIDE2:.+]] = llvm.mul %[[STRIDE1]], %[[DIM1_ZEXT]]
  // CHECK: %[[OFFSET_PTR0:.+]] = llvm.getelementptr %[[BASE_PTR]][%[[BASE_OFFSET]]]
  // CHECK: %[[INDEX2:.+]] = llvm.mul %[[STRIDE2]], %[[C7]]
  // CHECK: %[[INDEX1:.+]] = llvm.mul %[[STRIDE1]], %[[C5]]
  // CHECK: %[[T1:.+]] = llvm.add %[[INDEX2]], %[[INDEX1]]
  // CHECK: %[[T2:.+]] = llvm.add %[[T1]], %[[C3]]
  // CHECK: %[[OFFSET_PTR1:.+]] = llvm.getelementptr %[[OFFSET_PTR0]][%[[T2]]]
  // CHECK: %[[VALUE:.+]] = llvm.load %[[OFFSET_PTR1]]
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %value = memref.load %memref[%c7, %c5, %c3] : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>

  // CHECK: llvm.call @sink(%[[VALUE]])
  llvm.call @sink(%value) : (f32) -> ()
  return
}
llvm.func @sink(%arg0: f32) {
  llvm.return
}
