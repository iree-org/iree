// RUN: iree-opt -allow-unregistered-dialect -iree-convert-to-llvm -split-input-file %s | IreeFileCheck %s

llvm.func @sink(f32)

// CHECK-LABEL: llvm.func internal @binding_ptrs
func @binding_ptrs() {
  // CHECK-DAG: %[[C72:.+]] = llvm.mlir.constant(72 : index) : i64
  %c72 = arith.constant 72 : index

  // CHECK: %[[STATE:.+]] = llvm.load %arg0 : !llvm.ptr<struct<[[DISPATCH_STATE_TYPE:.+]]>>
  // CHECK: %[[PC:.+]] = llvm.extractvalue %[[STATE]][3] : !llvm.struct<[[DISPATCH_STATE_TYPE]]>
  // CHECK: %[[C2:.+]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[DIM_PTR:.+]] = llvm.getelementptr %[[PC]][%[[C2]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
  // CHECK: %[[DIM_I32:.+]] = llvm.load %[[DIM_PTR]] : !llvm.ptr<i32>
  // CHECK: %[[DIM:.+]] = llvm.zext %[[DIM_I32]] : i32 to i64
  %dim = hal.interface.constant.load[2] : index

  // CHECK: %[[STATE:.+]] = llvm.load %arg0 : !llvm.ptr<struct<[[DISPATCH_STATE_TYPE]]>>
  // CHECK: %[[BINDING_PTRS:.+]] = llvm.extractvalue %[[STATE]][5] : !llvm.struct<[[DISPATCH_STATE_TYPE]]>
  // CHECK: %[[C1:.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[ARRAY_PTR:.+]] = llvm.getelementptr %[[BINDING_PTRS]][%[[C1]]] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
  // CHECK: %[[BASE_PTR_I8:.+]] = llvm.load %[[ARRAY_PTR]] : !llvm.ptr<ptr<i8>>
  // CHECK: %[[BUFFER_I8:.+]] = llvm.getelementptr %[[BASE_PTR_I8]][%[[C72]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
  // CHECK: %[[BUFFER_F32:.+]] = llvm.bitcast %[[BUFFER_I8]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: %[[DESC_A:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC_B:.+]] = llvm.insertvalue %[[BUFFER_F32]], %[[DESC_A]][0]
  // CHECK: %[[DESC_C:.+]] = llvm.insertvalue %[[BUFFER_F32]], %[[DESC_B]][1]
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: %[[DESC_D:.+]] = llvm.insertvalue %[[C0]], %[[DESC_C]][2]
  // CHECK: %[[DESC_E:.+]] = llvm.insertvalue %[[DIM]], %[[DESC_D]][3, 0]
  // CHECK: %[[C2:.+]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: %[[DESC_F:.+]] = llvm.insertvalue %[[C2]], %[[DESC_E]][3, 1]
  // CHECK: %[[C1:.+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[DESC_G:.+]] = llvm.insertvalue %[[C1]], %[[DESC_F]][4, 1]
  // CHECK: %[[STRIDE1:.+]] = llvm.extractvalue %[[DESC_G]][4, 1]
  // CHECK: %[[DIM1:.+]] = llvm.extractvalue %[[DESC_G]][3, 1]
  // CHECK: %[[STRIDE0:.+]] = llvm.mul %[[STRIDE1]], %[[DIM1]]  : i64
  // CHECK: %[[DESC_H:.+]] = llvm.insertvalue %[[STRIDE0]], %[[DESC_G]][4, 0]
  %memref = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c72) : memref<?x2xf32>{%dim}

  // CHECK: %[[VAL:.+]] = llvm.load
  %c0 = arith.constant 0 : index
  %val = memref.load %memref[%c0, %c0] : memref<?x2xf32>

  // CHECK: llvm.call @sink(%[[VAL]])
  llvm.call @sink(%val) : (f32) -> ()
  return
}
