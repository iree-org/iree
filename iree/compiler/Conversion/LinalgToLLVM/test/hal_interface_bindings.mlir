// RUN: iree-opt -allow-unregistered-dialect -iree-codegen-convert-to-llvm -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: llvm.func @binding_ptrs
func @binding_ptrs() {
  // CHECK-DAG: %[[C72:.+]] = llvm.mlir.constant(72 : index) : i64
  %c72 = constant 72 : index
  // CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[ARRAY_PTR:.+]] = llvm.getelementptr %arg0[%[[C1]]] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
  // CHECK: %[[BASE_PTR_I8:.+]] = llvm.load %[[ARRAY_PTR]] : !llvm.ptr<ptr<i8>>
  // CHECK: %[[BUFFER_I8:.+]] = llvm.getelementptr %[[BASE_PTR_I8]][%[[C72]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
  // CHECK: %[[BUFFER_F32:.+]] = llvm.bitcast %[[BUFFER_I8]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: %[[DESC_A:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[DESC_B:.+]] = llvm.insertvalue %[[BUFFER_F32]], %[[DESC_A]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[DESC_C:.+]] = llvm.insertvalue %[[BUFFER_F32]], %[[DESC_B]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %memref = hal.interface.binding.subspan @io::@ret0[%c72] : memref<?xf32>
  // CHECK: "test.sink"(%[[DESC_C]])
  "test.sink"(%memref) : (memref<?xf32>) -> ()
  return
}
hal.interface @io attributes {push_constants = 2 : i32, sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
}
