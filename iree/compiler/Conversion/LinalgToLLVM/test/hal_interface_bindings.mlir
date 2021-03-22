// RUN: iree-opt -allow-unregistered-dialect -iree-codegen-convert-to-llvm -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: llvm.func internal @binding_ptrs
func @binding_ptrs() {
  // CHECK-DAG: %[[C72:.+]] = llvm.mlir.constant(72 : index) : i64
  %c72 = constant 72 : index
  // CHECK: %[[STATE:.+]] =  llvm.load %arg0 : !llvm.ptr<struct<"iree_hal_executable_dispatch_state_v0_t", (array<3 x i32>, array<3 x i32>, i64, ptr<i32>, i64, ptr<ptr<i8>>, ptr<i64>)>>
  // CHECK: %[[BINDING_PTRS:.+]] = llvm.extractvalue %[[STATE]][5]
  // CHECK: %[[C1:.+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[ARRAY_PTR:.+]] = llvm.getelementptr %[[BINDING_PTRS]][%[[C1]]] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
  // CHECK: %[[BASE_PTR_I8:.+]] = llvm.load %[[ARRAY_PTR]] : !llvm.ptr<ptr<i8>>
  // CHECK: %[[BUFFER_I8:.+]] = llvm.getelementptr %[[BASE_PTR_I8]][%[[C72]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
  // CHECK: %[[BUFFER_F32:.+]] = llvm.bitcast %[[BUFFER_I8]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: %[[DESC_A:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[DESC_B:.+]] = llvm.insertvalue %[[BUFFER_F32]], %[[DESC_A]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[DESC_C:.+]] = llvm.insertvalue %[[BUFFER_F32]], %[[DESC_B]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: %[[DESC_D:.+]] = llvm.insertvalue %[[C0]], %[[DESC_C]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %memref = hal.interface.binding.subspan @io::@ret0[%c72] : memref<?xf32>
  // CHECK: "test.sink"(%[[DESC_D]])
  "test.sink"(%memref) : (memref<?xf32>) -> ()
  return
}
hal.interface @io attributes {push_constants = 2 : i32, sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
}

// -----

// CHECK-LABEL: llvm.func internal @tie_shape
func @tie_shape() {
  %c72 = constant 72 : index
  // ...
  // CHECK: %[[DYN_MEMREF_T1:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DYN_MEMREF_T2:.+]] = llvm.insertvalue %{{.+}}, %[[DYN_MEMREF_T1]][0]
  // CHECK: %[[DYN_MEMREF_T3:.+]] = llvm.insertvalue %{{.+}}, %[[DYN_MEMREF_T2]][1]
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: %[[DYN_MEMREF:.+]] = llvm.insertvalue %[[C0]], %[[DYN_MEMREF_T3]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %memref = hal.interface.binding.subspan @io::@ret0[%c72] : memref<?x2xf32>
  // ...
  //  CHECK: %[[CDIM0_I32:.+]] = llvm.load %16 : !llvm.ptr<i32>
  //  CHECK: %[[CDIM0:.+]] = llvm.zext %[[CDIM0_I32]] : i32 to i64
  %dim = hal.interface.load.constant offset = 0 : index
  %shape = shapex.make_ranked_shape %dim : (index) -> !shapex.ranked_shape<[?,2]>
  //  CHECK: %[[MEMREF_T0:.+]] = llvm.insertvalue %[[CDIM0]], %[[DYN_MEMREF]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  //  CHECK: %[[C2:.+]] = llvm.mlir.constant(2 : index) : i64
  //  CHECK: %[[MEMREF_T1:.+]] = llvm.insertvalue %[[C2]], %[[MEMREF_T0]][3, 1]
  //  CHECK: %[[C1:.+]] = llvm.mlir.constant(1 : index) : i64
  //  CHECK: %[[TIED_MEMREF:.+]] = llvm.insertvalue %[[C1]], %[[MEMREF_T1]][4, 1]
  //  CHECK: %[[STRIDE1:.+]] = llvm.extractvalue %[[TIED_MEMREF]][4, 1]
  //  CHECK: %[[DIM1:.+]] = llvm.extractvalue %[[TIED_MEMREF]][3, 1]
  //  CHECK: %[[STRIDE0:.+]] = llvm.mul %[[STRIDE1]], %[[DIM1]]  : i64
  //  CHECK: %[[FINAL_MEMREF:.+]] = llvm.insertvalue %[[STRIDE0]], %[[TIED_MEMREF]][4, 0]
  %tied_memref = shapex.tie_shape %memref, %shape : memref<?x2xf32>, !shapex.ranked_shape<[?,2]>
  //  CHECK: "test.sink"(%[[FINAL_MEMREF]])
  "test.sink"(%tied_memref) : (memref<?x2xf32>) -> ()
  return
}
hal.interface @io attributes {push_constants = 2 : i32, sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
}
