// RUN: iree-opt -iree-codegen-convert-to-llvm -split-input-file %s | IreeFileCheck %s

// CHECK_LABEL: @convert_dynamic_shape
func @convert_dynamic_shape() -> f32 {
  %c0 = constant 0 : index
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x?xf32>
  %1 = hal.interface.load.constant offset = 0 : index
  %2 = hal.interface.load.constant offset = 1 : index
  %3 = shapex.make_ranked_shape %1, %2 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %6 = shapex.tie_shape %0, %3 : memref<?x?xf32>, !shapex.ranked_shape<[?,?]>
  %7 = load %6[%c0, %c0] : memref<?x?xf32>
  return %7 : f32
}
hal.interface @legacy_io attributes {push_constants = 2 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
}
// CHECK: llvm.func @convert_dynamic_shape(%[[ARG0:.+]]: !llvm.ptr<ptr<i8>>, %[[ARG1:.+]]: !llvm.ptr<i32>, %[[THREAD_X_ID:.+]]: !llvm.i32, %[[THREAD_Y_ID:.+]]: !llvm.i32, %[[THREAD_Z_ID:.+]]: !llvm.i32)
// CHECK: %[[PACKED_ARGS_PTR:.+]] = llvm.bitcast %[[ARG0]] : !llvm.ptr<ptr<i8>> to !llvm.ptr<struct<(ptr<float>)>>
// CHECK: %[[PACKED_ARGS:.+]] = llvm.load %[[PACKED_ARGS_PTR]] : !llvm.ptr<struct<(ptr<float>)>>
// CHECK: %[[MEMREF0_DATA_PTR:.+]] = llvm.extractvalue %[[PACKED_ARGS]][0] : !llvm.struct<(ptr<float>)>
// CHECK: %[[MEMREF0:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_0:.+]] = llvm.insertvalue %[[MEMREF0_DATA_PTR]], %[[MEMREF0]][0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_1:.+]] = llvm.insertvalue %[[MEMREF0_DATA_PTR]], %[[MEMREF0_0]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[CONST0:.+]] = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK: %[[DIM0_PTR:.+]] = llvm.getelementptr %[[ARG1]][%[[CONST0]]] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
// CHECK: %[[DIM0:.+]] = llvm.load %[[DIM0_PTR]] : !llvm.ptr<i32>
// CHECK: %[[DIM0CASTED:.+]] = llvm.zext %[[DIM0]] : !llvm.i32 to !llvm.i64
// CHECK: %[[CONST1:.+]] = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK: %[[DIM1_PTR:.+]] = llvm.getelementptr %[[ARG1]][%[[CONST1]]] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
// CHECK: %[[DIM1:.+]] = llvm.load %[[DIM1_PTR]] : !llvm.ptr<i32>
// CHECK: %[[DIM1CASTED:.+]] = llvm.zext %[[DIM1]] : !llvm.i32 to !llvm.i64
// CHECK: %[[MEMREF0_2:.+]] = llvm.insertvalue %[[DIM0CASTED]], %[[MEMREF0_1]][3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_3:.+]] = llvm.insertvalue %[[DIM1CASTED]], %[[MEMREF0_2]][3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[CONST1_STRIDE:.+]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: %[[MEMREF0_4:.+]] = llvm.insertvalue %[[CONST1_STRIDE]], %[[MEMREF0_3]][4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STRIDE_DIM1:.+]] = llvm.extractvalue %[[MEMREF0_4]][4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[DIM1_0:.+]] = llvm.extractvalue %[[MEMREF0_4]][3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STRIDE_DIM0:.+]] = llvm.mul %[[STRIDE_DIM1]], %[[DIM1_0]] : !llvm.i64
// CHECK: llvm.insertvalue %[[STRIDE_DIM0]], %[[MEMREF0_4]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>

// -----

// CHECK_LABEL: @convert_dynamic_shape2
func @convert_dynamic_shape2() -> f32 {
  %c0 = constant 0 : index
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io2::@arg0} : memref<2x?xf32>
  %1 = hal.interface.load.constant offset = 0 : index
  %2 = shapex.make_ranked_shape %1 : (index) -> !shapex.ranked_shape<[2,?]>
  %3 = shapex.tie_shape %0, %2 : memref<2x?xf32>, !shapex.ranked_shape<[2,?]>
  %4 = load %3[%c0, %c0] : memref<2x?xf32>
  return %4 : f32
}
hal.interface @legacy_io2 attributes {push_constants = 1 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
}
// CHECK: llvm.func @convert_dynamic_shape2(%[[ARG0:.+]]: !llvm.ptr<ptr<i8>>, %[[ARG1:.+]]: !llvm.ptr<i32>, %[[THREAD_X_ID:.+]]: !llvm.i32, %[[THREAD_Y_ID:.+]]: !llvm.i32, %[[THREAD_Z_ID:.+]]: !llvm.i32)
// CHECK: %[[PACKED_ARGS_PTR:.+]] = llvm.bitcast %[[ARG0]] : !llvm.ptr<ptr<i8>> to !llvm.ptr<struct<(ptr<float>)>>
// CHECK: %[[PACKED_ARGS:.+]] = llvm.load %[[PACKED_ARGS_PTR]] : !llvm.ptr<struct<(ptr<float>)>>
// CHECK: %[[MEMREF0_DATA_PTR:.+]] = llvm.extractvalue %[[PACKED_ARGS]][0] : !llvm.struct<(ptr<float>)>
// CHECK: %[[MEMREF0:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_0:.+]] = llvm.insertvalue %[[MEMREF0_DATA_PTR]], %[[MEMREF0]][0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_1:.+]] = llvm.insertvalue %[[MEMREF0_DATA_PTR]], %[[MEMREF0_0]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[CONST0:.+]] = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK: %[[DIM0_PTR:.+]] = llvm.getelementptr %[[ARG1]][%[[CONST0]]] : (!llvm.ptr<i32>, !llvm.i64) -> !llvm.ptr<i32>
// CHECK: %[[DIM0:.+]] = llvm.load %[[DIM0_PTR]] : !llvm.ptr<i32>
// CHECK: %[[DIM0CASTED:.+]] = llvm.zext %[[DIM0]] : !llvm.i32 to !llvm.i64
// CHECK: %[[CONST0_2:.+]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK: %[[CONST2:.+]] = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK: %[[MEMREF1:.+]] = llvm.insertvalue %[[CONST2:.+]], %[[MEMREF0_1:.+]][3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF2:.+]] = llvm.insertvalue %[[DIM0CASTED:.+]], %[[MEMREF1:.+]][3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[CONST1:.+]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: %[[MEMREF3:.+]] = llvm.insertvalue %[[CONST1:.+]], %[[MEMREF2:.+]][4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STRIDE_DIM1:.+]] = llvm.extractvalue %[[MEMREF3:.+]][4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[DIM1_0:.+]] = llvm.extractvalue %[[MEMREF3:.+]][3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STRIDE_DIM0:.+]] = llvm.mul %[[STRIDE_DIM1:.+]], %[[DIM1_0:.+]]: !llvm.i64
// CHECK: %[[INSERT_DIM0:.+]] = llvm.insertvalue %[[STRIDE_DIM0:.+]], %[[MEMREF3:.+]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[EXTRACT1:.+]] = llvm.extractvalue %[[INSERT_DIM0:.+]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[CONST0_3:.+]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK: %[[EXTRACT2:.+]] = llvm.extractvalue %[[INSERT_DIM0:.+]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MUL1:.+]] = llvm.mul %[[CONST0_2:.+]], %[[EXTRACT2:.+]] : !llvm.i64
// CHECK: %[[ADD1:.+]] = llvm.add %[[CONST0_3:.+]], %[[MUL1:.+]] : !llvm.i64
// CHECK: %[[CONST1_2:.+]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: %[[MUL2:.+]] = llvm.mul %[[CONST0_2:.+]], %[[CONST1_2:.+]] : !llvm.i64
// CHECK: %[[ADD2:.+]] = llvm.add %[[ADD1:.+]], %[[MUL2:.+]] : !llvm.i64
// CHECK: %[[GET_PTR:.+]] = llvm.getelementptr %[[EXTRACT1:.+]][%[[ADD2:.+]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// CHECK: %[[LOAD:.+]] = llvm.load %[[GET_PTR:.+]] : !llvm.ptr<float>

// -----

// CHECK_LABEL: @distribute_lookup
func @distribute_lookup() -> f32 {
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io3::@arg0} : memref<2x2x2xf32>
  %1 = iree.workgroup_id {dimension = "x"} : index
  %2 = iree.workgroup_id {dimension = "y"} : index
  %3 = iree.workgroup_id {dimension = "z"} : index
  %4 = load %0[%1, %2, %3] : memref<2x2x2xf32>
  return %4 : f32
}
hal.interface @legacy_io3 attributes {push_constants = 1 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
}
