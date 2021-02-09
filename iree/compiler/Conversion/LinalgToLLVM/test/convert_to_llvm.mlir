// RUN: iree-opt -iree-codegen-convert-to-llvm -cse -split-input-file %s | IreeFileCheck %s

// CHECK_LABEL: @convert_dynamic_shape
func @convert_dynamic_shape() {
  %c0 = constant 0 : index
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<?x?xf32>
  %1 = hal.interface.load.constant offset = 0 : index
  %2 = hal.interface.load.constant offset = 1 : index
  %3 = shapex.make_ranked_shape %1, %2 : (index, index) -> !shapex.ranked_shape<[?,?]>
  %6 = shapex.tie_shape %0, %3 : memref<?x?xf32>, !shapex.ranked_shape<[?,?]>
  %7 = load %6[%c0, %c0] : memref<?x?xf32>
  %8 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x?xf32>
  %9 = shapex.tie_shape %8, %3 : memref<?x?xf32>, !shapex.ranked_shape<[?,?]>
  store %7, %8[%c0, %c0] : memref<?x?xf32>
  return
}
hal.interface @legacy_io attributes {push_constants = 2 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
}
// CHECK: llvm.func @convert_dynamic_shape(%[[ARG0:.+]]: !llvm.ptr<ptr<i8>>, %[[ARG1:.+]]: !llvm.ptr<i32>, %[[WORKGROUP_ID:.+]]: !llvm.ptr<array<3 x i32>>, %[[WORKGROUP_COUNT:.+]]: !llvm.ptr<array<3 x i32>>, %[[WORKGROUP_SIZE:.+]]: !llvm.ptr<array<3 x i32>>) {
// CHECK: %[[CONST0:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[MEMREF0_PTR_PTR:.+]] = llvm.getelementptr %[[ARG0]][%[[CONST0]]] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
// CHECK: %[[MEMREF0_PTR:.+]] = llvm.load %[[MEMREF0_PTR_PTR]] : !llvm.ptr<ptr<i8>>
// CHECK: %[[MEMREF0_BASE_PTR:.+]] = llvm.bitcast %[[MEMREF0_PTR]] : !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK: %[[MEMREF_DESC:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_1:.+]] = llvm.insertvalue %[[MEMREF0_BASE_PTR]], %[[MEMREF_DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_2:.+]] = llvm.insertvalue %[[MEMREF0_BASE_PTR]], %[[MEMREF0_1]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[DIM0_PTR:.+]] = llvm.getelementptr %[[ARG1]][%[[CONST0]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK: %[[DIM0:.+]] = llvm.load %[[DIM0_PTR]] : !llvm.ptr<i32>
// CHECK: %[[DIM0_i64:.+]] = llvm.zext %[[DIM0]] : i32 to i64
// CHECK: %[[CONST1:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[DIM1_PTR:.+]] = llvm.getelementptr %[[ARG1]][%[[CONST1]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK: %[[DIM1:.+]] = llvm.load %[[DIM1_PTR]] : !llvm.ptr<i32>
// CHECK: %[[DIM1_i64:.+]] = llvm.zext %[[DIM1]] : i32 to i64
// CHECK: %[[MEMREF0_3:.+]] = llvm.insertvalue %[[DIM0_i64]], %[[MEMREF0_2]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_4:.+]] = llvm.insertvalue %[[DIM1_i64]], %[[MEMREF0_3]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_5:.+]] = llvm.insertvalue %[[CONST1]], %[[MEMREF0_4]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_STRIDE_1:.+]] = llvm.extractvalue %[[MEMREF0_5]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_SIZE_1:.+]] = llvm.extractvalue %[[MEMREF0_5]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_STRIDE_0:.+]] = llvm.mul %[[MEMREF0_STRIDE_1]], %[[MEMREF0_SIZE_1]]  : i64
// CHECK: %[[MEMREF0_6:.+]] = llvm.insertvalue %[[MEMREF0_STRIDE_0]], %[[MEMREF0_5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_BASE_PTR_1:.+]] = llvm.extractvalue %[[MEMREF0_6]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_STRIDE0_0:.+]] = llvm.extractvalue %[[MEMREF0_6]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_00_BASE:.+]] = llvm.mul %[[CONST0]], %[[MEMREF0_STRIDE0_0]]  : i64
// CHECK: %[[MEMREF0_00_OFFSET:.+]] = llvm.add %[[MEMREF0_00_BASE]], %[[CONST0]]  : i64
// CHECK: %[[MEMREF0_00_PTR:.+]] = llvm.getelementptr %[[MEMREF0_BASE_PTR_1]][%[[MEMREF0_00_OFFSET]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[MEMREF0_00:.+]] = llvm.load %[[MEMREF0_00_PTR]] : !llvm.ptr<f32>
// CHECK: %[[MEMREF1_PTR_PTR:.+]] = llvm.getelementptr %[[ARG0]][%[[CONST1]]] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
// CHECK: %[[MEMREF1_PTR:.+]] = llvm.load %[[MEMREF1_PTR_PTR]] : !llvm.ptr<ptr<i8>>
// CHECK: %[[MEMREF1_BASE_PTR:.+]] = llvm.bitcast %[[MEMREF1_PTR]] : !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK: %[[MEMREF1_0:.+]] = llvm.insertvalue %[[MEMREF1_BASE_PTR]], %[[MEMREF_DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_1:.+]] = llvm.insertvalue %[[MEMREF1_BASE_PTR]], %[[MEMREF1_0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_BASE:.+]] = llvm.extractvalue %[[MEMREF1_1]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_STRIDE_0:.+]] = llvm.extractvalue %[[MEMREF1_1]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_00_OFFSET:.+]] = llvm.mul %[[CONST0]], %[[MEMREF1_STRIDE_0]]  : i64
// CHECK: %[[MEMREF1_00_ADDRS:.+]] = llvm.add %[[MEMREF1_00_OFFSET]], %[[CONST0]]  : i64
// CHECK: %[[MEMREF1_00_PTR:.+]] = llvm.getelementptr %[[MEMREF1_BASE]][%[[MEMREF1_00_ADDRS]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: llvm.store %[[MEMREF0_00]], %[[MEMREF1_00_PTR]] : !llvm.ptr<f32>
// CHECK: llvm.return

// -----

// CHECK_LABEL: @convert_dynamic_shape2
func @convert_dynamic_shape2() {
  %c0 = constant 0 : index
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io2::@arg0} : memref<2x?xf32>
  %1 = hal.interface.load.constant offset = 0 : index
  %2 = shapex.make_ranked_shape %1 : (index) -> !shapex.ranked_shape<[2,?]>
  %3 = shapex.tie_shape %0, %2 : memref<2x?xf32>, !shapex.ranked_shape<[2,?]>
  %4 = load %3[%c0, %c0] : memref<2x?xf32>
  %5 = iree.placeholder for "interface buffer" {binding = @legacy_io2::@ret0} : memref<2x?xf32>
  %9 = shapex.tie_shape %5, %2 : memref<2x?xf32>, !shapex.ranked_shape<[2,?]>
  store %4, %9[%c0, %c0] : memref<2x?xf32>
  return
}
hal.interface @legacy_io2 attributes {push_constants = 1 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
}
// CHECK: llvm.func @convert_dynamic_shape2(%[[ARG0:.+]]: !llvm.ptr<ptr<i8>>, %[[ARG1:.+]]: !llvm.ptr<i32>, %[[WORKGROUP_ID:.+]]: !llvm.ptr<array<3 x i32>>, %[[WORKGROUP_COUNT:.+]]: !llvm.ptr<array<3 x i32>>, %[[WORKGROUP_SIZE:.+]]: !llvm.ptr<array<3 x i32>>) {
// CHECK: %[[CONST0:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[MEMREF0_PTR_PTR:.+]] = llvm.getelementptr %[[ARG0]][%[[CONST0]]] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
// CHECK: %[[MEMREF0_PTR:.+]] = llvm.load %[[MEMREF0_PTR_PTR]] : !llvm.ptr<ptr<i8>>
// CHECK: %[[MEMREF0_BASE_PTR:.+]] = llvm.bitcast %[[MEMREF0_PTR]] : !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK: %[[MEMREF_DESC:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_1:.+]] = llvm.insertvalue %[[MEMREF0_BASE_PTR]], %[[MEMREF_DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_2:.+]] = llvm.insertvalue %[[MEMREF0_BASE_PTR]], %[[MEMREF0_1]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[DIM1_PTR:.+]] = llvm.getelementptr %[[ARG1]][%[[CONST0]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK: %[[DIM1:.+]] = llvm.load %[[DIM1_PTR]] : !llvm.ptr<i32>
// CHECK: %[[DIM1_i64:.+]] = llvm.zext %[[DIM1]] : i32 to i64
// CHECK: %[[CONST2:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK: %[[MEMREF0_3:.+]] = llvm.insertvalue %[[CONST2]], %[[MEMREF0_2]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_4:.+]] = llvm.insertvalue %[[DIM1_i64]], %[[MEMREF0_3]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[CONST1:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[MEMREF0_5:.+]] = llvm.insertvalue %[[CONST1]], %[[MEMREF0_4]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_STRIDE_1:.+]] = llvm.extractvalue %[[MEMREF0_5]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_SIZE_1:.+]] = llvm.extractvalue %[[MEMREF0_5]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_STRIDE_0:.+]] = llvm.mul %[[MEMREF0_STRIDE_1]], %[[MEMREF0_SIZE_1]]  : i64
// CHECK: %[[MEMREF0_6:.+]] = llvm.insertvalue %[[MEMREF0_STRIDE_0]], %[[MEMREF0_5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_BASE_PTR_1:.+]] = llvm.extractvalue %[[MEMREF0_6]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_STRIDE_0_0:.+]] = llvm.extractvalue %[[MEMREF0_6]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF0_00_ADDRS:.+]] = llvm.mul %[[CONST0]], %[[MEMREF0_STRIDE_0_0]]  : i64
// CHECK: %[[MEMREF0_00_INDEX:.+]] = llvm.add %[[MEMREF0_00_ADDRS]], %[[CONST0]]  : i64
// CHECK: %[[MEMREF0_00_PTR:.+]] = llvm.getelementptr %19[%[[MEMREF0_00_INDEX]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[MEMREF0_00:.+]] = llvm.load %[[MEMREF0_00_PTR]] : !llvm.ptr<f32>
// CHECK: %[[MEMREF1_PTR_PTR:.+]] = llvm.getelementptr %[[ARG0]][%[[CONST1]]] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
// CHECK: %[[MEMREF1_PTR:.+]] = llvm.load %[[MEMREF1_PTR_PTR]] : !llvm.ptr<ptr<i8>>
// CHECK: %[[MEMREF1_BASE_PTR:.+]] = llvm.bitcast %[[MEMREF1_PTR]] : !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK: %[[MEMREF1_1:.+]] = llvm.insertvalue %[[MEMREF1_BASE_PTR]], %[[MEMREF_DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_2:.+]] = llvm.insertvalue %[[MEMREF1_BASE_PTR]], %[[MEMREF1_1]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_3:.+]] = llvm.insertvalue %[[CONST2]], %[[MEMREF1_2]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_4:.+]] = llvm.insertvalue %[[DIM1_i64]], %[[MEMREF1_3]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_5:.+]] = llvm.insertvalue %[[CONST1]], %[[MEMREF1_4]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_STRIDE1:.+]] = llvm.extractvalue %[[MEMREF1_5]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_SIZE1:.+]] = llvm.extractvalue %[[MEMREF1_5]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_STRIDE0:.+]] = llvm.mul %[[MEMREF1_STRIDE1]], %[[MEMREF1_SIZE1]]  : i64
// CHECK: %[[MEMREF1_6:.+]] = llvm.insertvalue %[[MEMREF1_STRIDE0]], %[[MEMREF1_5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_BASE_PTR:.+]] = llvm.extractvalue %[[MEMREF1_6]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_STRIDE0_0:.+]] = llvm.extractvalue %[[MEMREF1_6]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF1_00_OFFSET:.+]] = llvm.mul %[[CONST0]], %[[MEMREF1_STRIDE0_0]]  : i64
// CHECK: %[[MEMREF1_00_INDEX:.+]] = llvm.add %[[MEMREF1_00_OFFSET]], %[[CONST0]]  : i64
// CHECK: %[[MEMREF1_00_PTR:.+]] = llvm.getelementptr %[[MEMREF1_BASE_PTR]][%[[MEMREF1_00_INDEX]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: llvm.store %[[MEMREF0_00]], %[[MEMREF1_00_PTR]] : !llvm.ptr<f32>
// CHECK: llvm.return

// -----

// CHECK_LABEL: @distribute_lookup
func @distribute_lookup() {
  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io3::@arg0} : memref<2x2x2xf32>
  %1 = hal.interface.workgroup.id[0] : index
  %2 = hal.interface.workgroup.id[1] : index
  %3 = hal.interface.workgroup.id[2] : index
  %4 = load %0[%1, %2, %3] : memref<2x2x2xf32>
  %5 = iree.placeholder for "interface buffer" {binding = @legacy_io3::@ret0} : memref<2x2x2xf32>
  store %4, %5[%1, %2, %3] : memref<2x2x2xf32>
  return
}
hal.interface @legacy_io3 attributes {push_constants = 1 : i32, sym_visibility = "private"} {
  hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
}
// CHECK: llvm.func @distribute_lookup(%[[ARG0:.+]]: !llvm.ptr<ptr<i8>>, %[[ARG1:.+]]: !llvm.ptr<i32>, %[[WORKGROUP_ID:.+]]: !llvm.ptr<array<3 x i32>>, %[[WORKGROUP_COUNT:.+]]: !llvm.ptr<array<3 x i32>>, %[[WORKGROUP_SIZE:.+]]: !llvm.ptr<array<3 x i32>>) {
// CHECK: %[[CONST0:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[CONST2:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK: %[[CONST4:.+]] = llvm.mlir.constant(4 : index) : i64
// CHECK: %[[WORKGROUP_ID_DATA_0:.+]] = llvm.load %[[WORKGROUP_ID]] : !llvm.ptr<array<3 x i32>>
// CHECK: %[[WORKGROUP_ID_Z:.+]] = llvm.extractvalue %[[WORKGROUP_ID_DATA_0]][0 : i32] : !llvm.array<3 x i32>
// CHECK: %[[WORKGROUP_ID_Z_i64:.+]] = llvm.zext %[[WORKGROUP_ID_Z]] : i32 to i64
// CHECK: %[[WORKGROUP_ID_DATA_1:.+]] = llvm.load %[[WORKGROUP_ID]] : !llvm.ptr<array<3 x i32>>
// CHECK: %[[WORKGROUP_ID_Y:.+]] = llvm.extractvalue %[[WORKGROUP_ID_DATA_1]][1 : i32] : !llvm.array<3 x i32>
// CHECK: %[[WORKGROUP_ID_Y_i64:.+]] = llvm.zext %[[WORKGROUP_ID_Y]] : i32 to i64
// CHECK: %[[WORKGROUP_ID_DATA_2:.+]] = llvm.load %[[WORKGROUP_ID]] : !llvm.ptr<array<3 x i32>>
// CHECK: %[[WORKGROUP_ID_X:.+]] = llvm.extractvalue %[[WORKGROUP_ID_DATA_2]][2 : i32] : !llvm.array<3 x i32>
// CHECK: %[[WORKGROUP_ID_X_i64:.+]] = llvm.zext %[[WORKGROUP_ID_X]] : i32 to i64
// CHECK: %[[LOAD_STRIDE_Z:.+]] = llvm.mul %[[WORKGROUP_ID_Z_i64]], %[[CONST4]]
// CHECK: %[[LOAD_STRIDE_Y:.+]] llvm.mul %[[WORKGROUP_ID_Y_i64]], %[[CONST2]]
