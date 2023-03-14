// RUN: iree-opt --split-input-file --iree-codegen-flatten-memref-subspan --canonicalize --allow-unregistered-dialect %s | FileCheck %s

func.func @load_subspan_with_offset(%offset : index, %i0: index, %i1: index, %i2: index) -> f32 {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<6x7x8xf32, strided<[56, 8, 1], offset: ?>>
  %val = memref.load %subspan[%i0, %i1, %i2] : memref<6x7x8xf32, strided<[56, 8, 1], offset: ?>>
  return %val: f32
}

//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 336)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 56 + s1 * 8 + s2 + s3 floordiv 4)>
//CHECK-LABEL: func.func @load_subspan_with_offset
// CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//  CHECK-DAG:   %[[ZERO:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[SIZE:.+]] = affine.apply #[[$MAP0]]()[%[[OFFSET]]]
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[ZERO]]) : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP1]]()[%[[I0]], %[[I1]], %[[I2]], %[[OFFSET]]]
//      CHECK:   %[[LOAD:.+]] = memref.load %[[SUBSPAN]][%[[INDEX]]]
//      CHECK:   return %[[LOAD]]

// -----

func.func @store_subspan_with_offset(%value: f32, %offset : index, %i0: index, %i1: index, %i2: index) {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<2x3x4xf32, strided<[12, 4, 1], offset: ?>>
  memref.store %value, %subspan[%i0, %i1, %i2] : memref<2x3x4xf32, strided<[12, 4, 1], offset: ?>>
  return
}

//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 24)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 12 + s1 * 4 + s2 + s3 floordiv 4)>
//CHECK-LABEL: func.func @store_subspan_with_offset
// CHECK-SAME: (%[[VALUE:.+]]: f32, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//  CHECK-DAG:   %[[ZERO:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[SIZE:.+]] = affine.apply #[[$MAP0]]()[%[[OFFSET]]
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[ZERO]]) : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP1]]()[%[[I0]], %[[I1]], %[[I2]], %[[OFFSET]]]
//      CHECK:   memref.store %[[VALUE]], %[[SUBSPAN]][%[[INDEX]]] : memref<?xf32>

// -----

func.func @load_subspan_with_vector_element(%offset : index, %i0: index, %i1: index, %i2: index) -> vector<4xf32> {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<6x7x8xvector<4xf32>, strided<[56, 8, 1], offset:?>>
  %val = memref.load %subspan[%i0, %i1, %i2] : memref<6x7x8xvector<4xf32>, strided<[56, 8, 1], offset:?>>
  return %val: vector<4xf32>
}

//      CHECK: #[[$MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 56 + s1 * 8 + s2 + s3 floordiv 16)>
//CHECK-LABEL: func.func @load_subspan_with_vector_element
//      CHECK:   affine.apply #[[$MAP]]()

// -----

func.func @load_subspan_with_16bit_element(%offset : index, %i0: index, %i1: index, %i2: index) -> f16 {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<6x7x8xf16, strided<[56, 8, 1], offset:?>>
  %val = memref.load %subspan[%i0, %i1, %i2] : memref<6x7x8xf16, strided<[56, 8, 1], offset:?>>
  return %val: f16
}

//      CHECK: #[[$MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 56 + s1 * 8 + s2 + s3 floordiv 2)>
//CHECK-LABEL: func.func @load_subspan_with_16bit_element
//      CHECK:   affine.apply #[[$MAP]]()

// -----

func.func @store_subspan_with_leading_dynamic_dim(%value: f32, %offset : index, %i0: index, %i1: index, %i2: index) {
  %dim = hal.interface.constant.load[0] : index
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<?x3x4xf32, strided<[12, 4, 1], offset:?>>{%dim}
  memref.store %value, %subspan[%i0, %i1, %i2] : memref<?x3x4xf32, strided<[12, 4, 1], offset:?>>
  return
}

//      CHECK: #[[$SIZE_MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 12 + s1 floordiv 4)
//      CHECK: #[[$OFFSET_MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 12 + s1 * 4 + s2 + s3 floordiv 4)>
//CHECK-LABEL: func.func @store_subspan_with_leading_dynamic_dim
// CHECK-SAME: (%[[VALUE:.+]]: f32, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM:.+]] = hal.interface.constant.load[0] : index
//      CHECK:   %[[SIZE:.+]] = affine.apply #[[$SIZE_MAP]]()[%[[DIM]], %[[OFFSET]]]
//      CHECK:   %[[DST:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$OFFSET_MAP]]()[%[[I0]], %[[I1]], %[[I2]], %[[OFFSET]]]
//      CHECK:   memref.store %[[VALUE]], %[[DST]][%[[INDEX]]] : memref<?xf32>

// -----

func.func @store_subspan_with_all_dynamic_dim(%value: f32, %offset : index, %i0: index, %i1: index, %i2: index, %i3: index) {
  %dim0 = hal.interface.constant.load[0] : index
  %dim1 = hal.interface.constant.load[1] : index
  %dim2 = hal.interface.constant.load[2] : index
  %dim3 = hal.interface.constant.load[3] : index
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>>{%dim0, %dim1, %dim2, %dim3}
  memref.store %value, %subspan[%i0, %i1, %i2, %i3] : memref<?x?x?x?xf32, strided<[?, ?, ?, 1], offset: ?>>
  return
}

//      CHECK: #[[$SIZE_MAP:.+]] = affine_map<()[s0, s1, s2, s3, s4] -> (((s0 * s1) * s2) * s3 + s4 floordiv 4)>
//      CHECK: #[[$OFFSET_MAP:.+]] = affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7] -> (s1 + (s3 + (s6 + s4 * s5) * s2) * s0 + s7 floordiv 4)>
//CHECK-LABEL: func.func @store_subspan_with_all_dynamic_dim
// CHECK-SAME: (%[[VALUE:.+]]: f32, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index)
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM0:.+]] = hal.interface.constant.load[0] : index
//      CHECK:   %[[DIM1:.+]] = hal.interface.constant.load[1] : index
//      CHECK:   %[[DIM2:.+]] = hal.interface.constant.load[2] : index
//      CHECK:   %[[DIM3:.+]] = hal.interface.constant.load[3] : index
//      CHECK:   %[[SIZE:.+]] = affine.apply #[[$SIZE_MAP]]()[%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]], %[[OFFSET]]]
//      CHECK:   %[[DST:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$OFFSET_MAP]]()[%[[DIM3]], %[[I3]], %[[DIM2]], %[[I2]], %[[I0]], %[[DIM1]], %[[I1]], %[[OFFSET]]]
//      CHECK:   memref.store %[[VALUE]], %[[DST]][%[[INDEX]]]

// -----

func.func @store_subspan_with_mixed_dynamic_dim(%value: f32, %offset : index, %i0: index, %i1: index, %i2: index, %i3: index) {
  %dim0 = hal.interface.constant.load[0] : index
  %dim1 = hal.interface.constant.load[1] : index
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<?x4x?x8xf32, strided<[?, ?, 8, 1], offset: ?>>{%dim0, %dim1}
  memref.store %value, %subspan[%i0, %i1, %i2, %i3] : memref<?x4x?x8xf32, strided<[?, ?, 8, 1], offset: ?>>
  return
}

//      CHECK: #[[$SIZE_MAP:.+]] = affine_map<()[s0, s1, s2] -> ((s0 * s1) * 32 + s2 floordiv 4)>
//      CHECK: #[[$OFFSET_MAP:.+]] = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s0 + s2 * 8 + ((s3 * 4 + s4) * s1) * 8 + s5 floordiv 4)>
//CHECK-LABEL: func.func @store_subspan_with_mixed_dynamic_dim
// CHECK-SAME: (%[[VALUE:.+]]: f32, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index)
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM0:.+]] = hal.interface.constant.load[0] : index
//      CHECK:   %[[DIM2:.+]] = hal.interface.constant.load[1] : index
//      CHECK:   %[[SIZE:.+]] = affine.apply #[[$SIZE_MAP]]()[%[[DIM0]], %[[DIM2]], %[[OFFSET]]]
//      CHECK:   %[[DST:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$OFFSET_MAP]]()[%[[I3]], %[[DIM2]], %[[I2]], %[[I0]], %[[I1]], %[[OFFSET]]]
//      CHECK:   memref.store %[[VALUE]], %[[DST]][%[[INDEX]]]

// -----

func.func @store_subspan_with_flow_control(%value: f32, %offset : index, %i0: index, %i1: index, %i2: index) {
  %dim = hal.interface.constant.load[0] : index
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<?x3x4xf32, strided<[12, 4, 1], offset:?>>{%dim}
  scf.for %i = %i0 to %i1 step %i2 {
    memref.store %value, %subspan[%i0, %i1, %i2] : memref<?x3x4xf32, strided<[12, 4, 1], offset:?>>
  }
  return
}

//      CHECK: #[[$SIZE_MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 12 + s1 floordiv 4)
//      CHECK: #[[$OFFSET_MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 12 + s1 * 4 + s2 + s3 floordiv 4)>
//CHECK-LABEL: func.func @store_subspan_with_flow_control
// CHECK-SAME: (%[[VALUE:.+]]: f32, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM:.+]] = hal.interface.constant.load[0] : index
//      CHECK:   %[[SIZE:.+]] = affine.apply #[[$SIZE_MAP]]()[%[[DIM]], %[[OFFSET]]]
//      CHECK:   %[[DST:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[SIZE]]}
//      CHECK: scf.for
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$OFFSET_MAP]]()[%[[I0]], %[[I1]], %[[I2]], %[[OFFSET]]]
//      CHECK:   memref.store %[[VALUE]], %[[DST]][%[[INDEX]]] : memref<?xf32>

// -----

func.func @load_store_alloc_static(%value : f32, %i0: index, %i1 : index, %i2: index) -> f32 {
  %alloc = memref.alloc() : memref<2x3x4xf32, 3>
  memref.store %value, %alloc[%i0, %i1, %i2] : memref<2x3x4xf32, 3>
  %val = memref.load %alloc[%i0, %i1, %i2] : memref<2x3x4xf32, 3>
  return %val: f32
}

//      CHECK: #[[$MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 * 12 + s1 * 4 + s2)>
//CHECK-LABEL: func.func @load_store_alloc_static
// CHECK-SAME: (%[[VAL:.+]]: f32, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//      CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<24xf32, 3>
//      CHECK:   %[[INDEX0:.+]] = affine.apply #[[$MAP]]()[%[[I0]], %[[I1]], %[[I2]]]
//      CHECK:   memref.store %[[VAL]], %[[ALLOC]][%[[INDEX0]]] : memref<24xf32, 3>
//      CHECK:   %[[INDEX1:.+]] = affine.apply #[[$MAP]]()[%[[I0]], %[[I1]], %[[I2]]]
//      CHECK:   %[[LOAD:.+]] = memref.load %[[ALLOC]][%[[INDEX1]]] : memref<24xf32, 3>
//      CHECK:   return %[[LOAD]]

// -----

func.func @load_store_alloca_dynamic(%value : f32, %dim0 : index, %dim1: index, %dim2: index, %i0: index, %i1 : index, %i2: index) -> f32 {
  %alloc = memref.alloca(%dim0, %dim1, %dim2) : memref<?x?x?xf32>
  memref.store %value, %alloc[%i0, %i1, %i2] : memref<?x?x?xf32>
  %val = memref.load %alloc[%i0, %i1, %i2] : memref<?x?x?xf32>
  return %val: f32
}


//      CHECK: #[[$SIZE_MAP:.+]] = affine_map<()[s0, s1, s2] -> ((s0 * s1) * s2)>
//      CHECK: #[[$INDEX_MAP:.+]] = affine_map<()[s0, s1, s2, s3, s4] -> (s1 + (s4 + s2 * s3) * s0)>
// CHECK: func.func @load_store_alloca_dynamic
// CHECK-SAME: (%[[VAL:.+]]: f32, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index, %[[DIM2:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//      CHECK:   %[[SIZE:.+]] = affine.apply #[[$SIZE_MAP]]()[%[[DIM0]], %[[DIM1]], %[[DIM2]]]
//      CHECK:   %[[ALLOC:.+]] = memref.alloca(%[[SIZE]]) : memref<?xf32>
//      CHECK:   %[[INDEX0:.+]] = affine.apply #[[$INDEX_MAP]]()[%[[DIM2]], %[[I2]], %[[I0]], %[[DIM1]], %[[I1]]]
//      CHECK:   memref.store %[[VAL]], %[[ALLOC]][%[[INDEX0]]] : memref<?xf32>
//      CHECK:   %[[INDEX1:.+]] = affine.apply #[[$INDEX_MAP]]()[%[[DIM2]], %[[I2]], %[[I0]], %[[DIM1]], %[[I1]]]
//      CHECK:   %[[LOAD:.+]] = memref.load %[[ALLOC]][%[[INDEX1]]] : memref<?xf32>
//      CHECK:   return %[[LOAD]]

// -----

memref.global "private" constant @constant_3x3x1x1xf32 : memref<3x3x1x1xf32> = dense<[[[[-1.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]], [[[-2.000000e+00]], [[0.000000e+00]], [[2.000000e+00]]], [[[-1.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]]]>
func.func @load_global_with_offset(%i0: index, %i1: index, %i2: index, %i3: index) -> f32 {
  %global = memref.get_global @constant_3x3x1x1xf32 : memref<3x3x1x1xf32>
  %val = memref.load %global[%i0, %i1, %i2, %i3] : memref<3x3x1x1xf32>
  return %val: f32
}

//      CHECK: #[[$MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 3 + s1 + s2 + s3)>
//      CHECK: memref.global "private" constant @constant_3x3x1x1xf32 : memref<9xf32> = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00, -2.000000e+00, 0.000000e+00, 2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00]>
//CHECK-LABEL: func.func @load_global_with_offset
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index)
//      CHECK:   %[[GLOBAL:.+]] = memref.get_global @constant_3x3x1x1xf32 : memref<9xf32>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP]]()[%[[I0]], %[[I1]], %[[I2]], %[[I3]]]
//      CHECK:   %[[LOAD:.+]] = memref.load %[[GLOBAL]][%[[INDEX]]]
//      CHECK:   return %[[LOAD]]

// -----

func.func @transfer_read_subspan_with_offset(
    %arg0 : index, %arg1: index, %arg2: index, %arg3: index) -> vector<4xf32> {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%arg0) : memref<6x7x8xf32, strided<[56, 8, 1], offset:?>>
  %cst = arith.constant 0.0 : f32
  %val = vector.transfer_read %subspan[%arg1, %arg2, %arg3], %cst {in_bounds = [true]} : memref<6x7x8xf32, strided<[56, 8, 1], offset:?>>, vector<4xf32>
  return %val: vector<4xf32>
}

//  CHECK-DAG: #[[$MAP0:.+]] =  affine_map<()[s0] -> (s0 floordiv 4 + 336)>
//  CHECK-DAG: #[[$MAP1:.+]] =  affine_map<()[s0, s1, s2, s3] -> (s0 * 56 + s1 * 8 + s2 + s3 floordiv 4)>
//CHECK-LABEL: func.func @transfer_read_subspan_with_offset
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[SIZE:.+]] = affine.apply #[[$MAP0]]()[%[[ARG0]]]
//      CHECK:   %[[MEMREF:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP1]]()[%[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG0]]]
//      CHECK:   %[[VEC:.+]] = vector.transfer_read %[[MEMREF]][%[[INDEX]]]
//      CHECK:   return %[[VEC]]

// -----

func.func @transfer_write_subspan_with_offset(
    %arg0 : index, %arg1: index, %arg2: index, %arg3: index, %arg4 : vector<4xf32>) {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%arg0) : memref<6x7x8xf32, strided<[56, 8, 1], offset:?>>
  vector.transfer_write %arg4, %subspan[%arg1, %arg2, %arg3] {in_bounds = [true]} :  vector<4xf32>, memref<6x7x8xf32, strided<[56, 8, 1], offset:?>>
  return
}

//  CHECK-DAG: #[[$MAP0:.+]] =  affine_map<()[s0] -> (s0 floordiv 4 + 336)>
//  CHECK-DAG: #[[$MAP1:.+]] =  affine_map<()[s0, s1, s2, s3] -> (s0 * 56 + s1 * 8 + s2 + s3 floordiv 4)>
//CHECK-LABEL: func.func @transfer_write_subspan_with_offset
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: vector<4xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[SIZE:.+]] = affine.apply #[[$MAP0]]()[%[[ARG0]]]
//      CHECK:   %[[MEMREF:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP1]]()[%[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG0]]]
//      CHECK:   vector.transfer_write %[[ARG4]], %[[MEMREF]][%[[INDEX]]]

// -----

func.func @load_store_subspan_with_zero_offset(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index) {
  %subspan0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x?xf32>{%arg0, %arg1}
  %subspan1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x?xf32>{%arg0, %arg1}
  %val = memref.load %subspan0[%arg2, %arg3] : memref<?x?xf32>
  memref.store %val, %subspan1[%arg2, %arg3] : memref<?x?xf32>
  return
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>
// CHECK-LABEL: func @load_store_subspan_with_zero_offset(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
//       CHECK:  %[[C0:.+]] = arith.constant 0 : index
//       CHECK:  %[[D0:.+]] = affine.apply #[[$MAP0]]()[%[[ARG0]], %[[ARG1]]]
//       CHECK:  %[[BINDING0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[D0]]}
//       CHECK:  %[[D1:.+]] = affine.apply #[[$MAP0]]()[%[[ARG0]], %[[ARG1]]]
//       CHECK:  %[[BINDING1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[D1]]}
//       CHECK:  %[[OFFSET0:.+]] = affine.apply #[[$MAP1]]()[%[[ARG2]], %[[ARG1]], %[[ARG3]]]
//       CHECK:  %[[VAL:.+]] = memref.load %[[BINDING0]][%[[OFFSET0]]]
//       CHECK:  %[[OFFSET1:.+]] = affine.apply #[[$MAP1]]()[%[[ARG2]], %[[ARG1]], %[[ARG3]]]
//       CHECK:  memref.store %[[VAL]], %[[BINDING1]][%[[OFFSET1]]]

// -----

func.func @load_store_rank_zero_subspan_with_zero_offset() {
  %zero = arith.constant 0 : index
  %subspan0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%zero) : memref<f32>
  %subspan1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%zero) : memref<f32>
  %val = memref.load %subspan0[] : memref<f32>
  memref.store %val, %subspan1[] : memref<f32>
  return
}

//CHECK-LABEL: func.func @load_store_rank_zero_subspan_with_zero_offset
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[SPAN0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<f32>
//      CHECK:   %[[SPAN1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%[[C0]]) : memref<f32>

// -----

func.func @load_store_rank_zero_subspan_with_offset(%offset : index) {
  %subspan0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<f32, strided<[], offset:?>>
  %subspan1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%offset) : memref<f32, strided<[], offset:?>>
  %val = memref.load %subspan0[] : memref<f32, strided<[], offset:?>>
  memref.store %val, %subspan1[] : memref<f32, strided<[], offset:?>>
  return
}

//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 1)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 floordiv 4)>
//CHECK-LABEL: func.func @load_store_rank_zero_subspan_with_offset
// CHECK-SAME: (%[[OFFSET:.+]]: index)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[SIZE0:.+]] = affine.apply #[[$MAP0]]()[%[[OFFSET]]]
//      CHECK:   %[[SPAN0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[SIZE0]]}
//  CHECK-DAG:   %[[SIZE1:.+]] = affine.apply #[[$MAP0]]()[%[[OFFSET]]]
//      CHECK:   %[[SPAN1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[SIZE1]]}
//      CHECK:   %[[INDEX0:.+]] = affine.apply #[[$MAP1]]()[%[[OFFSET]]]
//      CHECK:   %[[LOAD:.+]] = memref.load %[[SPAN0]][%[[INDEX0]]] : memref<?xf32>
//      CHECK:   %[[INDEX1:.+]] = affine.apply #[[$MAP1]]()[%[[OFFSET]]]
//      CHECK:   memref.store %[[LOAD]], %[[SPAN1]][%[[INDEX1]]] : memref<?xf32>

// -----

func.func @collapse_shape(%offset : index, %i0 : index, %i1 : index) -> f32 {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<4x5x6x7xf32, strided<[210, 42, 7, 1], offset:?>>
  %collapse = memref.collapse_shape %subspan[[0, 1], [2, 3]] : memref<4x5x6x7xf32, strided<[210, 42, 7, 1], offset:?>> into memref<20x42xf32, strided<[42, 1], offset:?>>
  %value = memref.load %collapse[%i0, %i1] : memref<20x42xf32, strided<[42, 1], offset:?>>
  return %value : f32
}

//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 840)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1, s2] -> (s0 * 42 + s1 + s2 floordiv 4)>
//CHECK-LABEL: func.func @collapse_shape
// CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[SIZE:.+]] = affine.apply #[[$MAP0]]()[%[[OFFSET]]]
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP1]]()[%[[I0]], %[[I1]], %[[OFFSET]]]
//      CHECK:   memref.load %[[SUBSPAN]][%[[INDEX]]]

// -----

func.func @expand_shape(%offset : index, %i0: index, %i1: index, %i2: index, %i3: index) -> f32 {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<20x42xf32, strided<[42, 1], offset:?>>
  %expand = memref.expand_shape %subspan[[0, 1], [2, 3]] : memref<20x42xf32, strided<[42, 1], offset:?>> into memref<4x5x6x7xf32, strided<[210, 42, 7, 1], offset:?>>
  %value = memref.load %expand[%i0, %i1, %i2, %i3] : memref<4x5x6x7xf32, strided<[210, 42, 7, 1], offset:?>>
  return %value : f32
}

//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 840)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 210 + s1 * 42 + s2 * 7 + s3 + s4 floordiv 4)>
//CHECK-LABEL: func.func @expand_shape
// CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[SIZE:.+]] = affine.apply #[[$MAP0]]()[%[[OFFSET]]]
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP1]]()[%[[I0]], %[[I1]], %[[I2]], %[[I3]], %[[OFFSET]]]
//      CHECK:   memref.load %[[SUBSPAN]][%[[INDEX]]]

// -----

func.func @expand_shape2(%offset : index, %i0: index, %i1: index) -> f32 {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<128xf32, strided<[1], offset: ?>>
  %expand = memref.expand_shape %subspan [[0, 1]] : memref<128xf32, strided<[1], offset: ?>> into memref<1x128xf32, strided<[128, 1], offset: ?>>
  %value = memref.load %expand[%i0, %i1] : memref<1x128xf32, strided<[128, 1], offset: ?>>
  return %value : f32
}

//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 128)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1, s2] -> (s0 * 128 + s1 + s2 floordiv 4)>
//CHECK-LABEL: func.func @expand_shape2
// CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[SIZE:.+]] = affine.apply #[[$MAP0]]()[%[[OFFSET]]]
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP1]]()[%[[I0]], %[[I1]], %[[OFFSET]]]
//      CHECK:   memref.load %[[SUBSPAN]][%[[INDEX]]]

// -----

// An opaque consumer that already takes a collapsed, static 1d memref should
// be able to do so (a memref cast is inserted to move between unknown and
// known dim).
func.func @static_collapse_shape_to_1d_static(%offset : index, %i: index) {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<6x7x8xf32, strided<[56, 8, 1], offset:?>>
  %collapse = memref.collapse_shape %subspan [[0, 1, 2]] : memref<6x7x8xf32, strided<[56, 8, 1], offset:?>> into memref<336xf32, strided<[1], offset: ?>>
  "unregistered.opaque"(%collapse) : (memref<336xf32, strided<[1], offset: ?>>) -> ()
}
//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 4)
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 336)
// CHECK-LABEL: func.func @static_collapse_shape_to_1d_static(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[OFFSET:.+]] = affine.apply #[[$MAP0]]()[%[[ARG0]]
//   CHECK-DAG:   %[[SIZE:.+]] = affine.apply #[[$MAP1]]()[%[[ARG0]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[SIZE]]}
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[SUBSPAN]][%[[OFFSET]]] [336] [1] : memref<?xf32> to memref<336xf32, strided<[1], offset: ?>>
//       CHECK:   "unregistered.opaque"(%[[SUBVIEW]])

// -----

func.func @subview(%offset : index, %i0: index, %i1: index) -> f32 {
  %c0 = arith.constant 0 : index
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<32x128xf32, strided<[128, 1], offset: ?>>
  %expand = memref.subview %subspan[%i0, %i1][16, 8][1, 1] : memref<32x128xf32, strided<[128, 1], offset: ?>> to memref<16x8xf32, strided<[128, 1], offset: ?>>
  %value = memref.load %expand[%c0, %c0] : memref<16x8xf32, strided<[128, 1], offset: ?>>
  return %value : f32
}

//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 4096)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1, s2] -> (s0 * 128 + s1 + s2 floordiv 4)>
//CHECK-LABEL: func.func @subview
// CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[SIZE:.+]] = affine.apply #[[$MAP0]]()[%[[OFFSET]]]
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP1]]()[%[[I0]], %[[I1]], %[[OFFSET]]]
//      CHECK:   memref.load %[[SUBSPAN]][%[[INDEX]]]

// -----

func.func @subgroup_mma_load(%i0: index, %i1: index) -> !gpu.mma_matrix<16x16xf16, "AOp"> {
  %alloc = memref.alloc() : memref<32x32xf16, 3>
  %0 = gpu.subgroup_mma_load_matrix %alloc[%i0, %i1] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
  return %0 : !gpu.mma_matrix<16x16xf16, "AOp">
}

//       CHECK: #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 32 + s1)>
// CHECK-LABEL: func.func @subgroup_mma_load
//  CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
//       CHECK:  %[[ALLOC:.+]] = memref.alloc() : memref<1024xf16, 3>
//       CHECK:  %[[IDX:.+]] = affine.apply #[[$MAP]]()[%[[I0]], %[[I1]]]
//       CHECK:  %[[LD:.+]] = gpu.subgroup_mma_load_matrix %[[ALLOC]][%[[IDX]]] {leadDimension = 32 : index} : memref<1024xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
//       CHECK:  return %[[LD]]

// -----

func.func @subgroup_mma_store(%i0: index, %i1: index, %val: !gpu.mma_matrix<16x16xf16, "COp">) {
  %alloc = memref.alloc() : memref<32x32xf16, 3>
  gpu.subgroup_mma_store_matrix %val, %alloc[%i0, %i1] {leadDimension = 128 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, 3>
  return
}

//       CHECK: #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 32 + s1)>
// CHECK-LABEL: func.func @subgroup_mma_store
//  CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index, %[[VAL:.+]]: !gpu.mma_matrix<16x16xf16, "COp">) {
//       CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<1024xf16, 3>
//       CHECK:   %[[IDX:.+]] = affine.apply #[[$MAP]]()[%[[I0]], %[[I1]]]
//       CHECK:   gpu.subgroup_mma_store_matrix %[[VAL]], %[[ALLOC]][%[[IDX]]] {leadDimension = 128 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1024xf16, 3>

// -----

func.func @subgroup_mma_load_with_offset(%offset : index, %i0: index, %i1: index) -> !gpu.mma_matrix<16x16xf16, "AOp"> {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<32x32xf16, strided<[32, 1], offset: ?>, 3>
  %0 = gpu.subgroup_mma_load_matrix %subspan[%i0, %i1] {leadDimension = 32 : index} : memref<32x32xf16, strided<[32, 1], offset: ?>, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
  return %0 : !gpu.mma_matrix<16x16xf16, "AOp">
}

//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 2)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 floordiv 2 + 1024)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0, s1] -> (s0 * 32 + s1)>
// CHECK-LABEL: func.func @subgroup_mma_load_with_offset
//  CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index)
//   CHECK-DAG:   %[[ZERO:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[SUBVIEW_OFFSET:.+]] = affine.apply #[[$MAP0]]()[%[[OFFSET]]]
//   CHECK-DAG:   %[[SIZE:.+]] = affine.apply #[[$MAP1]]()[%[[OFFSET]]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[ZERO]]) : memref<?xf16, 3>{%[[SIZE]]}
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[SUBSPAN]][%[[SUBVIEW_OFFSET]]] [1024] [1]
//       CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP2]]()[%[[I0]], %[[I1]]]
//       CHECK:  %[[LD:.+]] = gpu.subgroup_mma_load_matrix %[[SUBVIEW]][%[[INDEX]]] {leadDimension = 32 : index}
//       CHECK:  return %[[LD]]

// -----

func.func @subgroup_mma_store_with_offset(%offset : index, %i0: index, %i1: index, %val: !gpu.mma_matrix<16x16xf16, "COp">) {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%offset) : memref<32x32xf16, strided<[32, 1], offset: ?>, 3>
  gpu.subgroup_mma_store_matrix %val, %subspan[%i0, %i1] {leadDimension = 128 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, strided<[32, 1], offset: ?>, 3>
  return
}

//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 2)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 floordiv 2 + 1024)>
//   CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0, s1] -> (s0 * 32 + s1)>
// CHECK-LABEL: func.func @subgroup_mma_store_with_offset
//  CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[VAL:.+]]: !gpu.mma_matrix<16x16xf16, "COp">
//   CHECK-DAG:   %[[ZERO:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[SUBVIEW_OFFSET:.+]] = affine.apply #[[$MAP0]]()[%[[OFFSET]]]
//   CHECK-DAG:   %[[SIZE:.+]] = affine.apply #[[$MAP1]]()[%[[OFFSET]]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[ZERO]]) : memref<?xf16, 3>{%[[SIZE]]}
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[SUBSPAN]][%[[SUBVIEW_OFFSET]]] [1024] [1]
//       CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP2]]()[%[[I0]], %[[I1]]]
//       CHECK:   gpu.subgroup_mma_store_matrix %[[VAL]], %[[SUBVIEW]][%[[INDEX]]] {leadDimension = 128 : index}

// -----

func.func @load_uniform_buffer(%offset: index, %i0: index, %i1 : index, %i2: index) -> i32 {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(uniform_buffer) offset(%offset) : memref<2x3x4xi32, strided<[12, 4, 1], offset:?>, #hal.descriptor_type<uniform_buffer>>
  %val = memref.load %subspan[%i0, %i1, %i2] : memref<2x3x4xi32, strided<[12, 4, 1], offset:?>, #hal.descriptor_type<uniform_buffer>>
  return %val: i32
}

//       CHECK: #[[$MAP1:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 12 + s1 * 4 + s2 + s3 floordiv 4)>
// CHECK-LABEL: func.func @load_uniform_buffer
//  CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(uniform_buffer) offset(%[[C0]]) : memref<?xi32, #hal.descriptor_type<uniform_buffer>>
//       CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP1]]()[%[[I0]], %[[I1]], %[[I2]], %[[OFFSET]]]
//       CHECK:   %[[LD:.+]] = memref.load %[[SUBSPAN]][%[[INDEX]]] : memref<?xi32, #hal.descriptor_type<uniform_buffer>>
//       CHECK:   return %[[LD]] : i32


// -----

func.func @store_uniform_buffer(%value : i32, %offset: index, %i0: index, %i1 : index, %i2: index) {
  %subspan = hal.interface.binding.subspan set(0) binding(0) type(uniform_buffer) offset(%offset) : memref<2x3x4xi32, strided<[12, 4, 1], offset:?>, #hal.descriptor_type<uniform_buffer>>
  memref.store %value, %subspan[%i0, %i1, %i2] : memref<2x3x4xi32, strided<[12, 4, 1], offset:?>, #hal.descriptor_type<uniform_buffer>>
  return
}

//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 floordiv 4 + 24)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 12 + s1 * 4 + s2 + s3 floordiv 4)>
// CHECK-LABEL: func.func @store_uniform_buffer
//  CHECK-SAME: (%[[VAL:.+]]: i32, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[SIZE:.+]] = affine.apply #[[$MAP0]]()[%[[OFFSET]]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan set(0) binding(0) type(uniform_buffer) offset(%[[C0]]) : memref<?xi32, #hal.descriptor_type<uniform_buffer>>{%[[SIZE]]}
//       CHECK:   %[[INDEX:.+]] = affine.apply #[[$MAP1]]()[%[[I0]], %[[I1]], %[[I2]], %[[OFFSET]]]
//       CHECK:   memref.store %[[VAL]], %[[SUBSPAN]][%[[INDEX]]] : memref<?xi32, #hal.descriptor_type<uniform_buffer>>
