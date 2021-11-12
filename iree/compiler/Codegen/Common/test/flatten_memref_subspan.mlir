// RUN: iree-opt -split-input-file -iree-codegen-flatten-memref-subspan -canonicalize %s | IreeFileCheck %s

func @load_subspan_with_offset(%offset : index, %i0: index, %i1: index, %i2: index) -> f32 {
  %subspan = hal.interface.binding.subspan @io::@s0b0_ro_constant[%offset] : memref<6x7x8xf32>
  %val = memref.load %subspan[%i0, %i1, %i2] : memref<6x7x8xf32>
  return %val: f32
}

hal.interface private @io  {
  hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 56 + s1 * 8 + s2 + s3 floordiv 4)>
//      CHECK: func @load_subspan_with_offset
// CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//  CHECK-DAG:   %[[ZERO:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C336:.+]] = arith.constant 336 : index
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_constant[%[[ZERO]]] : memref<?xf32>{%[[C336]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[I0]], %[[I1]], %[[I2]], %[[OFFSET]]]
//      CHECK:   %[[LOAD:.+]] = memref.load %[[SUBSPAN]][%[[INDEX]]]
//      CHECK:   return %[[LOAD]]

// -----

func @store_subspan_with_offset(%value: f32, %offset : index, %i0: index, %i1: index, %i2: index) {
  %subspan = hal.interface.binding.subspan @io::@s0b0_xw_external[%offset] : memref<2x3x4xf32>
  memref.store %value, %subspan[%i0, %i1, %i2] : memref<2x3x4xf32>
  return
}

hal.interface private @io  {
  hal.interface.binding @s0b0_xw_external, set=0, binding=0, type="StorageBuffer", access="Write|Discard"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 12 + s1 * 4 + s2 + s3 floordiv 4)>
//      CHECK: func @store_subspan_with_offset
// CHECK-SAME: (%[[VALUE:.+]]: f32, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//  CHECK-DAG:   %[[ZERO:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C24:.+]] = arith.constant 24 : index
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan @io::@s0b0_xw_external[%[[ZERO]]] : memref<?xf32>{%[[C24]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[I0]], %[[I1]], %[[I2]], %[[OFFSET]]]
//      CHECK:   memref.store %[[VALUE]], %[[SUBSPAN]][%[[INDEX]]] : memref<?xf32>

// -----

func @load_subspan_with_vector_element(%offset : index, %i0: index, %i1: index, %i2: index) -> vector<4xf32> {
  %subspan = hal.interface.binding.subspan @io::@s0b0_ro_constant[%offset] : memref<6x7x8xvector<4xf32>>
  %val = memref.load %subspan[%i0, %i1, %i2] : memref<6x7x8xvector<4xf32>>
  return %val: vector<4xf32>
}

hal.interface private @io  {
  hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 56 + s1 * 8 + s2 + s3 floordiv 16)>
//      CHECK: func @load_subspan_with_vector_element
//      CHECK:   affine.apply #[[MAP]]()

// -----

func @load_subspan_with_16bit_element(%offset : index, %i0: index, %i1: index, %i2: index) -> f16 {
  %subspan = hal.interface.binding.subspan @io::@s0b0_ro_constant[%offset] : memref<6x7x8xf16>
  %val = memref.load %subspan[%i0, %i1, %i2] : memref<6x7x8xf16>
  return %val: f16
}

hal.interface private @io  {
  hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 56 + s1 * 8 + s2 + s3 floordiv 2)>
//      CHECK: func @load_subspan_with_16bit_element
//      CHECK:   affine.apply #[[MAP]]()

// -----

func @store_subspan_with_leading_dynamic_dim(%value: f32, %offset : index, %i0: index, %i1: index, %i2: index) {
  %dim = hal.interface.load.constant offset = 0 : index
  %subspan = hal.interface.binding.subspan @io::@s0b0_xw_external[%offset] : memref<?x3x4xf32>{%dim}
  memref.store %value, %subspan[%i0, %i1, %i2] : memref<?x3x4xf32>
  return
}

hal.interface private @io  {
  hal.interface.binding @s0b0_xw_external, set=0, binding=0, type="StorageBuffer", access="Write|Discard"
}

//      CHECK: #[[SIZE_MAP:.+]] = affine_map<()[s0] -> (s0 * 12)
//      CHECK: #[[OFFSET_MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 12 + s1 * 4 + s2 + s3 floordiv 4)>
//      CHECK: func @store_subspan_with_leading_dynamic_dim
// CHECK-SAME: (%[[VALUE:.+]]: f32, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM:.+]] = hal.interface.load.constant offset = 0 : index
//      CHECK:   %[[SIZE:.+]] = affine.apply #[[SIZE_MAP]]()[%[[DIM]]]
//      CHECK:   %[[DST:.+]] = hal.interface.binding.subspan @io::@s0b0_xw_external[%[[C0]]] : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[OFFSET_MAP]]()[%[[I0]], %[[I1]], %[[I2]], %[[OFFSET]]]
//      CHECK:   memref.store %[[VALUE]], %[[DST]][%[[INDEX]]] : memref<?xf32>

// -----

func @store_subspan_with_all_dynamic_dim(%value: f32, %offset : index, %i0: index, %i1: index, %i2: index, %i3: index) {
  %dim0 = hal.interface.load.constant offset = 0 : index
  %dim1 = hal.interface.load.constant offset = 1 : index
  %dim2 = hal.interface.load.constant offset = 2 : index
  %dim3 = hal.interface.load.constant offset = 3 : index
  %subspan = hal.interface.binding.subspan @io::@s0b0_xw_external[%offset] : memref<?x?x?x?xf32>{%dim0, %dim1, %dim2, %dim3}
  memref.store %value, %subspan[%i0, %i1, %i2, %i3] : memref<?x?x?x?xf32>
  return
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @s0b0_xw_external, set=0, binding=0, type="StorageBuffer", access="Write|Discard"
}

//      CHECK: #[[SIZE_MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (((s0 * s1) * s2) * s3)>
//      CHECK: #[[OFFSET_MAP:.+]] = affine_map<()[s0, s1, s2, s3, s4, s5, s6, s7] -> (((s4 * s5 + s6) * s2 + s3) * s0 + s1 + s7 floordiv 4)>
//      CHECK: func @store_subspan_with_all_dynamic_dim
// CHECK-SAME: (%[[VALUE:.+]]: f32, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index)
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM0:.+]] = hal.interface.load.constant offset = 0 : index
//      CHECK:   %[[DIM1:.+]] = hal.interface.load.constant offset = 1 : index
//      CHECK:   %[[DIM2:.+]] = hal.interface.load.constant offset = 2 : index
//      CHECK:   %[[DIM3:.+]] = hal.interface.load.constant offset = 3 : index
//      CHECK:   %[[SIZE:.+]] = affine.apply #[[SIZE_MAP]]()[%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]]]
//      CHECK:   %[[DST:.+]] = hal.interface.binding.subspan @io::@s0b0_xw_external[%[[C0]]] : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[OFFSET_MAP]]()[%[[DIM3]], %[[I3]], %[[DIM2]], %[[I2]], %[[I0]], %[[DIM1]], %[[I1]], %[[OFFSET]]]
//      CHECK:   memref.store %[[VALUE]], %[[DST]][%[[INDEX]]]

// -----

func @store_subspan_with_mixed_dynamic_dim(%value: f32, %offset : index, %i0: index, %i1: index, %i2: index, %i3: index) {
  %dim0 = hal.interface.load.constant offset = 0 : index
  %dim1 = hal.interface.load.constant offset = 1 : index
  %subspan = hal.interface.binding.subspan @io::@s0b0_xw_external[%offset] : memref<?x4x?x8xf32>{%dim0, %dim1}
  memref.store %value, %subspan[%i0, %i1, %i2, %i3] : memref<?x4x?x8xf32>
  return
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @s0b0_xw_external, set=0, binding=0, type="StorageBuffer", access="Write|Discard"
}

//      CHECK: #[[SIZE_MAP:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) * 32)>
//      CHECK: #[[OFFSET_MAP:.+]] = affine_map<()[s0, s1, s2, s3, s4, s5] -> (((s3 * 4 + s4) * s1 + s2) * 8 + s0 + s5 floordiv 4)>
//      CHECK: func @store_subspan_with_mixed_dynamic_dim
// CHECK-SAME: (%[[VALUE:.+]]: f32, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index)
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM0:.+]] = hal.interface.load.constant offset = 0 : index
//      CHECK:   %[[DIM2:.+]] = hal.interface.load.constant offset = 1 : index
//      CHECK:   %[[SIZE:.+]] = affine.apply #[[SIZE_MAP]]()[%[[DIM0]], %[[DIM2]]]
//      CHECK:   %[[DST:.+]] = hal.interface.binding.subspan @io::@s0b0_xw_external[%[[C0]]] : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[OFFSET_MAP]]()[%[[I3]], %[[DIM2]], %[[I2]], %[[I0]], %[[I1]], %[[OFFSET]]]
//      CHECK:   memref.store %[[VALUE]], %[[DST]][%[[INDEX]]]

// -----

func @store_subspan_with_flow_control(%value: f32, %offset : index, %i0: index, %i1: index, %i2: index) {
  %dim = hal.interface.load.constant offset = 0 : index
  %subspan = hal.interface.binding.subspan @io::@s0b0_xw_external[%offset] : memref<?x3x4xf32>{%dim}
  scf.for %i = %i0 to %i1 step %i2 {
    memref.store %value, %subspan[%i0, %i1, %i2] : memref<?x3x4xf32>
  }
  return
}

hal.interface private @io  {
  hal.interface.binding @s0b0_xw_external, set=0, binding=0, type="StorageBuffer", access="Write|Discard"
}

//      CHECK: #[[SIZE_MAP:.+]] = affine_map<()[s0] -> (s0 * 12)
//      CHECK: #[[OFFSET_MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 12 + s1 * 4 + s2 + s3 floordiv 4)>
//      CHECK: func @store_subspan_with_flow_control
// CHECK-SAME: (%[[VALUE:.+]]: f32, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM:.+]] = hal.interface.load.constant offset = 0 : index
//      CHECK:   %[[SIZE:.+]] = affine.apply #[[SIZE_MAP]]()[%[[DIM]]]
//      CHECK:   %[[DST:.+]] = hal.interface.binding.subspan @io::@s0b0_xw_external[%[[C0]]] : memref<?xf32>{%[[SIZE]]}
//      CHECK: scf.for
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[OFFSET_MAP]]()[%[[I0]], %[[I1]], %[[I2]], %[[OFFSET]]]
//      CHECK:   memref.store %[[VALUE]], %[[DST]][%[[INDEX]]] : memref<?xf32>

// -----

func @load_store_alloc_static(%value : f32, %i0: index, %i1 : index, %i2: index) -> f32 {
  %alloc = memref.alloc() : memref<2x3x4xf32, 3>
  memref.store %value, %alloc[%i0, %i1, %i2] : memref<2x3x4xf32, 3>
  %val = memref.load %alloc[%i0, %i1, %i2] : memref<2x3x4xf32, 3>
  return %val: f32
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 * 12 + s1 * 4 + s2)>
//      CHECK: func @load_store_alloc_static
// CHECK-SAME: (%[[VAL:.+]]: f32, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//      CHECK:   %[[ALLOC:.+]] = memref.alloc() : memref<24xf32, 3>
//      CHECK:   %[[INDEX0:.+]] = affine.apply #[[MAP]]()[%[[I0]], %[[I1]], %[[I2]]]
//      CHECK:   memref.store %[[VAL]], %[[ALLOC]][%[[INDEX0]]] : memref<24xf32, 3>
//      CHECK:   %[[INDEX1:.+]] = affine.apply #[[MAP]]()[%[[I0]], %[[I1]], %[[I2]]]
//      CHECK:   %[[LOAD:.+]] = memref.load %0[%[[INDEX1]]] : memref<24xf32, 3>
//      CHECK:   return %[[LOAD]]

// -----

func @load_store_alloca_dynamic(%value : f32, %dim0 : index, %dim1: index, %dim2: index, %i0: index, %i1 : index, %i2: index) -> f32 {
  %alloc = memref.alloca(%dim0, %dim1, %dim2) : memref<?x?x?xf32>
  memref.store %value, %alloc[%i0, %i1, %i2] : memref<?x?x?xf32>
  %val = memref.load %alloc[%i0, %i1, %i2] : memref<?x?x?xf32>
  return %val: f32
}


//      CHECK: #[[SIZE_MAP:.+]] = affine_map<()[s0, s1, s2] -> ((s0 * s1) * s2)>
//      CHECK: #[[INDEX_MAP:.+]] = affine_map<()[s0, s1, s2, s3, s4] -> ((s2 * s3 + s4) * s0 + s1)>
// CHECK: func @load_store_alloca_dynamic
// CHECK-SAME: (%[[VAL:.+]]: f32, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index, %[[DIM2:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//      CHECK:   %[[SIZE:.+]] = affine.apply #[[SIZE_MAP]]()[%[[DIM0]], %[[DIM1]], %[[DIM2]]]
//      CHECK:   %[[ALLOC:.+]] = memref.alloca(%[[SIZE]]) : memref<?xf32>
//      CHECK:   %[[INDEX0:.+]] = affine.apply #[[INDEX_MAP]]()[%[[DIM2]], %[[I2]], %[[I0]], %[[DIM1]], %[[I1]]]
//      CHECK:   memref.store %[[VAL]], %[[ALLOC]][%[[INDEX0]]] : memref<?xf32>
//      CHECK:   %[[INDEX1:.+]] = affine.apply #[[INDEX_MAP]]()[%[[DIM2]], %[[I2]], %[[I0]], %[[DIM1]], %[[I1]]]
//      CHECK:   %[[LOAD:.+]] = memref.load %[[ALLOC]][%[[INDEX1]]] : memref<?xf32>
//      CHECK:   return %[[LOAD]]

// -----

func @use_subspan_with_unrealized_conversion_cast(%offset : index, %i: index) -> f32 {
  %subspan = hal.interface.binding.subspan @io::@s0b0_ro_constant[%offset] : memref<6x7x8xf32>
  %use = builtin.unrealized_conversion_cast %subspan : memref<6x7x8xf32> to memref<?xf32>
  %val = memref.load %use[%i] : memref<?xf32>
  return %val: f32
}

hal.interface private @io  {
  hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1 floordiv 4)>
//      CHECK: func @use_subspan_with_unrealized_conversion_cast
// CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I:.+]]: index)
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_constant[%[[C0]]] : memref<?xf32>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[I]], %[[OFFSET]]]
//      CHECK:   memref.load %[[SUBSPAN]][%[[INDEX]]]

// -----

memref.global "private" constant @constant_3x3x1x1xf32 : memref<3x3x1x1xf32> = dense<[[[[-1.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]], [[[-2.000000e+00]], [[0.000000e+00]], [[2.000000e+00]]], [[[-1.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]]]>
func @load_global_with_offset(%i0: index, %i1: index, %i2: index, %i3: index) -> f32 {
  %global = memref.get_global @constant_3x3x1x1xf32 : memref<3x3x1x1xf32>
  %val = memref.load %global[%i0, %i1, %i2, %i3] : memref<3x3x1x1xf32>
  return %val: f32
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 * 3 + s1 + s2 + s3)>
//      CHECK: memref.global "private" constant @constant_3x3x1x1xf32 : memref<9xf32> = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00, -2.000000e+00, 0.000000e+00, 2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00]>
//      CHECK: func @load_global_with_offset
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index)
//      CHECK:   %[[GLOBAL:.+]] = memref.get_global @constant_3x3x1x1xf32 : memref<9xf32>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[I0]], %[[I1]], %[[I2]], %[[I3]]]
//      CHECK:   %[[LOAD:.+]] = memref.load %[[GLOBAL]][%[[INDEX]]]
//      CHECK:   return %[[LOAD]]

// -----

func @transfer_read_subspan_with_offset(
    %arg0 : index, %arg1: index, %arg2: index, %arg3: index) -> vector<4xf32> {
  %subspan = hal.interface.binding.subspan @io::@ro[%arg0] : memref<6x7x8xf32>
  %cst = arith.constant 0.0 : f32
  %val = vector.transfer_read %subspan[%arg1, %arg2, %arg3], %cst {in_bounds = [true]} : memref<6x7x8xf32>, vector<4xf32>
  return %val: vector<4xf32>
}
hal.interface private @io  {
  hal.interface.binding @ro, set=0, binding=0, type="StorageBuffer", access="Read"
}
//      CHECK: #[[MAP:.+]] =  affine_map<()[s0, s1, s2] -> (s0 * 56 + s1 * 8 + s2)>
//      CHECK: func @transfer_read_subspan_with_offset
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
//      CHECK:   %[[MEMREF:.+]] = hal.interface.binding.subspan @io::@ro[%[[ARG0]]] : memref<?xf32>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]], %[[ARG3]]]
//      CHECK:   %[[VEC:.+]] = vector.transfer_read %[[MEMREF]][%[[INDEX]]]
//      CHECK:   return %[[VEC]]

// -----

func @transfer_write_subspan_with_offset(
    %arg0 : index, %arg1: index, %arg2: index, %arg3: index, %arg4 : vector<4xf32>) {
  %subspan = hal.interface.binding.subspan @io::@ro[%arg0] : memref<6x7x8xf32>
  vector.transfer_write %arg4, %subspan[%arg1, %arg2, %arg3] {in_bounds = [true]} :  vector<4xf32>, memref<6x7x8xf32>
  return
}
hal.interface private @io  {
  hal.interface.binding @ro, set=0, binding=0, type="StorageBuffer", access="Read|Write"
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 * 56 + s1 * 8 + s2)>
//      CHECK: func @transfer_write_subspan_with_offset
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG4:[a-zA-Z0-9_]+]]: vector<4xf32>
//      CHECK:   %[[MEMREF:.+]] = hal.interface.binding.subspan @io::@ro[%[[ARG0]]] : memref<?xf32>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]], %[[ARG3]]]
//      CHECK:   vector.transfer_write %[[ARG4]], %[[MEMREF]][%[[INDEX]]]

// -----

func @load_store_rank_zero_subspan_with_offset(%offset : index) {
  %subspan0 = hal.interface.binding.subspan @io::@s0b0_ro_constant[%offset] : memref<f32>
  %subspan1 = hal.interface.binding.subspan @io::@s0b1_xw_external[%offset] : memref<f32>
  %val = memref.load %subspan0[] : memref<f32>
  memref.store %val, %subspan1[] : memref<f32>
  return
}

hal.interface private @io  {
  hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @s0b1_xw_external, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 4)>
//      CHECK: func @load_store_rank_zero_subspan_with_offset
// CHECK-SAME: (%[[OFFSET:.+]]: index)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[SPAN0:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_constant[%[[C0]]] : memref<?xf32>{%[[C1]]}
//      CHECK:   %[[SPAN1:.+]] = hal.interface.binding.subspan @io::@s0b1_xw_external[%[[C0]]] : memref<?xf32>{%[[C1]]}
//      CHECK:   %[[INDEX0:.+]] = affine.apply #[[MAP]]()[%[[OFFSET]]]
//      CHECK:   %[[LOAD:.+]] = memref.load %[[SPAN0]][%[[INDEX0]]] : memref<?xf32>
//      CHECK:   %[[INDEX1:.+]] = affine.apply #[[MAP]]()[%[[OFFSET]]]
//      CHECK:   memref.store %[[LOAD]], %[[SPAN1]][%[[INDEX1]]] : memref<?xf32>

// -----

func @collapse_shape(%offset : index, %i0 : index, %i1 : index) -> f32 {
  %subspan = hal.interface.binding.subspan @io::@s0b0_ro_constant[%offset] : memref<4x5x6x7xf32>
  %collapse = memref.collapse_shape %subspan[[0, 1], [2, 3]] : memref<4x5x6x7xf32> into memref<20x42xf32>
  %value = memref.load %collapse[%i0, %i1] : memref<20x42xf32>
  return %value : f32
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 * 42 + s1 + s2 floordiv 4)>
//      CHECK: func @collapse_shape
// CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index)
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[SIZE:.+]] = arith.constant 840 : index
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_constant[%[[C0]]] : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[I0]], %[[I1]], %[[OFFSET]]]
//      CHECK:   memref.load %[[SUBSPAN]][%[[INDEX]]]

// -----

func @expand_shape(%offset : index, %i0: index, %i1: index, %i2: index, %i3: index) -> f32 {
  %subspan = hal.interface.binding.subspan @io::@s0b0_ro_constant[%offset] : memref<20x42xf32>
  %expand = memref.expand_shape %subspan[[0, 1], [2, 3]] : memref<20x42xf32> into memref<4x5x6x7xf32>
  %value = memref.load %expand[%i0, %i1, %i2, %i3] : memref<4x5x6x7xf32>
  return %value : f32
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * 210 + s1 * 42 + s2 * 7 + s3 + s4 floordiv 4)>
//      CHECK: func @expand_shape
// CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index)
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[SIZE:.+]] = arith.constant 840 : index
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_constant[%[[C0]]] : memref<?xf32>{%[[SIZE]]}
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[I0]], %[[I1]], %[[I2]], %[[I3]], %[[OFFSET]]]
//      CHECK:   memref.load %[[SUBSPAN]][%[[INDEX]]]
