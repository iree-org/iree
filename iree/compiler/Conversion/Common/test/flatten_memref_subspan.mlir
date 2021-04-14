// RUN: iree-opt -split-input-file -iree-codegen-flatten-memref-subspan -canonicalize %s | IreeFileCheck %s

func @load_subspan_with_offset(%offset : index, %i0: index, %i1: index, %i2: index) -> f32 {
  %subspan = hal.interface.binding.subspan @io::@s0b0_ro_constant[%offset] : memref<6x7x8xf32>
  %val = memref.load %subspan[%i0, %i1, %i2] : memref<6x7x8xf32>
  return %val: f32
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 + s2 * 8 + s3 * 56 + s1 floordiv 4)>
//      CHECK: func @load_subspan_with_offset
// CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//      CHECK:   %[[ZERO:.+]] = constant 0 : index
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_constant[%[[ZERO]]] : memref<?xf32>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[I2]], %[[OFFSET]], %[[I1]], %[[I0]]]
//      CHECK:   %[[LOAD:.+]] = memref.load %[[SUBSPAN]][%[[INDEX]]]
//      CHECK:   return %[[LOAD]]

// -----

func @store_subspan_with_offset(%value: f32, %offset : index, %i0: index, %i1: index, %i2: index) {
  %subspan = hal.interface.binding.subspan @io::@s0b0_xw_external[%offset] : memref<2x3x4xf32>
  memref.store %value, %subspan[%i0, %i1, %i2] : memref<2x3x4xf32>
  return
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @s0b0_xw_external, set=0, binding=0, type="StorageBuffer", access="Write|Discard"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 + s2 * 4 + s3 * 12 + s1 floordiv 4)>
//      CHECK: func @store_subspan_with_offset
// CHECK-SAME: (%[[VALUE:.+]]: f32, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
//      CHECK:   %[[ZERO:.+]] = constant 0 : index
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan @io::@s0b0_xw_external[%[[ZERO]]] : memref<?xf32>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[I2]], %[[OFFSET]], %[[I1]], %[[I0]]]
//      CHECK:   memref.store %[[VALUE]], %[[SUBSPAN]][%[[INDEX]]] : memref<?xf32>

// -----

func @load_subspan_with_vector_element(%offset : index, %i0: index, %i1: index, %i2: index) -> vector<4xf32> {
  %subspan = hal.interface.binding.subspan @io::@s0b0_ro_constant[%offset] : memref<6x7x8xvector<4xf32>>
  %val = memref.load %subspan[%i0, %i1, %i2] : memref<6x7x8xvector<4xf32>>
  return %val: vector<4xf32>
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 + s2 * 8 + s3 * 56 + s1 floordiv 16)>
//      CHECK: func @load_subspan_with_vector_element
//      CHECK:   affine.apply #[[MAP]]()

// -----

func @load_subspan_with_16bit_element(%offset : index, %i0: index, %i1: index, %i2: index) -> f16 {
  %subspan = hal.interface.binding.subspan @io::@s0b0_ro_constant[%offset] : memref<6x7x8xf16>
  %val = memref.load %subspan[%i0, %i1, %i2] : memref<6x7x8xf16>
  return %val: f16
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 + s2 * 8 + s3 * 56 + s1 floordiv 2)>
//      CHECK: func @load_subspan_with_16bit_element
//      CHECK:   affine.apply #[[MAP]]()

// -----

func @store_subspan_with_leading_unknown_dim(%value: f32, %offset : index, %i0: index, %i1: index, %i2: index) {
  %subspan = hal.interface.binding.subspan @io::@s0b0_xw_external[%offset] : memref<?x3x4xf32>
  memref.store %value, %subspan[%i0, %i1, %i2] : memref<?x3x4xf32>
  return
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @s0b0_xw_external, set=0, binding=0, type="StorageBuffer", access="Write|Discard"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2, s3] -> (s0 + s2 * 4 + s3 * 12 + s1 floordiv 4)>
//      CHECK: func @store_subspan_with_leading_unknown_dim
//      CHECK:   affine.apply #[[MAP]]()

// -----

func @ignore_load_store_alloc(%value : f32, %i0: index, %i1 : index, %i2: index) -> f32 {
  %alloc = memref.alloc() : memref<2x3x4xf32, 3>
  memref.store %value, %alloc[%i0, %i1, %i2] : memref<2x3x4xf32, 3>
  %val = memref.load %alloc[%i0, %i1, %i2] : memref<2x3x4xf32, 3>
  return %val: f32
}

// CHECK-LABEL: func @ignore_load_store_alloc
//       CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<2x3x4xf32, 3>
//       CHECK: memref.store %{{[a-z0-9]+}}, %[[ALLOC]]
//       CHECK: memref.load %[[ALLOC]]

// -----

func @use_subspan_with_unrealized_conversion_cast(%offset : index, %i: index) -> f32 {
  %subspan = hal.interface.binding.subspan @io::@s0b0_ro_constant[%offset] : memref<6x7x8xf32>
  %use = unrealized_conversion_cast %subspan : memref<6x7x8xf32> to memref<?xf32>
  %val = memref.load %use[%i] : memref<?xf32>
  return %val: f32
}

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
}

//      CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1 floordiv 4)>
//      CHECK: func @use_subspan_with_unrealized_conversion_cast
// CHECK-SAME: (%[[OFFSET:.+]]: index, %[[I:.+]]: index)
//      CHECK:   %[[C0:.+]] = constant 0 : index
//      CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan @io::@s0b0_ro_constant[%[[C0]]] : memref<?xf32>
//      CHECK:   %[[INDEX:.+]] = affine.apply #[[MAP]]()[%[[I]], %[[OFFSET]]]
//      CHECK:   memref.load %[[SUBSPAN]][%[[INDEX]]]
