// RUN: iree-opt -split-input-file -iree-vmvx-conversion -canonicalize %s | IreeFileCheck %s

hal.interface @io attributes {sym_visibility = "private"} {
  hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @s0b1_xw_external, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
}

// CHECK: memref.global "private" constant @__constant_5xi32 : memref<5xi32> = dense<[1, 2, 3, 4, 5]>
memref.global "private" constant @__constant_5xi32 : memref<5xi32> = dense<[1, 2, 3, 4, 5]>

// CHECK-LABEL: func @entry(
//  CHECK-SAME:   %[[SCRATCHPAD:.+]]: memref<?xi8>,
//  CHECK-SAME:   %[[CONSTANTS:.+]]: memref<?xi32>,
//  CHECK-SAME:   %[[BINDINGS:.+]]: !util.list<memref<?xi8>>,
//  CHECK-SAME:   %[[WORKGROUP_X:[a-z0-9]+]]: index,
//  CHECK-SAME:   %[[WORKGROUP_Y:[a-z0-9]+]]: index,
//  CHECK-SAME:   %[[WORKGROUP_Z:[a-z0-9]+]]: index,
//  CHECK-SAME:   %[[WORKGROUP_SIZE_X:[a-z0-9]+]]: index,
//  CHECK-SAME:   %[[WORKGROUP_SIZE_Y:[a-z0-9]+]]: index,
//  CHECK-SAME:   %[[WORKGROUP_SIZE_Z:[a-z0-9]+]]: index,
//  CHECK-SAME:   %[[WORKGROUP_COUNT_X:[a-z0-9]+]]: index,
//  CHECK-SAME:   %[[WORKGROUP_COUNT_Y:[a-z0-9]+]]: index,
//  CHECK-SAME:   %[[WORKGROUP_COUNT_Z:[a-z0-9]+]]: index) {
func @entry() {
  %cst = constant 0.000000e+00 : f32
  %c5 = constant 5 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = memref.get_global @__constant_5xi32 : memref<5xi32>
  //      CHECK: %[[BINDING0_RAW:.+]] = util.list.get %[[BINDINGS]][%c0] : !util.list<memref<?xi8>>
  // CHECK-NEXT: %[[BINDING0:.+]] = builtin.unrealized_conversion_cast %[[BINDING0_RAW]] : memref<?xi8> to memref<5xf32>
  %1 = hal.interface.binding.subspan @io::@s0b0_ro_external[%c0] : memref<5xf32>
  //      CHECK: %[[BINDING1_RAW:.+]] = util.list.get %[[BINDINGS]][%c1] : !util.list<memref<?xi8>>
  // CHECK-NEXT: %[[BINDING1:.+]] = builtin.unrealized_conversion_cast %[[BINDING1_RAW]] : memref<?xi8> to memref<5xi32>
  %2 = hal.interface.binding.subspan @io::@s0b1_xw_external[%c0] : memref<5xi32>
  %workgroup_size_x = hal.interface.workgroup.size[0] : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  //      CHECK: = affine.apply #{{.+}}[%[[WORKGROUP_X]], %[[WORKGROUP_SIZE_X]]]
  %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
  // CHECK-NEXT: = affine.apply #{{.+}}[%[[WORKGROUP_COUNT_X]], %[[WORKGROUP_SIZE_X]]]
  %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
  scf.for %arg0 = %3 to %c5 step %4 {
    %5 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 5)>(%arg0)[%workgroup_size_x]
    scf.for %arg1 = %c0 to %5 step %c1 {
      %6 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg1)[%arg0]
      %7 = memref.load %1[%6] : memref<5xf32>
      %8 = memref.load %0[%6] : memref<5xi32>
      %9 = cmpf oeq, %7, %cst : f32
      %10 = zexti %9 : i1 to i32
      %11 = muli %10, %8 : i32
      memref.store %11, %2[%6] : memref<5xi32>
    }
  }
  return
}
