// RUN: iree-opt --split-input-file --iree-vmvx-conversion --canonicalize %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

// CHECK: util.global private @__constant_5xi32 : !util.buffer
// CHECK: util.initializer {
// CHECK:   %[[CST:.*]] = util.buffer.constant : !util.buffer = dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
// CHECK:   util.global.store %[[CST]], @__constant_5xi32
memref.global "private" constant @__constant_5xi32 : memref<5xi32> = dense<[1, 2, 3, 4, 5]>

// CHECK-LABEL: func.func @entry(
//  CHECK-SAME:   %[[SCRATCHPAD:[a-z0-9]+]]: !util.buffer,
//  CHECK-SAME:   %[[CONSTANTS:[a-z0-9]+]]: !util.buffer,
//  CHECK-SAME:   %[[BINDINGS:[a-z0-9]+]]: !util.list<!util.buffer>,
//  CHECK-SAME:   %[[WORKGROUP_X:[a-z0-9]+]]: i32,
//  CHECK-SAME:   %[[WORKGROUP_Y:[a-z0-9]+]]: i32,
//  CHECK-SAME:   %[[WORKGROUP_Z:[a-z0-9]+]]: i32,
//  CHECK-SAME:   %[[WORKGROUP_SIZE_X:[a-z0-9]+]]: i32,
//  CHECK-SAME:   %[[WORKGROUP_SIZE_Y:[a-z0-9]+]]: i32,
//  CHECK-SAME:   %[[WORKGROUP_SIZE_Z:[a-z0-9]+]]: i32,
//  CHECK-SAME:   %[[WORKGROUP_COUNT_X:[a-z0-9]+]]: i32,
//  CHECK-SAME:   %[[WORKGROUP_COUNT_Y:[a-z0-9]+]]: i32,
//  CHECK-SAME:   %[[WORKGROUP_COUNT_Z:[a-z0-9]+]]: i32) {
func.func @entry() {
  %cst = arith.constant 0.000000e+00 : f32
  %c5 = arith.constant 5 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.get_global @__constant_5xi32 : memref<5xi32>
  //      CHECK: %[[BINDING0:.+]] = util.list.get %[[BINDINGS]][%c0] : !util.list<!util.buffer>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<5xf32>
  //      CHECK: %[[BINDING1:.+]] = util.list.get %[[BINDINGS]][%c1] : !util.list<!util.buffer>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<5xi32>
  %workgroup_size_x = hal.interface.workgroup.size[0] : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  //  CHECK-DAG: %[[WORKGROUP_X_IDX:.+]] = arith.index_cast %[[WORKGROUP_X]]
  //  CHECK-DAG: %[[WORKGROUP_SIZE_X_IDX:.+]] = arith.index_cast %[[WORKGROUP_SIZE_X]]
  //  CHECK-DAG: %[[WORKGROUP_COUNT_X_IDX:.+]] = arith.index_cast %[[WORKGROUP_COUNT_X]]
  //      CHECK: = affine.apply #{{.+}}[%[[WORKGROUP_X_IDX]], %[[WORKGROUP_SIZE_X_IDX]]]
  %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
  //      CHECK: = affine.apply #{{.+}}[%[[WORKGROUP_COUNT_X_IDX]], %[[WORKGROUP_SIZE_X_IDX]]]
  %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
  scf.for %arg0 = %3 to %c5 step %4 {
    %5 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 5)>(%arg0)[%workgroup_size_x]
    scf.for %arg1 = %c0 to %5 step %c1 {
      %6 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg1)[%arg0]
      %7 = memref.load %1[%6] : memref<5xf32>
      %8 = memref.load %0[%6] : memref<5xi32>
      %9 = arith.cmpf oeq, %7, %cst : f32
      %10 = arith.extui %9 : i1 to i32
      %11 = arith.muli %10, %8 : i32
      memref.store %11, %2[%6] : memref<5xi32>
    }
  }
  return
}
