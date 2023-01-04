// RUN: iree-opt --allow-unregistered-dialect --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @interface_workgroup_info
func.func @interface_workgroup_info() {
  // CHECK: %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %0 = hal.interface.workgroup.id[0] : index
  // CHECK: %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %1 = hal.interface.workgroup.count[1] : index
  // CHECK: %workgroup_size_z = hal.interface.workgroup.size[2] : index
  %2 = hal.interface.workgroup.size[2] : index
  return
}

// -----

// CHECK-LABEL: @interface_io_subspan
//  CHECK-SAME: (%[[DIM0:.+]]: index, %[[DIM2:.+]]: index)
func.func @interface_io_subspan(%dim0: index, %dim2: index) {
  %c8 = arith.constant 8 : index

  // CHECK: = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c8) : memref<?x4x?x16xi8>{%[[DIM0]], %[[DIM2]]}
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c8) : memref<?x4x?x16xi8>{%dim0, %dim2}

  // CHECK: = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) alignment(16) : memref<16xi8>
  %1 = hal.interface.binding.subspan set(1) binding(2) type(storage_buffer) alignment(16) : memref<16xi8>

  return
}

// -----

func.func @interface_io_subspan_wrong_dynamic_dim(%dim: index) {
  %c8 = arith.constant 8 : index

  // expected-error @+1{{result type 'memref<?x4x?x16xi8>' has 2 dynamic dimensions but 1 associated dimension SSA values}}
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c8) : memref<?x4x?x16xi8>{%dim}

  return
}
