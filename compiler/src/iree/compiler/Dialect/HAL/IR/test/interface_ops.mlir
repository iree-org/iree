// RUN: iree-opt --split-input-file --verify-diagnostics %s | FileCheck %s
// RUN: iree-opt --split-input-file -cse --verify-diagnostics %s | FileCheck %s --check-prefix=CSE

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

// CSE-LABEL: @interface_subspan_cse
func.func @interface_subspan_cse() {
  %c0 = arith.constant 0 : index

//   CSE: %[[BIND:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset({{.+}}) : !flow.dispatch.tensor<readwrite:tensor<2x32x16384xf32>
  %0 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<2x32x16384xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<2x32x16384xf32>>

//   CSE-NEXT: %[[LOAD:.+]] = flow.dispatch.tensor.load %[[BIND]]
  %2 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, 32, 16384], strides = [1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<2x32x16384xf32>> -> tensor<2x32x16384xf32>
//   CSE-NEXT: flow.dispatch.tensor.store %[[LOAD]], %[[BIND]]
  flow.dispatch.tensor.store %2, %0, offsets = [0, 0, 0], sizes = [2, 32, 16384], strides = [1, 1, 1] : tensor<2x32x16384xf32> -> !flow.dispatch.tensor<readwrite:tensor<2x32x16384xf32>>
  return
}

// -----

func.func @interface_io_subspan_wrong_dynamic_dim(%dim: index) {
  %c8 = arith.constant 8 : index

  // expected-error @+1{{result type 'memref<?x4x?x16xi8>' has 2 dynamic dimensions but 1 associated dimension SSA values}}
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c8) : memref<?x4x?x16xi8>{%dim}

  return
}
