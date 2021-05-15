// RUN: iree-opt -allow-unregistered-dialect -split-input-file %s | iree-opt -allow-unregistered-dialect -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @interface_workgroup_info
func @interface_workgroup_info() {
  // CHECK: %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %0 = hal.interface.workgroup.id[0] : index
  // CHECK: %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %1 = hal.interface.workgroup.count[1] : index
  // CHECK: %workgroup_size_z = hal.interface.workgroup.size[2] : index
  %2 = hal.interface.workgroup.size[2] : index
  return
}

// -----

// CHECK-LABEL: @interface_io_tensors
func @interface_io_tensors() {
  %c16 = constant 16 : index
  // CHECK: %[[ARG0:.+]] = hal.interface.load.tensor @interface::@s0b0, offset = %c16 : tensor<4xf32>
  %arg0 = hal.interface.load.tensor @interface::@s0b0, offset=%c16 : tensor<4xf32>
  // CHECK-NEXT: %[[TEMP:.+]] = mhlo.add %[[ARG0]], %[[ARG0]]
  %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
  %c32 = constant 32 : index
  // CHECK: hal.interface.store.tensor %[[TEMP]], @interface::@s0b1, offset = %c32 : tensor<4xf32>
  hal.interface.store.tensor %0, @interface::@s0b1, offset=%c32 : tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: @interface_io_subspan
func @interface_io_subspan() {
  %c8 = constant 8 : index
  %c16 = constant 16 : index

  // CHECK: = hal.interface.binding.subspan @interface::@s0b0[%c8] : memref<?xi8>
  %0 = hal.interface.binding.subspan @interface::@s0b0[%c8] : memref<?xi8>

  // CHECK: = hal.interface.binding.subspan @interface::@s0b0[%c8, %c16] : memref<16xi8>
  %1 = hal.interface.binding.subspan @interface::@s0b0[%c8, %c16] : memref<16xi8>

  return
}
