// RUN: iree-opt -allow-unregistered-dialect -split-input-file %s | iree-opt -allow-unregistered-dialect -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @interface_workgroup_info
func @interface_workgroup_info() {
  // CHECK: %workgroup_id = hal.interface.workgroup.id : tuple<index, index, index>
  %0 = hal.interface.workgroup.id : tuple<index, index, index>
  // CHECK: %workgroup_count = hal.interface.workgroup.count : tuple<index, index, index>
  %1 = hal.interface.workgroup.count : tuple<index, index, index>
  // CHECK: %workgroup_size = hal.interface.workgroup.size : tuple<index, index, index>
  %2 = hal.interface.workgroup.size : tuple<index, index, index>
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

// CHECK-LABEL: @interface_io_tiles
func @interface_io_tiles() {
  %c16 = constant 16 : index
  //      CHECK: %[[ARG0:.+]] = hal.interface.load.tensor.tile @interface::@s0b0, base_offset = %c16
  // CHECK-SAME:   offsets = [0], sizes = [4], strides = [1] : tensor<4xf32>
  %arg0 = hal.interface.load.tensor.tile @interface::@s0b0, base_offset = %c16,
    offsets = [0], sizes = [4], strides = [1]: tensor<4xf32>
  // CHECK-NEXT: %[[TEMP:.+]] = mhlo.add %[[ARG0]], %[[ARG0]]
  %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
  %c32 = constant 32 : index
  //      CHECK: hal.interface.store.tensor.tile %[[TEMP]], @interface::@s0b1, base_offset = %c32
  // CHECK-SAME:   offsets = [4], sizes = [7], strides = [1] : tensor<4xf32>
  hal.interface.store.tensor.tile %0, @interface::@s0b1, base_offset = %c32,
    offsets = [4], sizes = [7], strides = [1]: tensor<4xf32>
  return
}
