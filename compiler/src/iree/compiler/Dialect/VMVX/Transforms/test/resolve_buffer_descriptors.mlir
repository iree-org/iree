// RUN: iree-opt --split-input-file \
// RUN:   --iree-vmvx-resolve-buffer-descriptors="allow-unresolved=true" \
// RUN:   --canonicalize %s | FileCheck %s

// CHECK-LABEL: @resolve_subview
#map0 = affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>
module @resolve_subview{
  func.func @f(%arg0: memref<384x128xf32>, %arg1 : index, %arg2 : index) -> (!util.buffer, index, index, index, index, index) {
    // CHECK-DAG: %[[BASE_BUFFER:.*]], %[[BASE_OFFSET:.*]], %[[BASE_SIZES:.*]]:2, %[[BASE_STRIDES:.*]]:2 = vmvx.get_buffer_descriptor %arg0
    // CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
    // CHECK-DAG: %[[I0:.*]] = arith.muli %arg1, %[[BASE_STRIDES]]#0 : index
    // CHECK-DAG: %[[I1:.*]] = arith.addi %[[BASE_OFFSET]], %0 : index
    // CHECK-DAG: %[[I2:.*]] = arith.muli %arg2, %[[BASE_STRIDES]]#1 : index
    // CHECK-DAG: %[[SUB_OFFSET:.*]] = arith.addi %[[I1]], %[[I2]] : index
    //     CHECK: return %[[BASE_BUFFER]], %[[SUB_OFFSET]], %[[C64]], %[[C64]], %[[BASE_STRIDES]]#0, %[[BASE_STRIDES]]#1
    %0 = memref.subview %arg0[%arg1, %arg2] [64, 64] [1, 1] : memref<384x128xf32> to memref<64x64xf32, #map0>
    %base_buffer, %offset, %sizes:2, %strides:2 = vmvx.get_buffer_descriptor %0 : memref<64x64xf32, #map0> -> !util.buffer, index, index, index, index, index
    return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 : !util.buffer, index, index, index, index, index
  }
}

// -----

// CHECK-LABEL: @resolve_binding_subspan_zero_offset
func.func @resolve_binding_subspan_zero_offset() -> (!util.buffer, index, index, index, index, index) {
  // CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index
  // CHECK-DAG: %[[C384:.*]] = arith.constant 384 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  //     CHECK: %[[CAST:.*]] = vmvx.get_raw_interface_binding_buffer set(0) binding(0)
  //     CHECK: return %[[CAST]], %[[C0]], %[[C512]], %[[C384]], %[[C384]], %[[C1]]
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<512x384xf32>
  %base_buffer, %offset, %sizes:2, %strides:2 = vmvx.get_buffer_descriptor %0 : memref<512x384xf32> -> !util.buffer, index, index, index, index, index
  return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 : !util.buffer, index, index, index, index, index
}

// -----

// CHECK-LABEL: @resolve_binding_subspan_offset_index
func.func @resolve_binding_subspan_offset_index(%arg0 : index) -> (!util.buffer, index, index, index, index, index) {
  // CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index
  // CHECK-DAG: %[[C384:.*]] = arith.constant 384 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[INDEX_SIZE:.*]] = util.sizeof index
  // CHECK-DAG: %[[OFFSET:.*]] = arith.divui %arg0, %[[INDEX_SIZE]] : index
  //     CHECK: %[[CAST:.*]] = vmvx.get_raw_interface_binding_buffer set(0) binding(0)
  //     CHECK: return %[[CAST]], %[[OFFSET]], %[[C512]], %[[C384]], %[[C384]], %[[C1]]
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%arg0) alignment(64) : memref<512x384xindex>
  %base_buffer, %offset, %sizes:2, %strides:2 = vmvx.get_buffer_descriptor %0 : memref<512x384xindex> -> !util.buffer, index, index, index, index, index
  return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 : !util.buffer, index, index, index, index, index
}

// -----

// CHECK-LABEL: @resolve_binding_subspan_dyn_dims
func.func @resolve_binding_subspan_dyn_dims(%arg0 : index, %arg1 : index) -> (!util.buffer, index, index, index, index, index) {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  //     CHECK: %[[CAST:.*]] = vmvx.get_raw_interface_binding_buffer set(0) binding(0)
  //     CHECK: return %[[CAST]], %{{.*}}, %arg0, %arg1, %arg1, %[[C1]]
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<?x?xindex>{%arg0, %arg1}
  %base_buffer, %offset, %sizes:2, %strides:2 = vmvx.get_buffer_descriptor %0 : memref<?x?xindex> -> !util.buffer, index, index, index, index, index
  return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 : !util.buffer, index, index, index, index, index
}
