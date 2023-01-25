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

// CHECK-LABEL: @resolve_subview_rankreducing
#map0 = affine_map<(d0)[s0] -> (d0 * 128 + s0)>
func.func @resolve_subview_rankreducing(%arg0: memref<384x128xf32>, %arg1 : index, %arg2 : index) -> (!util.buffer, index, index, index) {
  // CHECK-DAG: %[[BASE_BUFFER:.*]], %[[BASE_OFFSET:.*]], %[[BASE_SIZES:.*]]:2, %[[BASE_STRIDES:.*]]:2 = vmvx.get_buffer_descriptor %arg0
  // CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
  // CHECK-DAG: %[[I0:.*]] = arith.muli %arg1, %[[BASE_STRIDES]]#0 : index
  // CHECK:     %[[I1:.*]] = arith.addi %[[BASE_OFFSET]], %[[I0]] : index
  // CHECK:     %[[I2:.*]] = arith.muli %arg2, %[[BASE_STRIDES]]#1 : index
  // CHECK:     %[[I3:.*]] = arith.addi %[[I1]], %[[I2]] : index
  // CHECK:     return %[[BASE_BUFFER]], %[[I3]], %[[C64]], %[[BASE_STRIDES]]#0
  %0 = memref.subview %arg0[%arg1, %arg2] [64, 1] [1, 1] : memref<384x128xf32> to memref<64xf32, #map0>
  %base_buffer, %offset, %size, %stride = vmvx.get_buffer_descriptor %0 : memref<64xf32, #map0> -> !util.buffer, index, index, index
  return %base_buffer, %offset, %size, %stride : !util.buffer, index, index, index
}

// -----

// Check that we properly resolve subview with rankreducing when the dropped
// rank is not the last one.
// Orig strides: [%strides#0, %strides#1, %strides#2]
// Sub strides: [1, 1, 1]
// => New strides: [%strides#0, %strides#1, %strides#2]
// Final strides == filterOutReducedDim(new strides, 0) == [%strides#1 , %strides#2]
//
// Orig offset: %offset
// Sub offsets: [%arg1, %arg2, 0]
// => Final offset: %arg1 * %strides#0 + %arg2 * %strides#1 + 0 * %strides#2 + %offset
//
// Final sizes == filterOutReducedDim(subview sizes, 0) == [6, 3]
//
// CHECK-LABEL: @resolve_subview_rankreducing_not_at_the_end
func.func @resolve_subview_rankreducing_not_at_the_end(%arg0: memref<8x16x4xf32>, %arg1 : index, %arg2 : index) -> (!util.buffer, index, index, index, index, index) {
  // CHECK-DAG: %[[BASE_BUFFER:.*]], %[[BASE_OFFSET:.*]], %[[BASE_SIZES:.*]]:3, %[[BASE_STRIDES:.*]]:3 = vmvx.get_buffer_descriptor %arg0
  // CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[I0:.*]] = arith.muli %arg1, %[[BASE_STRIDES]]#0 : index
  // CHECK:     %[[I1:.*]] = arith.addi %[[BASE_OFFSET]], %[[I0]] : index
  // CHECK:     %[[I2:.*]] = arith.muli %arg2, %[[BASE_STRIDES]]#1 : index
  // CHECK:     %[[I3:.*]] = arith.addi %[[I1]], %[[I2]] : index
  // CHECK:     return %[[BASE_BUFFER]], %[[I3]], %[[C6]], %[[C3]], %[[BASE_STRIDES]]#1, %[[BASE_STRIDES]]#2

  %0 = memref.subview %arg0[%arg1, %arg2, 0] [1, 6, 3] [1, 1, 1] : memref<8x16x4xf32> to memref<6x3xf32, strided<[4,1], offset : ?>>
  %base_buffer, %offset, %sizes:2, %strides:2 = vmvx.get_buffer_descriptor %0 : memref<6x3xf32, strided<[4,1], offset : ?>> -> !util.buffer, index, index, index, index, index
  return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 : !util.buffer, index, index, index, index, index
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<512x384xf32>
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%arg0) : memref<512x384xindex>
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<?x?xindex>{%arg0, %arg1}
  %base_buffer, %offset, %sizes:2, %strides:2 = vmvx.get_buffer_descriptor %0 : memref<?x?xindex> -> !util.buffer, index, index, index, index, index
  return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 : !util.buffer, index, index, index, index, index
}

// -----

// CHECK-LABEL: @resolve_alloca_static
func.func @resolve_alloca_static() -> (!util.buffer, index, index, index, index, index) {
  // CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index
  // CHECK-DAG: %[[C384:.*]] = arith.constant 384 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  //     CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast
  //     CHECK: return %[[CAST]], %[[C0]], %[[C512]], %[[C384]], %[[C384]], %[[C1]]
  %0 = memref.alloca() : memref<512x384xf32>
  %base_buffer, %offset, %sizes:2, %strides:2 = vmvx.get_buffer_descriptor %0 : memref<512x384xf32> -> !util.buffer, index, index, index, index, index
  return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 : !util.buffer, index, index, index, index, index
}

// -----

// CHECK-LABEL: @resolve_alloca_dynamic
func.func @resolve_alloca_dynamic(%arg0 : index) -> (!util.buffer, index, index, index, index, index) {
  // CHECK-DAG: %[[C384:.*]] = arith.constant 384 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  //     CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast
  //     CHECK: return %[[CAST]], %[[C0]], %arg0, %[[C384]], %[[C384]], %[[C1]]
  %0 = memref.alloca(%arg0) : memref<?x384xf32>
  %base_buffer, %offset, %sizes:2, %strides:2 = vmvx.get_buffer_descriptor %0 : memref<?x384xf32> -> !util.buffer, index, index, index, index, index
  return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 : !util.buffer, index, index, index, index, index
}

// -----

// CHECK-LABEL: @resolve_global
memref.global "private" constant @__constant_2xi32 : memref<512x384xf32> = dense<0.0>

func.func @resolve_global() -> (!util.buffer, index, index, index, index, index) {
  // CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index
  // CHECK-DAG: %[[C384:.*]] = arith.constant 384 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  //     CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast
  //     CHECK: return %[[CAST]], %[[C0]], %[[C512]], %[[C384]], %[[C384]], %[[C1]]
  %0 = memref.get_global @__constant_2xi32 : memref<512x384xf32>
  %base_buffer, %offset, %sizes:2, %strides:2 = vmvx.get_buffer_descriptor %0 : memref<512x384xf32> -> !util.buffer, index, index, index, index, index
  return %base_buffer, %offset, %sizes#0, %sizes#1, %strides#0, %strides#1 : !util.buffer, index, index, index, index, index
}
