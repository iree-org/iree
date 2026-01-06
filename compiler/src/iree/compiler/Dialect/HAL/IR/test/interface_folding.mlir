// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: @hal_interface_binding_subspan_op
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
util.func public @hal_interface_binding_subspan_op(%arg0 : index, %arg1 : index) -> (index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = hal.interface.binding.subspan layout(<
      constants = 0, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">], flags = Indirect>)
      binding(0) : memref<64x?x?xf16>{%arg0, %arg1}
  %d0 = memref.dim %0, %c0 : memref<64x?x?xf16>
  %d1 = memref.dim %0, %c1 : memref<64x?x?xf16>
  %d2 = memref.dim %0, %c2 : memref<64x?x?xf16>
  // CHECK: %[[C64:.+]] = arith.constant 64 : index
  // CHECK: util.return %[[C64]], %[[ARG0]], %[[ARG1]]
  util.return %d0, %d1, %d2 : index, index, index
}

// -----

// CHECK-LABEL: @hal_interface_binding_subspan_op_and_assume_alignment
util.func public @hal_interface_binding_subspan_op_and_assume_alignment() -> index {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<
      constants = 0, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">], flags = Indirect>)
      binding(0) : memref<64xf16>
  %d0 = memref.dim %0, %c0 : memref<64xf16>
  // CHECK: %[[C64:.+]] = arith.constant 64 : index
  // CHECK: util.return %[[C64]]
  util.return %d0 : index
}

// -----

// CHECK-LABEL: @fold_dim_through_assume_alignment
// CHECK-SAME:    %[[DIM0:[a-zA-Z0-9]+]]
util.func public @fold_dim_through_assume_alignment(%dim0 : index) -> index {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<
      constants = 0, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">], flags = Indirect>)
      binding(0) : memref<?xf32>{%dim0}
  %assume = memref.assume_alignment %0, 64 : memref<?xf32>
  %d0 = memref.dim %assume, %c0 : memref<?xf32>
  // CHECK: util.return %[[DIM0]]
  util.return %d0 : index
}

// -----

// CHECK-LABEL: @fold_dim_through_assume_and_cast_chain
// CHECK-SAME:    %[[DIM0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DIM1:[a-zA-Z0-9]+]]
util.func public @fold_dim_through_assume_and_cast_chain(%dim0 : index, %dim1 : index) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = hal.interface.binding.subspan layout(<
      constants = 0, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">], flags = Indirect>)
      binding(0) : memref<?x?xf32>{%dim0, %dim1}
  %assume = memref.assume_alignment %0, 64 : memref<?x?xf32>
  %cast = amdgpu.fat_raw_buffer_cast %assume : memref<?x?xf32> to memref<?x?xf32, #amdgpu.address_space<fat_raw_buffer>>
  %d0 = memref.dim %cast, %c0 : memref<?x?xf32, #amdgpu.address_space<fat_raw_buffer>>
  %d1 = memref.dim %cast, %c1 : memref<?x?xf32, #amdgpu.address_space<fat_raw_buffer>>
  // CHECK: util.return %[[DIM0]], %[[DIM1]]
  util.return %d0, %d1 : index, index
}
