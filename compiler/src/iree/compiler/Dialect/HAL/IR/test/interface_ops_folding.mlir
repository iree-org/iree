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
