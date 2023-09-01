// RUN: iree-opt --split-input-file --iree-codegen-emulate-narrow-type %s | FileCheck %s

func.func @memref_i4_to_i8() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<3x15xi4>
  return
}
// CHECK-LABEL: func.func @memref_i4_to_i8
//       CHECK:    hal.interface.binding.subspan {{.+}} memref<23xi8>

// -----

func.func @memref_i4_to_i8_dynamic(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%arg0) flags(ReadOnly) : memref<?x?xi4, strided<[?, 1], offset: ?>>{%arg1, %arg2}
  return
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) floordiv 2)>
//      CHECK: func.func @memref_i4_to_i8_dynamic
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//      CHECK:   %[[SIZE:.+]] = affine.apply #[[MAP1]]()[%[[ARG1]], %[[ARG2]]]
//      CHECK:   hal.interface.binding.subspan
// CHECK-SAME:       offset(%[[ARG0]])
// CHECK-SAME:       memref<?xi8, strided<[1], offset: ?>>{%[[SIZE]]}
