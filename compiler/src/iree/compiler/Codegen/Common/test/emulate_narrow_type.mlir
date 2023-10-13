// RUN: iree-opt --split-input-file --iree-codegen-emulate-narrow-type %s | FileCheck %s

func.func @memref_i4_to_i8() -> i4 {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<3x15xi4>
  %1 = memref.load %0[%c0, %c0] : memref<3x15xi4>
  return %1 : i4
}
// CHECK-LABEL: func.func @memref_i4_to_i8
//       CHECK:    hal.interface.binding.subspan {{.+}} memref<23xi8>

// -----

func.func @memref_i4_to_i8_dynamic(%arg0 : index, %arg1 : index, %arg2 : index) -> i4 {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%arg0) flags(ReadOnly) : memref<?x?xi4, strided<[?, 1], offset: ?>>{%arg1, %arg2}
  %1 = memref.load %0[%c0, %c0] : memref<?x?xi4, strided<[?, 1], offset: ?>>
  return %1 : i4
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

// -----

func.func @broadcast_extui() -> vector<1x1x64xi32> {
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<64xi4>
  %1 = vector.load %0[%c0] : memref<64xi4>, vector<64xi4>
  %2 = vector.broadcast %1 : vector<64xi4> to vector<1x1x64xi4>
  %3 = arith.extui %2 : vector<1x1x64xi4> to vector<1x1x64xi32>
  return %3 : vector<1x1x64xi32>
}
// CHECK-LABEL: func @broadcast_extui()
//   CHECK-NOT:   vector.bitcast
//       CHECK:   vector.shuffle
