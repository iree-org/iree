// RUN: iree-opt --split-input-file --iree-codegen-emulate-narrow-type %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @memref_i4_to_i8() -> i4 {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<3x15xi4>
  %1 = memref.load %0[%c0, %c0] : memref<3x15xi4>
  return %1 : i4
}
// CHECK-LABEL: func.func @memref_i4_to_i8
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan {{.+}} : memref<23xi8>
//       CHECK:   %[[LOAD:.+]] = memref.load %[[SUBSPAN]][%[[C0]]] : memref<23xi8>
//       CHECK:   %[[TRUNC:.+]] = arith.trunci %[[LOAD]] : i8 to i4
//       CHECK:   return %[[TRUNC]] : i4

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @memref_i4_to_i8_dynamic(%arg0 : index, %arg1 : index, %arg2 : index) -> i4 {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%arg0) flags(ReadOnly) : memref<?x?xi4, strided<[?, 1], offset: ?>>{%arg1, %arg2}
  %1 = memref.load %0[%c0, %c0] : memref<?x?xi4, strided<[?, 1], offset: ?>>
  return %1 : i4
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> ((s0 * s1) floordiv 2, s0 floordiv 2)>
//      CHECK: func.func @memref_i4_to_i8_dynamic
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
//      CHECK:   %[[SIZE:.+]] = affine.max #[[MAP1]]()[%[[ARG2]], %[[ARG1]]]
//      CHECK:   hal.interface.binding.subspan
// CHECK-SAME:       offset(%[[ARG0]])
// CHECK-SAME:       memref<?xi8, strided<[1], offset: ?>>{%[[SIZE]]}

// -----
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @broadcast_extui() -> vector<1x1x64xi32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<64xi4>
  %1 = vector.load %0[%c0] : memref<64xi4>, vector<64xi4>
  %2 = vector.broadcast %1 : vector<64xi4> to vector<1x1x64xi4>
  %3 = arith.extui %2 : vector<1x1x64xi4> to vector<1x1x64xi32>
  return %3 : vector<1x1x64xi32>
}
// CHECK-LABEL: func @broadcast_extui()
//   CHECK-NOT:   vector.bitcast
//       CHECK:   vector.interleave

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @memref_load_2d_i4() -> i4 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x32xi4>
  %v = memref.load %0[%c1, %c0] : memref<16x32xi4>
  return %v : i4
}

// CHECK-LABEL:   func.func @memref_load_2d_i4()
//       CHECK:     %[[C16:.*]] = arith.constant 16 : index
//       CHECK:     %[[SUBSPAN:.*]] = hal.interface.binding.subspan {{.*}} : memref<256xi8>
//       CHECK:     %[[LOAD:.*]] = memref.load %[[SUBSPAN]][%[[C16]]] : memref<256xi8>
//       CHECK:     %[[TRUNC:.*]] = arith.trunci %[[LOAD]] : i8 to i4
//       CHECK:     return %[[TRUNC]] : i4
