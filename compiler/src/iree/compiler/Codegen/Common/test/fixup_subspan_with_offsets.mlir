// RUN: iree-opt --iree-codegen-fixup-subspan-with-offsets=keep-dead-subspan-ops %s --split-input-file | FileCheck %s

func.func @simple(%arg0 : index, %arg1 : index, %arg2 : index) -> memref<?x?xf32>{
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%arg0) : memref<?x?xf32>{%arg1, %arg2}
  return %0 : memref<?x?xf32>
}
//      CHECK: func @simple(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index)
//      CHECK:   hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[ARG0]]) : memref<?x?xf32, strided<[?, 1], offset: ?>>{%[[ARG1]], %[[ARG2]]}

// -----

func.func @simple_subview_use(%arg0  : index, %arg1 : index, %arg2 : index) -> memref<4x5xf32, strided<[?, 1], offset: ?>> {
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%arg0) : memref<?x?xf32>{%arg1, %arg2}
  %1 = memref.subview %0[0, 0] [4, 5] [1, 1] : memref<?x?xf32> to memref<4x5xf32, strided<[?, 1], offset: ?>>
  return %1 : memref<4x5xf32, strided<[?, 1], offset:?>>
}
//      CHECK: func @simple_subview_use(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index)
//      CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[ARG0]]) : memref<?x?xf32, strided<[?, 1], offset: ?>>{%[[ARG1]], %[[ARG2]]}
//      CHECK:   memref.subview %[[BINDING]][0, 0] [4, 5] [1, 1]

// -----

func.func @cast_use(%arg0  : index, %arg1 : index, %arg2 : index) -> memref<20x30xf32> {
  %c20 = arith.constant 20 : index
  %c30 = arith.constant 30 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%arg0) : memref<?x?xf32>{%c20, %c30}
  %1 = memref.cast %0 : memref<?x?xf32> to memref<20x30xf32>
  return %1 : memref<20x30xf32>
}
//      CHECK: func @cast_use(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index)
//  CHECK-DAG:   %[[C20:.+]] = arith.constant 20 : index
//  CHECK-DAG:   %[[C30:.+]] = arith.constant 30 : index
//      CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[ARG0]]) : memref<?x?xf32, strided<[?, 1], offset: ?>>{%[[C20]], %[[C30]]}
//      CHECK:   memref.cast %[[BINDING]] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<20x30xf32, strided<[?, 1], offset: ?>>

// -----

func.func @strided_subview_use() -> memref<4x5xf32, strided<[20, 1], offset: 62>> {
  %c64 = arith.constant 64 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c64) : memref<10x20xf32>
  %1 = memref.subview %0[3, 2] [4, 5] [2, 4] : memref<10x20xf32> to memref<4x5xf32, strided<[20, 1], offset: 62>>
  return %1 : memref<4x5xf32, strided<[20, 1], offset:62>>
}
//      CHECK: func @strided_subview_use()
//  CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//      CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[C64]]) : memref<10x20xf32, strided<[20, 1], offset: 16>>
//      CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[BINDING]][3, 2] [4, 5] [2, 4] : memref<10x20xf32, strided<[20, 1], offset: 16>> to memref<4x5xf32, strided<[40, 4], offset: 78>>

// -----

func.func @extract_metadata(%arg0  : index, %arg1 : index, %arg2 : index) -> memref<i32> {
  %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%arg0) : memref<?x?xi32>{%arg1, %arg2}
  %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %0 : memref<?x?xi32> -> memref<i32>, index, index, index, index, index
  return %base_buffer : memref<i32>
}
//      CHECK: func @extract_metadata(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index)
//      CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[ARG0]]) : memref<?x?xi32, strided<[?, 1], offset: ?>>{%[[ARG1]], %[[ARG2]]}
//      CHECK:   %[[BASE_BUFFER:.+]], %{{.+}}, %{{.+}}:2, %{{.+}}:2 = memref.extract_strided_metadata %[[BINDING]]
//      CHECK:   return %[[BASE_BUFFER]]
