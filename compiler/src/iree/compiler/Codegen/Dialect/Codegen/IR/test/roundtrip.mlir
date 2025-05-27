// RUN: iree-opt --split-input-file %s | FileCheck %s

func.func @load_from_buffer(%arg0: memref<4xf32>) -> tensor<4xf32> {
  %value = iree_codegen.load_from_buffer %arg0 : memref<4xf32> -> tensor<4xf32>
  return %value : tensor<4xf32>
}
// CHECK-LABEL: func.func @load_from_buffer(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK:         iree_codegen.load_from_buffer %[[ARG0]]
// CHECK-SAME:      : memref<4xf32> -> tensor<4xf32>

// -----

func.func @load_from_buffer_mixed_static_dynamic(%arg0: memref<?x4xf32>) -> tensor<4x?xf32> {
  %value = iree_codegen.load_from_buffer %arg0 : memref<?x4xf32> -> tensor<4x?xf32>
  return %value : tensor<4x?xf32>
}
// CHECK-LABEL: func.func @load_from_buffer_mixed_static_dynamic(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK:         iree_codegen.load_from_buffer %[[ARG0]]
// CHECK-SAME:      : memref<?x4xf32> -> tensor<4x?xf32>

// -----

func.func @load_from_strided_memref(
    %arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>
) -> tensor<?x?xf32> {
  %value = iree_codegen.load_from_buffer %arg0
    : memref<?x?xf32, strided<[?, 1], offset: ?>> -> tensor<?x?xf32>
  return %value : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @load_from_strided_memref(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]:
// CHECK:         iree_codegen.load_from_buffer %[[ARG0]]
// CHECK-SAME:      : memref<?x?xf32, strided<[?, 1], offset: ?>> -> tensor<?x?xf32>

// -----

func.func @store_to_buffer(%arg0: tensor<4xf32>, %arg1: memref<4xf32>) {
  iree_codegen.store_to_buffer %arg0, %arg1 : tensor<4xf32> into memref<4xf32>
  return
}
// CHECK-LABEL: func.func @store_to_buffer(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         iree_codegen.store_to_buffer %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      : tensor<4xf32> into memref<4xf32>

// -----

func.func @store_to_buffer_mixed_static_dynamic(%arg0: tensor<4x?xf32>, %arg1: memref<?x4xf32>) {
  iree_codegen.store_to_buffer %arg0, %arg1 : tensor<4x?xf32> into memref<?x4xf32>
  return
}
// CHECK-LABEL: func.func @store_to_buffer_mixed_static_dynamic(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         iree_codegen.store_to_buffer %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      : tensor<4x?xf32> into memref<?x4xf32>

// -----

func.func @store_to_strided_memref(
    %arg0: tensor<?x?xf32>, %arg1: memref<?x?xf32, strided<[?, 1], offset: ?>>
) {
  iree_codegen.store_to_buffer %arg0, %arg1
    : tensor<?x?xf32> into memref<?x?xf32, strided<[?, 1], offset: ?>>
  return
}
// CHECK-LABEL: func.func @store_to_strided_memref(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]:
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK:         iree_codegen.store_to_buffer %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      : tensor<?x?xf32> into memref<?x?xf32, strided<[?, 1], offset: ?>>
