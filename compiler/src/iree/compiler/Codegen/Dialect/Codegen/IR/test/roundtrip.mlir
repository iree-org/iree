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

// -----

func.func @fusion_barrier(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = iree_codegen.fusion_barrier %arg0 : tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL: func.func @fusion_barrier(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?xf32>
// CHECK:         iree_codegen.fusion_barrier %[[ARG0]] : tensor<?xf32>

// -----

func.func @index_hint_lane_constant(%idx: index) -> index {
  %hinted = iree_codegen.index_hint %idx {hint = #iree_gpu.lane_constant<16>} : index
  return %hinted : index
}
// CHECK-LABEL: func.func @index_hint_lane_constant(
// CHECK-SAME:    %[[IDX:[a-zA-Z0-9_]+]]: index
// CHECK:         %[[HINT:.+]] = iree_codegen.index_hint %[[IDX]] {hint = #iree_gpu.lane_constant<16>} : index
// CHECK:         return %[[HINT]]

// -----

func.func @index_hint_lane_increment(%idx: index) -> index {
  %hinted = iree_codegen.index_hint %idx {hint = #iree_gpu.lane_increment<16>} : index
  return %hinted : index
}
// CHECK-LABEL: func.func @index_hint_lane_increment(
// CHECK-SAME:    %[[IDX:[a-zA-Z0-9_]+]]: index
// CHECK:         %[[HINT:.+]] = iree_codegen.index_hint %[[IDX]] {hint = #iree_gpu.lane_increment<16>} : index
// CHECK:         return %[[HINT]]

// -----

func.func @index_hint_multiple(%idx0: index, %idx1: index, %idx2: index) -> (index, index, index) {
  %row0 = iree_codegen.index_hint %idx0 {hint = #iree_gpu.lane_constant<16>} : index
  %row1 = iree_codegen.index_hint %idx1 {hint = #iree_gpu.lane_constant<16>} : index
  %col = iree_codegen.index_hint %idx2 {hint = #iree_gpu.lane_increment<16>} : index
  return %row0, %row1, %col : index, index, index
}
// CHECK-LABEL: func.func @index_hint_multiple(
// CHECK:         iree_codegen.index_hint {{.*}} {hint = #iree_gpu.lane_constant<16>} : index
// CHECK:         iree_codegen.index_hint {{.*}} {hint = #iree_gpu.lane_constant<16>} : index
// CHECK:         iree_codegen.index_hint {{.*}} {hint = #iree_gpu.lane_increment<16>} : index
