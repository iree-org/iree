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

func.func @index_hint(%idx: index) -> index {
  %hinted = iree_codegen.index_hint %idx([]) : index
  return %hinted : index
}
// CHECK-LABEL: func.func @index_hint(
// CHECK-SAME:    %[[IDX:[a-zA-Z0-9_]+]]: index
// CHECK:         %[[HINT:.+]] = iree_codegen.index_hint %[[IDX]]([]) : index
// CHECK:         return %[[HINT]]

// -----

// Test workgroup_scope attribute roundtrip.
func.func private @workgroup_scope_attr() attributes {
    scope = #iree_codegen.workgroup_scope
}
// CHECK-LABEL: func.func private @workgroup_scope_attr()
// CHECK-SAME:    scope = #iree_codegen.workgroup_scope

// -----

// Test workgroup_scope attribute with linearize option.
func.func private @workgroup_scope_attr_linearize() attributes {
    scope = #iree_codegen.workgroup_scope<linearize>
}
// CHECK-LABEL: func.func private @workgroup_scope_attr_linearize()
// CHECK-SAME:    scope = #iree_codegen.workgroup_scope<linearize>
