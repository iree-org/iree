// RUN: iree-opt %s --iree-codegen-cleanup-buffer-alloc-view --split-input-file | FileCheck %s

func.func @alloc_remove(%arg0: index, %arg1: index) {
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  return
}
// CHECK-LABEL: func.func @alloc_remove
//  CHECK-NEXT:   return

// -----

func.func @alloc_keep(%arg0: index, %arg1: index) -> memref<?x?xf32> {
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  return %0 : memref<?x?xf32>
}
// CHECK-LABEL: func.func @alloc_keep
//  CHECK-NEXT:   %[[ALLOC:.+]] = memref.alloc
//  CHECK-NEXT:   return %[[ALLOC]]

// -----

func.func @cleanup_only_assume_alignment_uses() {
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<42xf32>
  memref.assume_alignment %0, 64 : memref<42xf32>
  return
}
// CHECK-LABEL: func.func @cleanup_only_assume_alignment_uses()
//  CHECK-NEXT:   return
