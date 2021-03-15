// RUN: iree-opt %s -iree-codegen-cleanup-buffer-alloc-view -split-input-file | IreeFileCheck %s

func @alloc_remove(%arg0: index, %arg1: index) {
  %0 = alloc(%arg0, %arg1) : memref<?x?xf32>
  return
}
// CHECK-LABEL: func @alloc_remove
//  CHECK-NEXT:   return

// -----

func @alloc_keep(%arg0: index, %arg1: index) -> memref<?x?xf32> {
  %0 = alloc(%arg0, %arg1) : memref<?x?xf32>
  return %0 : memref<?x?xf32>
}
// CHECK-LABEL: func @alloc_keep
//  CHECK-NEXT:   %[[ALLOC:.+]] = alloc
//  CHECK-NEXT:   return %[[ALLOC]]

