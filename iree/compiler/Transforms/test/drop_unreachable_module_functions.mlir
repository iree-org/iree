// RUN: iree-opt %s -iree-drop-unreachable-module-functions -split-input-file | IreeFileCheck %s --implicit-check-not @unused

// CHECK-LABEL: @exportedModuleFn
func @exportedModuleFn(%arg0 : memref<?xf32>) -> memref<?xf32>
    attributes {iree.module.export} {
  // CHECK: iree_hl_seq.call @fn1
  %0 = iree_hl_seq.call @fn1(%arg0) : (memref<?xf32>) -> memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK: @fn1
func @fn1(%arg0 : memref<?xf32>) -> memref<?xf32> {
  // CHECK: iree_hl_seq.call @fn2
  %0 = iree_hl_seq.call @fn2(%arg0) : (memref<?xf32>) -> memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK: @fn2
func @fn2(%arg0 : memref<?xf32>) -> memref<?xf32> {
  return %arg0 : memref<?xf32>
}

// CHECK-NOT: @unusedFn3
func @unusedFn3(%arg0 : memref<?xf32>) -> memref<?xf32> {
  return %arg0 : memref<?xf32>
}

// -----

// CHECK-NOT: @unusedFn
func @unusedFn(%arg0 : memref<?xf32>) -> memref<?xf32> {
  return %arg0 : memref<?xf32>
}

// -----

// CHECK-LABEL: @exportedFnWithImports
func @exportedFnWithImports(%arg0 : memref<?xf32>) -> memref<?xf32>
    attributes {iree.module.export} {
  // CHECK: iree_hl_seq.call @usedImportFn
  %0 = iree_hl_seq.call @usedImportFn(%arg0) : (memref<?xf32>) -> memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK: @usedImportFn
func @usedImportFn(%arg0 : memref<?xf32>) -> memref<?xf32>

// CHECK-NOT: @unusedImportFn
func @unusedImportFn(%arg0 : memref<?xf32>) -> memref<?xf32>
