// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-vectorize-memref-copy))" %s | FileCheck %s

func.func @memref_copy(%source: memref<2x2xf32>, %dest: memref<2x2xf32>) {
  memref.copy %source, %dest : memref<2x2xf32> to memref<2x2xf32>
  return
}

// CHECK-LABEL: func.func @memref_copy
//  CHECK-SAME:   %[[SOURCE:[A-Za-z0-9]+]]: memref<2x2xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9]+]]: memref<2x2xf32>
//       CHECK:   %[[RD:.+]] = vector.transfer_read %[[SOURCE]]
//       CHECK:   vector.transfer_write %[[RD]], %[[DEST]]
