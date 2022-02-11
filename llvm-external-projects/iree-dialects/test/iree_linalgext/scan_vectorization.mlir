// RUN: iree-dialects-opt -iree-linalg-ext-scan-vec -split-input-file %s | FileCheck  %s

#map = affine_map<(d0)[s0] -> (d0 + s0)>
func @scan_1d(%arg1 : memref<5024xi32>, %arg2 : memref<i32>, %arg3: memref<5024xi32>) {
  %c32 = arith.constant 32 : index
  %c5024 = arith.constant 5024 : index
  %c0 = arith.constant 0 : index
  scf.for %arg0 = %c0 to %c5024 step %c32 {
    %7 = memref.subview %arg1[%arg0] [32] [1] : memref<5024xi32> to memref<32xi32, #map>
    %8 = memref.subview %arg3[%arg0] [32] [1] : memref<5024xi32> to memref<32xi32, #map>
    iree_linalg_ext.scan dimension(0) inclusive(true) ins(%7 : memref<32xi32, #map>) outs(%8, %arg2 : memref<32xi32, #map>, memref<i32>) {
    ^bb0(%arg4: i32, %arg5: i32):
      %9 = arith.addi %arg4, %arg5 : i32
      iree_linalg_ext.yield %9 : i32
    }
  }
  return
}

// CHECK:       #map = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK:       func @scan_1d(
// CHECK-SAME:    %[[ARG1:.+]]: memref<5024xi32>,
// CHECK-SAME:    %[[ARG2:.+]]: memref<i32>,
// CHECK-SAME:    %[[ARG3:.+]]: memref<5024xi32>
// CHECK:         %[[C32:.+]] = arith.constant 32 : index
// CHECK:         %[[C5024:.+]] = arith.constant 5024 : index
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK:         scf.for %[[ARG0:.+]] = %[[C0]] to %[[C5024]] step %[[C32]]
// CHECK:           %[[V0:.+]] = memref.subview %[[ARG1]][%[[ARG0]]] [32] [1] : memref<5024xi32> to memref<32xi32, #map>
// CHECK:           %[[V1:.+]] = memref.subview %[[ARG3]][%[[ARG0]]] [32] [1] : memref<5024xi32> to memref<32xi32, #map>
// CHECK:           %[[V2:.+]] = vector.transfer_read %[[V0]][%[[C0]]], %[[C0_I32]] {in_bounds = [true]} : memref<32xi32, #map>, vector<32xi32>
// CHECK:           %[[V3:.+]] = vector.transfer_read %[[ARG2]][], %[[C0_I32]] : memref<i32>, vector<i32>
// CHECK:           %[[DEST:.+]], %[[ACCVAL:.+]] = vector.scan <add>, %[[V2]], %[[V3]] {inclusive = true, reduction_dim = 0 : i64} : vector<32xi32>, vector<i32>
// CHECK:           vector.transfer_write %[[DEST]], %[[V1]][%[[C0]]] {in_bounds = [true]} : vector<32xi32>, memref<32xi32, #map>
// CHECK:           vector.transfer_write %[[ACCVAL]], %[[ARG2]][] : vector<i32>, memref<i32>
// CHECK:         return
