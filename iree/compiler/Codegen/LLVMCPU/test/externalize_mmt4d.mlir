// RUN: iree-opt -split-input-file -iree-llvmcpu-externalize-mmt4d %s | IreeFileCheck %s

// CHECK-LABEL: @tiled_mmt4d_8x4x8_static
// CHECK-SAME: (%[[LHS:.+]]: memref<1x1x8x4xi8>, %[[RHS:.+]]: memref<1x1x8x4xi8>, %[[DST:.+]]: memref<1x1x8x8xi32>)
func @tiled_mmt4d_8x4x8_static(
    %lhs: memref<1x1x8x4xi8>, %rhs: memref<1x1x8x4xi8>,
    %dst: memref<1x1x8x8xi32>) {
  // CHECK-DAG: %[[K:.+]] = arith.constant 4 : i32
  // CHECK-DAG: %[[LHS_DYN:.+]] = memref.cast %[[LHS]] : memref<1x1x8x4xi8> to memref<?x?x8x4xi8>
  // CHECK-DAG: %[[RHS_DYN:.+]] = memref.cast %[[RHS]] : memref<1x1x8x4xi8> to memref<?x?x8x4xi8>
  // CHECK-DAG: %[[DST_DYN:.+]] = memref.cast %[[DST]] : memref<1x1x8x8xi32> to memref<?x?x8x8xi32>
  // CHECK: call @mmt4d_8x4x8_i8i8i32(%[[K]], %[[LHS_DYN]], %[[RHS_DYN]], %[[DST_DYN]])
  // CHECK-SAME: : (i32, memref<?x?x8x4xi8>, memref<?x?x8x4xi8>, memref<?x?x8x8xi32>) -> ()
  linalg.mmt4d ins(%lhs, %rhs: memref<1x1x8x4xi8>, memref<1x1x8x4xi8>)
              outs(%dst: memref<1x1x8x8xi32>)
  return
}

// -----

// CHECK-LABEL: @tiled_mmt4d_8x4x8_dynamic
// CHECK-SAME: (%[[LHS:.+]]: memref<?x?x8x4xi8>, %[[RHS:.+]]: memref<?x?x8x4xi8>, %[[DST:.+]]: memref<?x?x8x8xi32>)
func @tiled_mmt4d_8x4x8_dynamic(
    %lhs: memref<?x?x8x4xi8>, %rhs: memref<?x?x8x4xi8>,
    %dst: memref<?x?x8x8xi32>) {
  // CHECK-DAG: %[[K0:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[K1:.+]] = memref.dim %[[LHS]], %c1 : memref<?x?x8x4xi8>
  // CHECK-DAG: %[[K_SIZE:.+]] = arith.muli %[[K0]], %[[K1]]
  // CHECK-DAG: %[[K_SIZE_I32:.+]] = arith.index_cast %[[K_SIZE]] : index to i32
  // CHECK: call @mmt4d_8x4x8_i8i8i32(%[[K_SIZE_I32]], %[[LHS]], %[[RHS]], %[[DST]])
  // CHECK-SAME: : (i32, memref<?x?x8x4xi8>, memref<?x?x8x4xi8>, memref<?x?x8x8xi32>) -> ()
  linalg.mmt4d ins(%lhs, %rhs: memref<?x?x8x4xi8>, memref<?x?x8x4xi8>)
              outs(%dst: memref<?x?x8x8xi32>)
  return
}
