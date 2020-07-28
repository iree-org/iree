// RUN: iree-opt --iree-codegen-linalg-to-llvm-matmul-vectorization-pass -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: func @matmul_128x128x128
// CHECK-SAME: (%[[ARG0:.+]]: memref<128x128xf32>, %[[ARG1:.+]]: memref<128x128xf32>, %[[ARG2:.+]]: memref<128x128xf32>)
func @matmul_128x128x128(%arg0 : memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) {
    linalg.matmul %arg0, %arg1, %arg2 : (memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32>)
    return
}
// CHECK: %[[L3END:.+]] = constant 128 : index
// CHECK: %[[L3STEP:.+]] = constant 64 : index
// CHECK: %[[L1STEP:.+]] = constant 4 : index
// CHECK: %[[L2STEP:.+]] = constant 32 : index
// CHECK: %[[START:.+]] = constant 0 : index
// CHECK: scf.for %[[IL3:.+]] = %[[START]] to %[[L3END]] step %[[L3STEP]]
// CHECK: scf.for %[[JL3:.+]] = %[[START]] to %[[L3END]] step %[[L3STEP]]
// CHECK: scf.for %[[KL3:.+]] = %[[START]] to %[[L3END]] step %[[L3STEP]]
// CHECK: %[[ARG0_TILE_L3:.+]] = subview %[[ARG0]][%[[IL3]], %[[KL3]]] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32
// CHECK: %[[ARG1_TILE_L3:.+]] = subview %[[ARG1]][%[[KL3]], %[[JL3]]] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32
// CHECK: %[[ARG2_TILE_L3:.+]] = subview %[[ARG2]][%[[IL3]], %[[JL3]]] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32
// CHECK: scf.for %[[IL2:.+]] = %[[START]] to %[[L3STEP]] step %[[L2STEP]]
// CHECK: scf.for %[[JL2:.+]] = %[[START]] to %[[L3STEP]] step %[[L2STEP]]
// CHECK: scf.for %[[KL2:.+]] = %[[START]] to %[[L3STEP]] step %[[L2STEP]]
// CHECK: %[[ARG0_TILE_L2:.+]] = subview %[[ARG0_TILE_L3]][%[[IL2]], %[[KL2]]] [32, 32] [1, 1] : memref<64x64xf32
// CHECK: %[[ARG1_TILE_L2:.+]] = subview %[[ARG1_TILE_L3]][%[[KL2]], %[[JL2]]] [32, 32] [1, 1] : memref<64x64xf32
// CHECK: %[[ARG2_TILE_L2:.+]] = subview %[[ARG2_TILE_L3]][%[[IL2]], %[[JL2]]] [32, 32] [1, 1] : memref<64x64xf32
// CHECK: scf.for %[[IL1:.+]] = %[[START]] to %[[L2STEP]] step %[[L1STEP]]
// CHECK: scf.for %[[JL1:.+]] = %[[START]] to %[[L2STEP]] step %[[L1STEP]]
// CHECK: scf.for %[[KL1:.+]] = %[[START]] to %[[L2STEP]] step %[[L1STEP]]
// CHECK: %[[ARG0_TILE_L1:.+]] = subview %[[ARG0_TILE_L2]][%[[IL1]], %[[KL1]]] [4, 4] [1, 1] : memref<32x32xf32
// CHECK: %[[ARG1_TILE_L1:.+]] = subview %[[ARG1_TILE_L2]][%[[KL1]], %[[JL1]]] [4, 4] [1, 1] : memref<32x32xf32
// CHECK: %[[ARG2_TILE_L1:.+]] = subview %[[ARG2_TILE_L2]][%[[IL1]], %[[JL1]]] [4, 4] [1, 1] : memref<32x32xf32
