// RUN: iree-opt --iree-codegen-linalg-to-gpu-matmul-vectorization-pass
// RUN: -split-input-file %s --iree-codegen-linalg-to-gpu-unroll-size=8,8,32 \
// RUN: -iree-codegen-linalg-to-gpu-matmul-licm | IreeFileCheck %s

// CHECK-LABEL: func @matmul_128x128x128
// CHECK-SAME: (%[[ARG0:.+]]: memref<128x128xf32>, %[[ARG1:.+]]: memref<128x128xf32>, %[[ARG2:.+]]: memref<128x128xf32>)
func @matmul_128x128x128(%arg0 : memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) {
    linalg.matmul ins(%arg0, %arg1 : memref<128x128xf32>, memref<128x128xf32>) outs(%arg2 : memref<128x128xf32>)
    return
}

// CHECK-DAG: %[[TILESIZE:.+]] = constant 32 : index
// CHECK-DAG: %[[MATSIZE:.+]] = constant 128 : index
// CHECK-DAG: %[[START:.+]] = constant 0 : index
//     CHECK: scf.for %[[IL:.+]] = %[[START]] to %[[MATSIZE]] step %[[TILESIZE]]
//     CHECK:   scf.for %[[JL:.+]] = %[[START]] to %[[MATSIZE]] step %[[TILESIZE]]
//     CHECK:     %[[SUBVVIEWC:.+]] = memref.subview %[[ARG2]][%[[IL]], %[[JL]]] [32, 32] [1, 1] : memref<128x128xf32> to memref<32x32xf32
//     CHECK:     scf.for %[[KL:.+]] = %[[START]] to %[[MATSIZE]] step %[[TILESIZE]]
//     CHECK:       %[[SUBVVIEWA:.+]] = memref.subview %[[ARG0]][%[[IL]], %[[KL]]] [32, 32] [1, 1] : memref<128x128xf32> to memref<32x32xf32
//     CHECK:       %[[SUBVVIEWB:.+]] = memref.subview %[[ARG1]][%[[KL]], %[[JL]]] [32, 32] [1, 1] : memref<128x128xf32> to memref<32x32xf32

