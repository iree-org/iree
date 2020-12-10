// RUN: iree-opt -iree-codegen-llvm-linalg-tile-and-distribute -iree-codegen-linalg-to-llvm-workgroups-vectorization-pass -split-input-file %s | IreeFileCheck %s

func @matmul_128x128x128(%arg0 : memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) {
    linalg.matmul ins(%arg0, %arg1 : memref<128x128xf32>, memref<128x128xf32>) outs(%arg2 : memref<128x128xf32>)
    return
}
// CHECK: #[[MAP0:map.*]] =  affine_map<()[s0] -> (s0 * 64)>
// CHECK-LABEL: func @matmul_128x128x128
// CHECK-SAME: (%[[ARG0:.+]]: memref<128x128xf32>, %[[ARG1:.+]]: memref<128x128xf32>, %[[ARG2:.+]]: memref<128x128xf32>)
// CHECK-DaG: %[[WORKGROUP_TILE_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-DAG: %[[WORKGROUP_TILE_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK-DAG: %[[START:.+]] = constant 0
// CHECK-DAG: %[[WORGKROUP_SIZE:.+]] = constant 64
// CHECK-DAG: %[[VECTOR_SIZE:.+]] = constant 4
// CHECK-DAG: %[[L1_SIZE:.+]] = constant 32
// CHECK-DAG: %[[KDIM_SIZE:.+]] = constant 128
// CHECK:     scf.for {{.*}} = %[[START]] to %[[WORGKROUP_SIZE]] step %[[L1_SIZE]] {
// CHECK:       scf.for {{.*}} = %[[START]] to %[[WORGKROUP_SIZE]] step %[[L1_SIZE]] {
// CHECK:         scf.for {{.*}} = %[[START]] to %[[KDIM_SIZE]] step %[[L1_SIZE]] {
// CHECK:           scf.for {{.*}} = %[[START]] to %[[L1_SIZE]] step %[[VECTOR_SIZE]] {
// CHECK:             scf.for {{.*}} = %[[START]] to %[[L1_SIZE]] step %[[VECTOR_SIZE]] {
// CHECK:               %[[VEC_C_0:.+]] = vector.transfer_read %[[ARG2]]
// CHECK:               %[[VEC_C_1:.+]] = vector.transfer_read %[[ARG2]]
// CHECK:               %[[VEC_C_2:.+]] = vector.transfer_read %[[ARG2]]
// CHECK:               %[[VEC_C_3:.+]] = vector.transfer_read %[[ARG2]]
// CHECK:                  scf.for {{.*}} = %[[START]] to %[[L1_SIZE]] step %[[VECTOR_SIZE]]
// CHECK:                    %[[VEC_A_0:.+]] = vector.transfer_read %[[ARG0]]
// CHECK:                    %[[VEC_A_1:.+]] = vector.transfer_read %[[ARG0]]
// CHECK:                    %[[VEC_A_2:.+]] = vector.transfer_read %[[ARG0]]
// CHECK:                    %[[VEC_A_3:.+]] = vector.transfer_read %[[ARG0]]
// CHECK:                    %[[VEC_B_0:.+]] = vector.transfer_read %[[ARG1]]
// CHECK:                    %[[VEC_b_1:.+]] = vector.transfer_read %[[ARG1]]
// CHECK:                    %[[VEC_B_2:.+]] = vector.transfer_read %[[ARG1]]
// CHECK:                    %[[VEC_B_3:.+]] = vector.transfer_read %[[ARG1]]
