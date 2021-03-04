// TODO(#4901): Convert these tests back to use dynamic shapes when linalg on tensors becomes default.
// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-llvm-linalg-tile-and-distribute)),hal.executable(hal.executable.target(module(func(iree-codegen-linalg-to-llvm-workgroups-vectorization-pass))))" -split-input-file %s | IreeFileCheck %s
// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-llvm-linalg-tile-and-distribute)),hal.executable(hal.executable.target(module(func(iree-codegen-linalg-to-llvm-workgroups-vectorization-pass))))" -split-input-file -iree-codegen-llvm-promote-workgroup-to-full-tiles -cse %s | IreeFileCheck %s -check-prefix=CHECK-PROMOTED
hal.executable @dynamic_matmul attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @llvm_aot, filter="dylib*" {
    hal.executable.entry_point @matmul_128x128x128 attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<128x128xf32>, !flow.dispatch.input<128x128xf32>,
        !flow.dispatch.output<128x128xf32>) -> ()}
    module {
      func @matmul_128x128x128(%arg0 : memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) {
        linalg.matmul ins(%arg0, %arg1 : memref<128x128xf32>, memref<128x128xf32>) outs(%arg2 : memref<128x128xf32>)
        return
      }
    }
  }
}
// CHECK-LABEL: func @matmul_128x128x128(
// CHECK-SAME: %[[ARG0:.+]]: memref<128x128xf32>, %[[ARG1:.+]]: memref<128x128xf32>, %[[ARG2:.+]]: memref<128x128xf32>)
// CHECK-DAG: %[[WORKGROUP_TILE_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-DAG: %[[WORKGROUP_TILE_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK-DAG: %[[START:.+]] = constant 0 : index
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

// CHECK-PROMOTED: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 * 64)>
// CHECK-PROMOTED: #[[MAP1:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>
// CHECK-PROMOTED: func @matmul_128x128x128
// CHECK-PROMOTED: (%[[ARG0:.+]]: memref<128x128xf32>, %[[ARG1:.+]]: memref<128x128xf32>, %[[ARG2:.+]]: memref<128x128xf32>) {
// CHECK-PROMOTED: %[[KDIM_SIZE:.+]] = constant 128 : index
// CHECK-PROMOTED: %[[WORGKROUP_SIZE:.+]] = constant 64 : index
// CHECK-PROMOTED: %[[VECTOR_SIZE:.+]] = constant 4 : index
// CHECK-PROMOTED: %[[L1_SIZE:.+]] = constant 32 : index
// CHECK-PROMOTED: %[[START:.+]] = constant 0 : index
// CHECK-PROMOTED: %[[C1:.+]] = constant 1 : index
// CHECK-PROMOTED: %[[C1:.+]] = constant 2 : index
// CHECK-PROMOTED: %[[C1:.+]] = constant 3 : index
// CHECK-PROMOTED: %[[A_PROMOTED_TILE:.+]] = alloca() : memref<64x64xf32>
// CHECK-PROMOTED: %[[B_PROMOTED_TILE:.+]] = alloca() : memref<128x64xf32>
// CHECK-PROMOTED: %[[C_PROMOTED_TILE:.+]] = alloca() : memref<64x128xf32>
// CHECK-PROMOTED: %[[WORKGROUP_TILE_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-PROMOTED: %[[WORKGROUP_TILE_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK-PROMOTED: %[[WORKGROUP_ID_Y_INDEX:.+]] = affine.apply #[[MAP0]]()[%[[WORKGROUP_TILE_Y]]]
// CHECK-PROMOTED: %[[A_TILE:.+]] = subview %[[ARG0]][%[[WORKGROUP_ID_Y_INDEX]], 0] [64, 128] [1, 1] : memref<128x128xf32> to memref<64x128xf32, #[[MAP1]]>
// CHECK-PROMOTED: %[[WORKGROUP_ID_X_INDEX:.+]] = affine.apply #[[MAP0]]()[%[[WORKGROUP_TILE_X]]]
// CHECK-PROMOTED: %[[B_TILE:.+]] = subview %[[ARG1]][0, %[[WORKGROUP_ID_X_INDEX]]] [128, 64] [1, 1] : memref<128x128xf32> to memref<128x64xf32, #[[MAP1]]>
// CHECK-PROMOTED: %[[C_TILE:.+]] = subview %[[ARG2]][%[[WORKGROUP_ID_Y_INDEX]], %[[WORKGROUP_ID_X_INDEX]]] [64, 64] [1, 1] : memref<128x128xf32> to memref<64x64xf32, #[[MAP1]]>
// CHECK-PROMOTED:     scf.for {{.*}} = %[[START]] to %[[WORGKROUP_SIZE]] step %[[L1_SIZE]] {
// CHECK-PROMOTED:       scf.for {{.*}} = %[[START]] to %[[WORGKROUP_SIZE]] step %[[L1_SIZE]] {
// CHECK-PROMOTED:         scf.for {{.*}} = %[[START]] to %[[KDIM_SIZE]] step %[[L1_SIZE]] {
// CHECK-PROMOTED:           scf.for {{.*}} = %[[START]] to %[[L1_SIZE]] step %[[VECTOR_SIZE]] {
// CHECK-PROMOTED:             scf.for {{.*}} = %[[START]] to %[[L1_SIZE]] step %[[VECTOR_SIZE]] {
// CHECK-PROMOTED:               %[[VEC_C_0:.+]] = vector.transfer_read %[[C_PROMOTED_TILE]]
// CHECK-PROMOTED:               %[[VEC_C_1:.+]] = vector.transfer_read %[[C_PROMOTED_TILE]]
// CHECK-PROMOTED:               %[[VEC_C_2:.+]] = vector.transfer_read %[[C_PROMOTED_TILE]]
// CHECK-PROMOTED:               %[[VEC_C_3:.+]] = vector.transfer_read %[[C_PROMOTED_TILE]]

// -----

hal.executable @dynamic_matmul_i8_i8_i32 attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @llvm_aot, filter="dylib*" {
    hal.executable.entry_point @matmul_i8_i8_i32_128x128x128 attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<128x128xi8>, !flow.dispatch.input<128x128xi8>,
        !flow.dispatch.output<128x128xi32>) -> ()}
    module {
      func @matmul_i8_i8_i32_128x128x128(%arg0 : memref<128x128xi8>, %arg1: memref<128x128xi8>, %arg2: memref<128x128xi32>) {
        linalg.matmul_i8_i8_i32 ins(%arg0, %arg1 : memref<128x128xi8>, memref<128x128xi8>) outs(%arg2 : memref<128x128xi32>)
        return
      }
    }
  }
}
// CHECK-LABEL: func @matmul_i8_i8_i32_128x128x128(
// CHECK-SAME: %[[ARG0:.+]]: memref<128x128xi8>, %[[ARG1:.+]]: memref<128x128xi8>, %[[ARG2:.+]]: memref<128x128xi32>)
// CHECK-DAG: %[[WORKGROUP_TILE_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-DAG: %[[WORKGROUP_TILE_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK-DAG: %[[START:.+]] = constant 0 : index
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
