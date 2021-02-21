// RUN: iree-opt -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-llvm-linalg-tile-and-distribute)),hal.executable(hal.executable.target(module(func(iree-codegen-linalg-to-llvm-workgroups-vectorization-pass))))" -split-input-file %s | IreeFileCheck %s

// TODO(GH-4901): Convert these tests back to use dynamic shapes when linalg on tensors becomes default.
hal.executable @matmul_128x128x128 attributes {sym_visibility = "private"} {
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
// CHECK-LABEL: func @matmul_128x128x128
// CHECK-SAME: (%[[ARG0:.+]]: memref<128x128xf32>, %[[ARG1:.+]]: memref<128x128xf32>, %[[ARG2:.+]]: memref<128x128xf32>)
// CHECK-DaG: %[[WORKGROUP_TILE_X:.+]] = hal.interface.workgroup.id[0] : index
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
