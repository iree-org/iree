// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmcpu-mmt4d-vector-lowering,iree-codegen-optimize-vector-transfer{flatten=true})))))' --split-input-file %s | FileCheck %s

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>

hal.executable private @foo {
hal.executable.variant @system_elf_arm_64 target(<"llvm-cpu", "system-elf-arm_64", {cpu_features = "+dotprod", data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-linux-android29"}>) {
hal.executable.export @foo layout(#pipeline_layout)
builtin.module attributes {llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-none-linux-android29"} {

func.func @mmt4d_kernel_dispatch() {
  %c0_i8 = arith.constant 0 : i8
  %cst = arith.constant dense<0> : vector<1x1x8x8xi32>
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<1x2x8x4xi8>
  memref.assume_alignment %0, 64 : memref<1x2x8x4xi8>
  %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c64) : memref<1x2x8x4xi8>
  memref.assume_alignment %1, 64 : memref<1x2x8x4xi8>
  %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c128) : memref<1x1x8x8xi32>
  memref.assume_alignment %2, 64 : memref<1x1x8x8xi32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 48)>()[%workgroup_id_y]
  %4 = affine.apply affine_map<()[s0] -> (s0 * 48)>()[%workgroup_count_y]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
  %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
  scf.for %arg0 = %3 to %c1 step %4 {
    scf.for %arg1 = %5 to %c1 step %6 {
      vector.transfer_write %cst, %2[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xi32>, memref<1x1x8x8xi32>
      %7 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %cst) -> (vector<1x1x8x8xi32>) {
        %9 = vector.transfer_read %0[%c0, %arg2, %c0, %c0], %c0_i8 {in_bounds = [true, true, true, true]} : memref<1x2x8x4xi8>, vector<1x1x8x4xi8>
        %10 = vector.transfer_read %1[%c0, %arg2, %c0, %c0], %c0_i8 {in_bounds = [true, true, true, true]} : memref<1x2x8x4xi8>, vector<1x1x8x4xi8>
        %11 = arith.extsi %9 : vector<1x1x8x4xi8> to vector<1x1x8x4xi32>
        %12 = arith.extsi %10 : vector<1x1x8x4xi8> to vector<1x1x8x4xi32>
        %13 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %11, %12, %arg3 : vector<1x1x8x4xi32>, vector<1x1x8x4xi32> into vector<1x1x8x8xi32>
        scf.yield %13 : vector<1x1x8x8xi32>
      }
      vector.transfer_write %7, %2[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xi32>, memref<1x1x8x8xi32>
      %8 = memref.subview %2[%arg0, %arg1, 0, 0] [1, 1, 8, 8] [1, 1, 1, 1] : memref<1x1x8x8xi32> to memref<1x1x8x8xi32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 64 + s0 + d1 * 64 + d2 * 8 + d3)>>
      linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : memref<1x1x8x8xi32>) outs(%8 : memref<1x1x8x8xi32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 64 + s0 + d1 * 64 + d2 * 8 + d3)>>) {
      ^bb0(%arg2: i32, %arg3: i32):
        linalg.yield %arg2 : i32
      }
    }
  }
  return
}

}
}
}

// CHECK-LABEL:  @mmt4d_kernel_dispatch(
// CHECK:        %[[LHS_FLAT32:.+]] = vector.transfer_read {{.*}} : memref<1x2x32xi8>, vector<32xi8>
// CHECK:        %[[RHS_FLAT32:.+]] = vector.transfer_read {{.*}} : memref<1x2x32xi8>, vector<32xi8>
// CHECK:        %[[LHS_FLAT16_0:.+]] = vector.extract_strided_slice %[[LHS_FLAT32]] {offsets = [0], sizes = [16], strides = [1]} : vector<32xi8> to vector<16xi8>
// CHECK:        %[[LHS_FLAT16_1:.+]] = vector.extract_strided_slice %[[LHS_FLAT32]] {offsets = [16], sizes = [16], strides = [1]} : vector<32xi8> to vector<16xi8>
// CHECK:        %[[RHS_FLAT16_0:.+]] = vector.extract_strided_slice %[[RHS_FLAT32]] {offsets = [0], sizes = [16], strides = [1]} : vector<32xi8> to vector<16xi8>
// CHECK:        %[[RHS_FLAT16_1:.+]] = vector.extract_strided_slice %[[RHS_FLAT32]] {offsets = [16], sizes = [16], strides = [1]} : vector<32xi8> to vector<16xi8>
// CHECK:        llvm.inline_asm
// CHECK-SAME:      {{((.*sdot){16})}}
// CHECK-SAME:      %[[LHS_FLAT16_0]], %[[LHS_FLAT16_1]], %[[RHS_FLAT16_0]], %[[RHS_FLAT16_1]],
