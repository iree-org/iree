// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-llvmcpu-mmt4d-vector-lowering,iree-codegen-optimize-vector-transfer{flatten=true}))' %s | FileCheck %s

#target = #hal.executable.target<"llvm-cpu", "system-elf-arm_64", {
    cpu_features = "+dotprod",
    target_triple = "aarch64-none-linux-android29"}>
func.func @mmt4d_kernel_dispatch(%0: memref<1x2x8x4xi8>, %1: memref<1x2x8x4xi8>, %2: memref<1x1x8x8xi32>) attributes {hal.executable.target = #target} {
  %c0_i8 = arith.constant 0 : i8
  %cst = arith.constant dense<0> : vector<1x1x8x8xi32>
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
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
  return
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
