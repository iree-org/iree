// RUN: iree-opt %s --allow-unregistered-dialect --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-vector-alloc))" | FileCheck %s

func.func @matmul_256x256x256(%lhs: tensor<16x256xf16>,
                              %rhs: tensor<256x16xf16>,
                              %out: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %cst = arith.constant 0.000000e+00 : f16 
  %cst_0 = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %c32 = arith.constant 32 : index 
  %c256 = arith.constant 256 : index 
  %c0 = arith.constant 0 : index 
  %8 = scf.for %arg0 = %c0 to %c256 step %c32 iter_args(%arg1 = %cst_0) -> (vector<16x16xf32>) {
    %10 = vector.transfer_read %lhs[%c0, %arg0], %cst {in_bounds = [true, true]} : tensor<16x256xf16>, vector<16x32xf16>
    %11 = vector.transfer_read %rhs[%arg0, %c0], %cst {in_bounds = [true, true]} : tensor<256x16xf16>, vector<32x16xf16>
    %12 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %10, %11, %arg1 : vector<16x32xf16>, vector<32x16xf16> into vector<16x16xf32>
    scf.yield %12 : vector<16x16xf32>
  }
  %9 = vector.transfer_write %8, %out[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, tensor<16x16xf32>
  return %9 : tensor<16x16xf32>
}


//    CHECK-LABEL: func.func @matmul_256x256x256
//         CHECK:    scf.for {{.*}} -> (vector<16x16xf32>) {
//     CHECK-DAG:      %[[A:.*]] = vector.transfer_read %{{.*}} : tensor<16x256xf16>, vector<16x32xf16>
//     CHECK-DAG:      %[[B:.*]] = vector.transfer_read %{{.*}} : tensor<256x16xf16>, vector<32x16xf16>

// LHS copy.
//         CHECK:      %[[PA:.*]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16x32xf16, #gpu.address_space<workgroup>>
//         CHECK:      %[[LWRITE:.+]] = vector.transfer_write %[[A]], %[[PA]]{{.*}} : vector<16x32xf16>, tensor<16x32xf16, #gpu.address_space<workgroup>>
//         CHECK:      %[[LCOPY:.+]] = bufferization.materialize_in_destination %[[LWRITE]] in %[[LWRITE]]

// RHS copy.
//         CHECK:      %[[PB:.*]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<32x16xf16, #gpu.address_space<workgroup>>
//         CHECK:      %[[RWRITE:.+]] = vector.transfer_write %[[B]], %[[PB]]{{.*}} : vector<32x16xf16>, tensor<32x16xf16, #gpu.address_space<workgroup>>
//         CHECK:      %[[RCOPY:.+]] = bufferization.materialize_in_destination %[[RWRITE]] in %[[RWRITE]]
//         CHECK:      gpu.barrier

//         CHECK:      %[[LHS:.+]] = vector.transfer_read %[[LCOPY]]{{.*}} : tensor<16x32xf16, #gpu.address_space<workgroup>>, vector<16x32xf16>
//         CHECK:      %[[RHS:.+]] = vector.transfer_read %[[RCOPY]]{{.*}} : tensor<32x16xf16, #gpu.address_space<workgroup>>, vector<32x16xf16>
//         CHECK:      %12 = vector.contract {{.*}} %[[LHS]], %[[RHS]], %{{.*}}
