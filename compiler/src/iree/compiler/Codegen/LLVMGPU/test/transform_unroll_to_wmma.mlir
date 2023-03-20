// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule | FileCheck %s

hal.executable private @img2col_conv  {
builtin.module {
// CHECK-LABEL: func.func @img2col_conv
func.func @img2col_conv() {
  %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
  %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : vector<32xindex>
  %c32 = arith.constant 32 : index
  %cst_1 = arith.constant dense<3> : vector<16x32xindex>
  %cst_2 = arith.constant dense<9> : vector<16x32xindex>
  %cst_3 = arith.constant dense<128> : vector<32xindex>
  %cst_4 = arith.constant dense<true> : vector<16x32xi1>
  %cst_5 = arith.constant dense<0.000000e+00> : vector<16x32xf16>
  %cst_6 = arith.constant dense<16> : vector<16x32xindex>
  %cst_7 = arith.constant dense<130> : vector<16x32xindex>
  %cst_8 = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c144 = arith.constant 144 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x16x130x130xf16>>
  %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<2x32x16384xf16>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 16, 130, 130], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x16x130x130xf16>> -> tensor<2x16x130x130xf16>
  %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x144xf16>>
  %4 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [32, 144], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x144xf16>> -> tensor<32x144xf16>
  %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<2x32x16384xf16>>
  %6 = flow.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [2, 32, 16384], strides = [1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<2x32x16384xf16>> -> tensor<2x32x16384xf16>
  %7 = scf.forall (%arg0, %arg1, %arg2) in (2, 1, 512) shared_outs(%arg3 = %6) -> (tensor<2x32x16384xf16>) {
    %8 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg1)
    %9 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg2)
    %extracted_slice = tensor.extract_slice %4[%8, 0] [32, 144] [1, 1] : tensor<32x144xf16> to tensor<32x144xf16>
    %extracted_slice_9 = tensor.extract_slice %arg3[%arg0, %8, %9] [1, 32, 32] [1, 1, 1] : tensor<2x32x16384xf16> to tensor<1x32x32xf16>
    %10 = scf.for %arg4 = %c0 to %c144 step %c16 iter_args(%arg5 = %extracted_slice_9) -> (tensor<1x32x32xf16>) {
      %extracted_slice_10 = tensor.extract_slice %extracted_slice[0, %arg4] [32, 16] [1, 1] : tensor<32x144xf16> to tensor<32x16xf16>
      %11 = tensor.empty() : tensor<1x16x32xf16>
//    CHECK: vector.broadcast {{.*}} : vector<16xindex> to vector<32x16xindex>
//    CHECK: vector.transpose {{.*}} : vector<32x16xindex> to vector<16x32xindex>
//    CHECK: vector.broadcast {{.*}} : index to vector<16x32xindex>
//    CHECK: arith.addi {{.*}} : vector<16x32xindex>
//    CHECK: arith.muli {{.*}} : index
//    CHECK: vector.broadcast {{.*}} : index to vector<32xindex>
//    CHECK: arith.addi {{.*}} : vector<32xindex>
//    CHECK: arith.remui {{.*}} : vector<16x32xindex>
//    CHECK: arith.remui {{.*}} : vector<16x32xindex>
//    CHECK: arith.divui {{.*}} : vector<16x32xindex>
//    CHECK: arith.divui {{.*}} : vector<16x32xindex>
//    CHECK: arith.remui {{.*}} : vector<32xindex>
//    CHECK: arith.divui {{.*}} : vector<32xindex>
//    CHECK: vector.broadcast {{.*}} : vector<32xindex> to vector<16x32xindex>
//    CHECK: arith.addi {{.*}} : vector<16x32xindex>
//    CHECK: vector.broadcast {{.*}} : vector<32xindex> to vector<16x32xindex>
//    CHECK: arith.addi {{.*}} : vector<16x32xindex>
//    CHECK: vector.broadcast {{.*}} : index to vector<16x32xindex>
//    CHECK: arith.muli {{.*}} : vector<16x32xindex>
//    CHECK: arith.addi {{.*}} : vector<16x32xindex>
//    CHECK: arith.muli {{.*}} : vector<16x32xindex>
//    CHECK: arith.addi {{.*}} : vector<16x32xindex>
//    CHECK: arith.muli {{.*}} : vector<16x32xindex>
//    CHECK: arith.addi {{.*}} : vector<16x32xindex>
      %12 = vector.broadcast %cst : vector<16xindex> to vector<32x16xindex>
      %13 = vector.transpose %12, [1, 0] : vector<32x16xindex> to vector<16x32xindex>
      %14 = vector.broadcast %arg4 : index to vector<16x32xindex>
      %15 = arith.addi %13, %14 : vector<16x32xindex>
      %16 = arith.muli %arg2, %c32 : index
      %17 = vector.broadcast %16 : index to vector<32xindex>
      %18 = arith.addi %17, %cst_0 : vector<32xindex>
      %19 = arith.remui %15, %cst_1 : vector<16x32xindex>
      %20 = arith.remui %15, %cst_2 : vector<16x32xindex>
      %21 = arith.divui %20, %cst_1 : vector<16x32xindex>
      %22 = arith.divui %15, %cst_2 : vector<16x32xindex>
      %23 = arith.remui %18, %cst_3 : vector<32xindex>
      %24 = arith.divui %18, %cst_3 : vector<32xindex>
      %25 = vector.broadcast %24 : vector<32xindex> to vector<16x32xindex>
      %26 = arith.addi %25, %21 : vector<16x32xindex>
      %27 = vector.broadcast %23 : vector<32xindex> to vector<16x32xindex>
      %28 = arith.addi %27, %19 : vector<16x32xindex>
      %29 = vector.broadcast %arg0 : index to vector<16x32xindex>
      %30 = arith.muli %29, %cst_6 : vector<16x32xindex>
      %31 = arith.addi %22, %30 : vector<16x32xindex>
      %32 = arith.muli %31, %cst_7 : vector<16x32xindex>
      %33 = arith.addi %26, %32 : vector<16x32xindex>
      %34 = arith.muli %33, %cst_7 : vector<16x32xindex>
      %35 = arith.addi %28, %34 : vector<16x32xindex>
      %36 = vector.gather %2[%c0, %c0, %c0, %c0] [%35], %cst_4, %cst_5 : tensor<2x16x130x130xf16>, vector<16x32xindex>, vector<16x32xi1>, vector<16x32xf16> into vector<16x32xf16>
      %37 = vector.transfer_write %36, %11[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<16x32xf16>, tensor<1x16x32xf16>
      %38 = bufferization.alloc_tensor() copy(%extracted_slice_10) {bufferization.escape = [false]} : tensor<32x16xf16>
      %39 = scf.forall (%arg6) in (1) shared_outs(%arg7 = %arg5) -> (tensor<1x32x32xf16>) {
        %40 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg6)
        %extracted_slice_11 = tensor.extract_slice %arg7[0, 0, %40] [1, 32, 32] [1, 1, 1] : tensor<1x32x32xf16> to tensor<1x32x32xf16>
        // CHECK-8: vector.transfer_read {{.*}} : tensor<1x32x32xf16>, vector<16x16xf16>
        // CHECK-4: vector.contract {{.*}} : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
        // CHECK-4: vector.transfer_write {{.*}} : vector<16x16xf16>, tensor<1x32x32xf16>
        %41 = vector.transfer_read %38[%c0, %c0], %cst_8 {in_bounds = [true, true]} : tensor<32x16xf16>, vector<32x16xf16>
        %42 = vector.transfer_read %37[%c0, %c0, %40], %cst_8 {in_bounds = [true, true]} : tensor<1x16x32xf16>, vector<16x32xf16>
        %43 = vector.transfer_read %arg7[%c0, %c0, %40], %cst_8 {in_bounds = [true, true]} : tensor<1x32x32xf16>, vector<32x32xf16>
        %44 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %41, %42, %43 : vector<32x16xf16>, vector<16x32xf16> into vector<32x32xf16>
        %45 = vector.transfer_write %44, %extracted_slice_11[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<32x32xf16>, tensor<1x32x32xf16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %45 into %arg7[0, 0, %40] [1, 32, 32] [1, 1, 1] : tensor<1x32x32xf16> into tensor<1x32x32xf16>
        }
      } {mapping = [#gpu.warp<x>]}
      scf.yield %39 : tensor<1x32x32xf16>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %10 into %arg3[%arg0, %8, %9] [1, 32, 32] [1, 1, 1] : tensor<1x32x32xf16> into tensor<2x32x16384xf16>
    }
  } {mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>]}
  flow.dispatch.tensor.store %7, %1, offsets = [0, 0, 0], sizes = [2, 32, 16384], strides = [1, 1, 1] : tensor<2x32x16384xf16> -> !flow.dispatch.tensor<readwrite:tensor<2x32x16384xf16>>
  return
}
}
transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %forallops = transform.structured.match ops{["scf.forall"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %matmul_forall, %block_forall = transform.split_handles %forallops in[2] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  transform.iree.apply_patterns_to_nested %matmul_forall { unroll_vectors_gpu_wmma } : (!pdl.operation) -> ()
}
}
