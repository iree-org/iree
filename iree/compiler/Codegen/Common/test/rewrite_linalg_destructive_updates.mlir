// RUN: iree-opt -iree-codegen-rewrite-linalg-destructive-updates %s | FileCheck %s

func @matmul() {
  %cst = arith.constant 0.000000e+00 : f32
  %c786432 = arith.constant 786432 : index
  %c0 = arith.constant 0 : index
  %c384 = arith.constant 384 : index
  %c128 = arith.constant 128 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x1536xf32>
  %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c786432) alignment(64) : !flow.dispatch.tensor<readonly:1536x384xf32>
  %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:128x384xf32>
  %3 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : !flow.dispatch.tensor<writeonly:128x384xf32> -> tensor<128x384xf32>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 1536], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x1536xf32> -> tensor<128x1536xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1536, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:1536x384xf32> -> tensor<1536x384xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %6 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
  %7 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
  %8 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
  %9 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
  %10 = scf.for %arg0 = %6 to %c128 step %7 iter_args(%arg1 = %3) -> (tensor<128x384xf32>) {
    %11 = tensor.extract_slice %4[%arg0, 0] [64, 1536] [1, 1] : tensor<128x1536xf32> to tensor<64x1536xf32>
    %12 = scf.for %arg2 = %8 to %c384 step %9 iter_args(%arg3 = %arg1) -> (tensor<128x384xf32>) {
      %13 = tensor.extract_slice %5[0, %arg2] [1536, 64] [1, 1] : tensor<1536x384xf32> to tensor<1536x64xf32>
      %14 = tensor.extract_slice %arg3[%arg0, %arg2] [64, 64] [1, 1] : tensor<128x384xf32> to tensor<64x64xf32>
      %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %16 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [8, 32, 0], [0, 0, 16]]>} ins(%11, %13 : tensor<64x1536xf32>, tensor<1536x64xf32>) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %17 = tensor.insert_slice %16 into %arg3[%arg0, %arg2] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x384xf32>
      scf.yield %17 : tensor<128x384xf32>
    }
    scf.yield %12 : tensor<128x384xf32>
  }
  flow.dispatch.tensor.store %10, %2, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : tensor<128x384xf32> -> !flow.dispatch.tensor<writeonly:128x384xf32>
  return
}
// CHECK-LABEL: func @matmul
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             %[[MATMUL:.+]] = linalg.matmul
// CHECK:             flow.dispatch.tensor.store %[[MATMUL]]
