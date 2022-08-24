// RUN: iree-opt %s -iree-llvmgpu-alloc | FileCheck %s

func.func @matmul_2048x512x1024() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:2048x1024xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:1024x512xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:2048x512xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
  %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
  %5 = flow.dispatch.tensor.load %2, offsets = [%3, %4], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<writeonly:2048x512xf32> -> tensor<32x128xf32>
  %6 = flow.dispatch.tensor.load %0, offsets = [%3, 0], sizes = [32, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:2048x1024xf32> -> tensor<32x1024xf32>
  %7 = flow.dispatch.tensor.load %1, offsets = [0, %4], sizes = [1024, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:1024x512xf32> -> tensor<1024x128xf32>
  %8 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>} ins(%cst : f32) outs(%5 : tensor<32x128xf32>) -> tensor<32x128xf32>
  %9 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>} ins(%6, %7 : tensor<32x1024xf32>, tensor<1024x128xf32>) outs(%8 : tensor<32x128xf32>) -> tensor<32x128xf32>
  flow.dispatch.tensor.store %9, %2, offsets = [%3, %4], sizes = [32, 128], strides = [1, 1] : tensor<32x128xf32> -> !flow.dispatch.tensor<writeonly:2048x512xf32>
  return
}

//    CHECK-LABEL: func.func @matmul_2048x512x1024
//         CHECK:    scf.for {{.*}} -> (tensor<32x128xf32>) {
//         CHECK:      %[[A:.*]] = tensor.extract_slice %{{.*}}[0, %{{.*}}] [32, 32] [1, 1] : tensor<32x1024xf32> to tensor<32x32xf32>
//         CHECK:      %[[B:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [32, 128] [1, 1] : tensor<1024x128xf32> to tensor<32x128xf32>
//         CHECK:      %[[PA:.*]] = bufferization.alloc_tensor() copy(%[[A]]) {bufferization.escape = [false]} : tensor<32x32xf32>
//         CHECK:      %[[PB:.*]] = bufferization.alloc_tensor() copy(%[[B]]) {bufferization.escape = [false]} : tensor<32x128xf32>
//         CHECK:      %[[M:.*]] = linalg.matmul {{.*}} ins(%[[PA]], %[[PB]] : tensor<32x32xf32>, tensor<32x128xf32>) outs(%{{.*}} : tensor<32x128xf32>) -> tensor<32x128xf32>
//         CHECK:      scf.yield %[[M]] : tensor<32x128xf32>
