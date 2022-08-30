// RUN: iree-opt --split-input-file --pass-pipeline='hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-tile-tensor))))' -cse %s | FileCheck %s

hal.executable private @add_tensor  {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @add_tensor ordinal(0)
  layout(#hal.pipeline.layout<push_constants = 0,
         sets = [<0, bindings = [<0, storage_buffer>, <1, storage_buffer>, <2, storage_buffer>]>]>)
         attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>,
                     workgroup_size = [64 : index, 1 : index, 1 : index]} {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.default_workgroup_count %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @add_tensor() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:233x1024xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:233x1024xf32>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:233x1024xf32>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %3 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_id_x]
      %4 = flow.dispatch.tensor.load %2, offsets = [%workgroup_id_y, %3], sizes = [1, 256], strides = [1, 1] : !flow.dispatch.tensor<writeonly:233x1024xf32> -> tensor<1x256xf32>
      %5 = flow.dispatch.tensor.load %0, offsets = [%workgroup_id_y, %3], sizes = [1, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:233x1024xf32> -> tensor<1x256xf32>
      %6 = flow.dispatch.tensor.load %1, offsets = [%workgroup_id_y, %3], sizes = [1, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:233x1024xf32> -> tensor<1x256xf32>
      %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %6 : tensor<1x256xf32>, tensor<1x256xf32>) outs(%4 : tensor<1x256xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 256]]>} {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
        %8 = arith.addf %arg0, %arg1 : f32
        linalg.yield %8 : f32
      } -> tensor<1x256xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [%workgroup_id_y, %3], sizes = [1, 256], strides = [1, 1] : tensor<1x256xf32> -> !flow.dispatch.tensor<writeonly:233x1024xf32>
      return
    }
  }
}
}

//         CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0 * 4)>
//   CHECK-LABEL: func.func @add_tensor
//         CHECK:   %[[C64:.*]] = arith.constant 64 : index
//         CHECK:   %[[T:.*]] = scf.foreach_thread (%[[ARG:.*]]) in (%[[C64]]) -> (tensor<1x256xf32>) {
//         CHECK:     %[[OFF:.*]] = affine.apply #[[$MAP]](%[[ARG]])
//         CHECK:     %[[C:.*]] = tensor.extract_slice %{{.*}}[0, %[[OFF]]] [1, 4] [1, 1] : tensor<1x256xf32> to tensor<1x4xf32>
//         CHECK:     %[[A:.*]] = tensor.extract_slice %{{.*}}[0, %[[OFF]]] [1, 4] [1, 1] : tensor<1x256xf32> to tensor<1x4xf32>
//         CHECK:     %[[B:.*]] = tensor.extract_slice %{{.*}}[0, %[[OFF]]] [1, 4] [1, 1] : tensor<1x256xf32> to tensor<1x4xf32>
//         CHECK:     %[[L:.*]] = linalg.generic {{.*}} ins(%[[A]], %[[B]] : tensor<1x4xf32>, tensor<1x4xf32>) outs(%[[C]] : tensor<1x4xf32>)
//         CHECK:       %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
//         CHECK:       linalg.yield %{{.*}} : f32
//         CHECK:     } -> tensor<1x4xf32>
//         CHECK:     scf.foreach_thread.perform_concurrently {
//         CHECK:       tensor.parallel_insert_slice %[[L]] into %{{.*}}[0, %[[OFF]]] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<1x256xf32>
//         CHECK:     }
//         CHECK:  } {thread_dim_mapping = [0, 1, 2]}
//         CHECK:   flow.dispatch.tensor.store %[[T]], %{{.*}}, offsets = [%{{.*}}, %{{.*}}], sizes = [1, 256], strides = [1, 1] : tensor<1x256xf32> -> !flow.dispatch.tensor<writeonly:233x1024xf32>

// -----

hal.executable private @reduction  {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @reduction ordinal(0)
  layout(#hal.pipeline.layout<push_constants = 0,
         sets = [<0, bindings = [<0, storage_buffer>, <1, storage_buffer>, <2, storage_buffer>]>]>)
         attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>,
                     workgroup_size = [64 : index, 1 : index, 1 : index]} {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.default_workgroup_count %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x384xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:128xf32>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %2 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
      %3 = flow.dispatch.tensor.load %1, offsets = [%2], sizes = [64], strides = [1] : !flow.dispatch.tensor<writeonly:128xf32> -> tensor<64xf32>
      %4 = flow.dispatch.tensor.load %0, offsets = [%2, 0], sizes = [64, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x384xf32> -> tensor<64x384xf32>
      %5 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 4]]>} ins(%cst : f32) outs(%3 : tensor<64xf32>) -> tensor<64xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%4 : tensor<64x384xf32>) outs(%5 : tensor<64xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 4]]>} {
      ^bb0(%arg0: f32, %arg1: f32):
        %7 = arith.addf %arg0, %arg1 : f32
        linalg.yield %7 : f32
      } -> tensor<64xf32>
      flow.dispatch.tensor.store %6, %1, offsets = [%2], sizes = [64], strides = [1] : tensor<64xf32> -> !flow.dispatch.tensor<writeonly:128xf32>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @reduction
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//     CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
//     CHECK-DAG:   %[[C384:.*]] = arith.constant 384 : index
//         First the scf.foreach for the linalg.fill.
//         CHECK:   scf.foreach_thread
//         then the reduction case.
//         CHECK:   %[[T:.*]] = scf.foreach_thread (%[[ARG:.*]]) in (%[[C64]]) -> (tensor<64xf32>) {
//         CHECK:     %[[A:.*]] = tensor.extract_slice %{{.*}}[%[[ARG]]] [1] [1] : tensor<64xf32> to tensor<1xf32>
//         CHECK:     %[[R:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C384]] step %[[C4]] iter_args(%[[ACC:.*]] = %[[A]]) -> (tensor<1xf32>) {
//         CHECK:       %[[E:.*]] = tensor.extract_slice %4[%[[ARG]], %[[IV]]] [1, 4] [1, 1] : tensor<64x384xf32> to tensor<1x4xf32>
//         CHECK:       %[[L:.*]] = linalg.generic {{.*}} ins(%[[E]] : tensor<1x4xf32>) outs(%[[ACC]] : tensor<1xf32>)
//         CHECK:         arith.addf
//         CHECK:         linalg.yield %{{.*}} : f32
//         CHECK:       } -> tensor<1xf32>
//         CHECK:       scf.yield %[[L]] : tensor<1xf32>
//         CHECK:     }
//         CHECK:     scf.foreach_thread.perform_concurrently {
//         CHECK:       tensor.parallel_insert_slice %[[R]] into %{{.*}}[%[[ARG]]] [1] [1] : tensor<1xf32> into tensor<64xf32>
//         CHECK:     }
//         CHECK:   } {thread_dim_mapping = [0, 1, 2]}
//         CHECK:   flow.dispatch.tensor.store %[[T]], %{{.}}, offsets = [%{{.*}}], sizes = [64], strides = [1] : tensor<64xf32> -> !flow.dispatch.tensor<writeonly:128xf32>
