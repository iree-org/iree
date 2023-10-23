// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-gpu-tensor-tile, cse)))))" %s | FileCheck %s

hal.executable private @add_tensor  {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>) {
  hal.executable.export public @add_tensor ordinal(0)
  layout(#hal.pipeline.layout<push_constants = 0,
         sets = [<0, bindings = [<0, storage_buffer>, <1, storage_buffer>, <2, storage_buffer>]>]>)
         attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>,
                     workgroup_size = [64 : index, 1 : index, 1 : index]} {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @add_tensor() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<233x1024xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<233x1024xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<233x1024xf32>>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %3 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_id_x]
      %4 = flow.dispatch.tensor.load %2, offsets = [%workgroup_id_y, %3], sizes = [1, 256], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<233x1024xf32>> -> tensor<1x256xf32>
      %5 = flow.dispatch.tensor.load %0, offsets = [%workgroup_id_y, %3], sizes = [1, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<233x1024xf32>> -> tensor<1x256xf32>
      %6 = flow.dispatch.tensor.load %1, offsets = [%workgroup_id_y, %3], sizes = [1, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<233x1024xf32>> -> tensor<1x256xf32>
      %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %6 : tensor<1x256xf32>, tensor<1x256xf32>) outs(%4 : tensor<1x256xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 256]]>} {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
        %8 = arith.addf %arg0, %arg1 : f32
        linalg.yield %8 : f32
      } -> tensor<1x256xf32>
      flow.dispatch.tensor.store %7, %2, offsets = [%workgroup_id_y, %3], sizes = [1, 256], strides = [1, 1] : tensor<1x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<233x1024xf32>>
      return
    }
  }
}
}

//         CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0 * 4)>
//   CHECK-LABEL: func.func @add_tensor
//     CHECK-DAG:   %[[A:.*]] = hal.interface.binding.subspan set(0) binding(0)
//     CHECK-DAG:   %[[B:.*]] = hal.interface.binding.subspan set(0) binding(1)
//     CHECK-DAG:   %[[C:.*]] = hal.interface.binding.subspan set(0) binding(2)
//     CHECK-DAG:   %[[LA:.*]] = flow.dispatch.tensor.load %[[A]]
//     CHECK-DAG:   %[[LB:.*]] = flow.dispatch.tensor.load %[[B]]
//     CHECK-DAG:   %[[LC:.*]] = flow.dispatch.tensor.load %[[C]]
//         CHECK:   %[[T:.*]] = scf.forall (%[[ARG:.*]]) in (64) shared_outs(%[[O:.*]] = %[[LC]]) -> (tensor<1x256xf32>) {
//         CHECK:     %[[OFF:.*]] = affine.apply #[[$MAP]](%[[ARG]])
//     CHECK-DAG:     %[[TA:.*]] = tensor.extract_slice %[[LA]][0, %[[OFF]]] [1, 4] [1, 1] : tensor<1x256xf32> to tensor<1x4xf32>
//     CHECK-DAG:     %[[TB:.*]] = tensor.extract_slice %[[LB]][0, %[[OFF]]] [1, 4] [1, 1] : tensor<1x256xf32> to tensor<1x4xf32>
//     CHECK-DAG:     %[[TC:.*]] = tensor.extract_slice %[[O]][0, %[[OFF]]] [1, 4] [1, 1] : tensor<1x256xf32> to tensor<1x4xf32>
//         CHECK:     %[[L:.*]] = linalg.generic {{.*}} ins(%[[TA]], %[[TB]] : tensor<1x4xf32>, tensor<1x4xf32>) outs(%[[TC]] : tensor<1x4xf32>)
//         CHECK:       %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
//         CHECK:       linalg.yield %{{.*}} : f32
//         CHECK:     } -> tensor<1x4xf32>
//         CHECK:     scf.forall.in_parallel {
//         CHECK:       tensor.parallel_insert_slice %[[L]] into %[[O]][0, %[[OFF]]] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<1x256xf32>
//         CHECK:     }
//         CHECK:   } {mapping = [#gpu.thread<x>]}
//         CHECK:   flow.dispatch.tensor.store %[[T]], %{{.*}}, offsets = [%{{.*}}, %{{.*}}], sizes = [1, 256], strides = [1, 1] : tensor<1x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<233x1024xf32>>

// -----

hal.executable private @reduction  {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>) {
  hal.executable.export public @reduction ordinal(0)
  layout(#hal.pipeline.layout<push_constants = 0,
         sets = [<0, bindings = [<0, storage_buffer>, <1, storage_buffer>, <2, storage_buffer>]>]>)
         attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>,
                     workgroup_size = [64 : index, 1 : index, 1 : index]} {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x384xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128xf32>>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %2 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
      %3 = flow.dispatch.tensor.load %1, offsets = [%2], sizes = [64], strides = [1] : !flow.dispatch.tensor<writeonly:tensor<128xf32>> -> tensor<64xf32>
      %4 = flow.dispatch.tensor.load %0, offsets = [%2, 0], sizes = [64, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x384xf32>> -> tensor<64x384xf32>
      %5 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 4]]>} ins(%cst : f32) outs(%3 : tensor<64xf32>) -> tensor<64xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%4 : tensor<64x384xf32>) outs(%5 : tensor<64xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 4]]>} {
      ^bb0(%arg0: f32, %arg1: f32):
        %7 = arith.addf %arg0, %arg1 : f32
        linalg.yield %7 : f32
      } -> tensor<64xf32>
      flow.dispatch.tensor.store %6, %1, offsets = [%2], sizes = [64], strides = [1] : tensor<64xf32> -> !flow.dispatch.tensor<writeonly:tensor<128xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @reduction
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//     CHECK-DAG:   %[[C384:.*]] = arith.constant 384 : index
//         First the scf.foreach for the linalg.fill.
//         CHECK:   scf.forall
//         then the reduction case.
//         CHECK:   %[[T:.*]] = scf.forall (%[[ARG:.*]]) in (64) shared_outs(%[[O:.+]] = %{{.+}}) -> (tensor<64xf32>) {
//         CHECK:     %[[OUTSLICE:.*]] = tensor.extract_slice %{{.*}}[%[[ARG]], 0] [1, 384] [1, 1] : tensor<64x384xf32> to tensor<1x384xf32>
//         CHECK:     %[[A:.*]] = tensor.extract_slice %[[O]][%[[ARG]]] [1] [1] : tensor<64xf32> to tensor<1xf32>
//         CHECK:     %[[R:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C384]] step %[[C4]] iter_args(%[[ACC:.*]] = %[[A]]) -> (tensor<1xf32>) {
//         CHECK:     %[[E:.*]] = tensor.extract_slice %[[OUTSLICE]][0, %[[IV]]] [1, 4] [1, 1] : tensor<1x384xf32> to tensor<1x4xf32>
//         CHECK:       %[[L:.*]] = linalg.generic {{.*}} ins(%[[E]] : tensor<1x4xf32>) outs(%[[ACC]] : tensor<1xf32>)
//         CHECK:         arith.addf
//         CHECK:         linalg.yield %{{.*}} : f32
//         CHECK:       } -> tensor<1xf32>
//         CHECK:       scf.yield %[[L]] : tensor<1xf32>
//         CHECK:     }
//         CHECK:     scf.forall.in_parallel {
//         CHECK:       tensor.parallel_insert_slice %[[R]] into %[[O]][%[[ARG]]] [1] [1] : tensor<1xf32> into tensor<64xf32>
//         CHECK:     }
//         CHECK:   } {mapping = [#gpu.thread<x>]}
//         CHECK:   flow.dispatch.tensor.store %[[T]], %{{.}}, offsets = [%{{.*}}], sizes = [64], strides = [1] : tensor<64xf32> -> !flow.dispatch.tensor<writeonly:tensor<128xf32>>

// -----

hal.executable private @reduction_broadcast  {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>) {
  hal.executable.export public @reduction_broadcast ordinal(0)
  layout(#hal.pipeline.layout<push_constants = 0,
         sets = [<0, bindings = [<0, storage_buffer>, <1, storage_buffer>, <2, storage_buffer>]>]>)
         attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorize>,
                     workgroup_size = [64 : index, 1 : index, 1 : index]} {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @reduction_broadcast() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x32x10x4096xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x32x10x4096xf32>>
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_id_y = hal.interface.workgroup.id[1] : index
      %2 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
      %3 = flow.dispatch.tensor.load %1, offsets = [%workgroup_id_y, %2, 0, 0], sizes = [1, 32, 10, 4096], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<2x32x10x4096xf32>> -> tensor<1x32x10x4096xf32>
      %4 = flow.dispatch.tensor.load %0, offsets = [%workgroup_id_y, %2, 0, 0], sizes = [1, 32, 10, 4096], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x32x10x4096xf32>> -> tensor<1x32x10x4096xf32>
      %5 = tensor.empty() : tensor<1x32xf32>
      %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 4, 4]]>} ins(%cst : f32) outs(%5 : tensor<1x32xf32>) -> tensor<1x32xf32>
      %7 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
        ins(%4 : tensor<1x32x10x4096xf32>) outs(%6 : tensor<1x32xf32>)
        attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 4, 4]]>} {
      ^bb0(%arg0: f32, %arg1: f32):
        %9 = arith.addf %arg0, %arg1 : f32
        linalg.yield %9 : f32
      } -> tensor<1x32xf32>
      %8 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%4, %7 : tensor<1x32x10x4096xf32>, tensor<1x32xf32>) outs(%3 : tensor<1x32x10x4096xf32>)
        attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 4, 4]]>} {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
        %9 = arith.addf %arg0, %arg1 : f32
        linalg.yield %9 : f32
      } -> tensor<1x32x10x4096xf32>
      flow.dispatch.tensor.store %8, %1, offsets = [%workgroup_id_y, %2, 0, 0], sizes = [1, 32, 10, 4096], strides = [1, 1, 1, 1] : tensor<1x32x10x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x10x4096xf32>>
      return
    }
}
}
}
//   Check that the parallel dimensions that didn't get distributed are being
//   tiled with a serial loop. This happens because the broadcast has extra
//   parallel dimension that won't get distributed by tile and distribute to
//   workgroup.
//   CHECK-LABEL: func.func @reduction_broadcast
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//     CHECK-DAG:   %[[C10:.*]] = arith.constant 10 : index
//     CHECK-DAG:   %[[C4096:.*]] = arith.constant 4096 : index
//         CHECK:   scf.forall
//         CHECK:     linalg.fill
//         CHECK:   scf.forall
//         CHECK:     scf.for %{{.*}} = %[[C0]] to %[[C10]] step %[[C4]]
//         CHECK:       scf.for %{{.*}} = %[[C0]] to %[[C4096]] step %[[C4]]
//         CHECK:         linalg.generic
//         CHECK:   scf.forall
//         CHECK:     linalg.generic
