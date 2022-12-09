// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target{test-lowering-configuration})))" --iree-codegen-llvmgpu-enable-transform-dialect-jit %s | FileCheck %s

hal.executable @group_reduction {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_reduction ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x64xf32>> -> tensor<8x64xf32>
      %3 = tensor.empty() : tensor<8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<8x64xf32>) outs(%4 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<8xf32>
      flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: func.func @group_reduction
//       CHECK:   transform.structured.canonicalized_sequence failures(propagate)
//       CHECK:     transform.structured.match ops{["linalg.fill"]} in %{{.+}}
//       CHECK:     transform.structured.match ops{["linalg.generic"]} in %{{.+}}
//       CHECK:     transform.structured.split_reduction %{{.+}} {insert_split_dimension = 1 : i64, split_factor = 2 : i64}
//       CHECK:     transform.iree.tile_to_foreach_thread_and_workgroup_count_region %{{.*}}   num_threads [] tile_sizes [1](mapping = [#gpu.block<x>])
//       CHECK:     transform.structured.tile_to_foreach_thread_op %{{.*}}   num_threads [] tile_sizes [1, 0, 0](mapping = [#gpu.thread<z>])
//       CHECK:     transform.structured.tile_to_foreach_thread_op %{{.*}}   num_threads [] tile_sizes [1, 1, 0](mapping = [#gpu.thread<z>, #gpu.thread<y>])
//       CHECK:     transform.iree.bufferize {target_gpu}
//       CHECK:     transform.iree.foreach_thread_to_workgroup
//       CHECK:     transform.iree.map_nested_foreach_thread_to_gpu_threads %{{.+}} {workgroup_size = [32, 2, 1]}
//       CHECK:     transform.structured.match ops{["scf.if"]} in %{{.+}}
//       CHECK:       transform.iree.vector.to_warp_execute_on_lane_0 %{{.+}} {warp_size = 32 : i64}
//       CHECK:     transform.iree.vector.warp_distribute %{{.+}}
