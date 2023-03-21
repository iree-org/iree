// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-tile-and-distribute-to-workgroups{max-workgroup-parallel-dims=1})), canonicalize, cse)' --split-input-file %s | FileCheck %s -check-prefix=CHECKW

module attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_86"}>], legacy_sync}>]} {
  hal.executable private @collapse_workgroups_dispatch_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_86"}> {
      hal.executable.export public @collapse_workgroups_dispatch_dispatch_0_generic_1024x128x16x64 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @collapse_workgroups_dispatch_dispatch_0_generic_1024x128x16x64() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x16x128x64xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x128x16x64xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1024, 16, 128, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x16x128x64xf32>> -> tensor<1024x16x128x64xf32>
          %3 = tensor.empty() : tensor<1024x128x16x64xf32>
          %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1024x16x128x64xf32>) outs(%3 : tensor<1024x128x16x64xf32>) attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1, 64]]>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<1024x128x16x64xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [1024, 128, 16, 64], strides = [1, 1, 1, 1] : tensor<1024x128x16x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x128x16x64xf32>>
          return
        }
      }
    }
  }
}

// CHECKW-LABEL:   hal.executable private @collapse_workgroups_dispatch_dispatch_0 {
// CHECKW:           hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
// CHECKW:             hal.executable.export public @collapse_workgroups_dispatch_dispatch_0_generic_1024x128x16x64 ordinal(0) layout(#pipeline_layout) {
// CHECKW:             ^bb0(%[[ARG0:.*]]: !hal.device, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index):
// CHECKW:               %[[C2097152:.*]] = arith.constant 2097152 : index
// CHECKW:               %[[C1:.*]] = arith.constant 1 : index
// CHECKW:               hal.return %[[C2097152]], %[[C1]], %[[C1]] : index, index, index
// CHECKW:             }
