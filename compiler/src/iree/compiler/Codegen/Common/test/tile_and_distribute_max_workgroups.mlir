// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-tile-and-distribute-to-workgroups{max-workgroup-parallel-dims=1})), canonicalize, cse)' --split-input-file %s | FileCheck %s

module attributes {hal.device.targets = [#hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_86"}>], legacy_sync}>]} {
  hal.executable private @collapse_workgroups_dispatch_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_86"}> {
      hal.executable.export public @collapse_workgroups_dispatch_dispatch_0_generic_1024x128x16x64 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @collapse_workgroups_dispatch_dispatch_0_generic_1024x128x16x64() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 1.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024x128x16x64xf32>>
          %1 = tensor.empty() : tensor<1024x128x16x64xf32>
          %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<1024x128x16x64xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<1024x128x16x64xf32>
          flow.dispatch.tensor.store %2, %0, offsets = [0, 0, 0, 0], sizes = [1024, 128, 16, 64], strides = [1, 1, 1, 1] : tensor<1024x128x16x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x128x16x64xf32>>
          return
        }
      }
    }
  }
  func.func @collapse_workgroups_dispatch() -> !hal.buffer_view attributes {iree.abi.stub} {
    %c0 = arith.constant 0 : index
    %c536870912 = arith.constant 536870912 : index
    %c1024 = arith.constant 1024 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %0 = stream.resource.alloc uninitialized : !stream.resource<external>{%c536870912}
    %1 = stream.cmd.execute with(%0 as %arg0: !stream.resource<external>{%c536870912}) {
      stream.cmd.dispatch @isolated_problem_dispatch_dispatch_0::@cuda_nvptx_fb::@isolated_problem_dispatch_dispatch_0_generic_1024x128x16x64[%c1024, %c128, %c16, %c64] {
        wo %arg0[%c0 for %c536870912] : !stream.resource<external>{%c536870912}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>]}
    } => !stream.timepoint
    %2 = stream.timepoint.await %1 => %0 : !stream.resource<external>{%c536870912}
    %3 = stream.tensor.export %2 : tensor<1024x128x16x64xf32> in !stream.resource<external>{%c536870912} -> !hal.buffer_view
    return %3 : !hal.buffer_view
  }
}

// CHECK-LABEL:   hal.executable private @collapse_workgroups_dispatch_dispatch_0 {
// CHECK:           hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
// CHECK:             hal.executable.export public @collapse_workgroups_dispatch_dispatch_0_generic_1024x128x16x64 ordinal(0) layout(#pipeline_layout) {
// CHECK:             ^bb0(%[[ARG0:.*]]: !hal.device, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index):
// CHECK:               %[[C1:.*]] = arith.constant 1 : index
// CHECK:               hal.return %[[C1]], %[[C1]], %[[C1]] : index, index, index
// CHECK:             }
// CHECK:             builtin.module {
// CHECK:               func.func @collapse_workgroups_dispatch_dispatch_0_generic_1024x128x16x64() {
// CHECK:                 %[[C0:.*]] = arith.constant 0 : index
// CHECK:                 %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:                 %[[D0:.*]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[C0]]) : !flow.dispatch.tensor<writeonly:tensor<1024x128x16x64xf32>>
// CHECK:                 %[[D1:.*]] = tensor.empty() : tensor<1024x128x16x64xf32>
// CHECK:                 %[[D2:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%[[D1]] : tensor<1024x128x16x64xf32>) {
// CHECK:                 ^bb0(%[[OUT:.*]]: f32):
// CHECK:                   linalg.yield %[[CST]] : f32
// CHECK:                 } -> tensor<1024x128x16x64xf32>
// CHECK:                 flow.dispatch.tensor.store %[[D2:.*]], %[[D0]], offsets = [0, 0, 0, 0], sizes = [1024, 128, 16, 64], strides = [1, 1, 1, 1] : tensor<1024x128x16x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x128x16x64xf32>>
// CHECK:                 return
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }
