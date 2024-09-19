// RUN: iree-opt --iree-hal-post-configuration-transformation-pipeline --verify-diagnostics %s -o -

// The final bitcode validation should error out on any external functions that
// remain in the final bitcode (post device bitcode linking).

module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #hal.device.target<"hip", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx1100", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<WMMA_F32_16x16x16_F16>, <WMMA_F16_16x16x16_F16>, <WMMA_I32_16x16x16_I8>], subgroup_size_choices = [32, 64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>, ukernels = "none"}>]> : !hal.device
  // expected-error @+1 {{failed to serialize executables}}
  hal.executable private @test_dispatch_0 {
    // expected-error @+2 {{found an unresolved external function 'external_func' in the final bitcode}}
    // expected-error @+1 {{failed to serialize executable for target backend rocm}}
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx1100", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<WMMA_F32_16x16x16_F16>, <WMMA_F16_16x16x16_F16>, <WMMA_I32_16x16x16_I8>], subgroup_size_choices = [32, 64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>, ukernels = "none"}>) {
      hal.executable.export public @test_dispatch_0_elementwise_D_i32 ordinal(0) layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) {
      ^bb0(%arg0: !hal.device, %arg1: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func private @external_func(i32) -> i32

        func.func @test_dispatch_0_elementwise_D_i32() attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorize workgroup_size = [64, 1, 1] subgroup_size = 32>} {
          %c0 = arith.constant 0 : index
          %c32_i64 = arith.constant 32 : i64
          %c1_i32 = arith.constant 1 : i32
          %0 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
          %1 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
          %2 = arith.extui %0 : i32 to i64
          %3 = arith.extui %1 : i32 to i64
          %4 = arith.shli %3, %c32_i64 : i64
          %5 = arith.ori %2, %4 : i64
          %6 = arith.index_castui %5 : i64 to index
          %7 = flow.dispatch.workload.ordinal %6, 0 : index
          %8 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<?xi32>>{%7}
          %9 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<?xi32>>{%7}
          %10 = flow.dispatch.tensor.load %8, offsets = [0], sizes = [%7], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xi32>>{%7} -> tensor<?xi32>
          %11 = tensor.empty(%7) : tensor<?xi32>
          %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%10 : tensor<?xi32>) outs(%11 : tensor<?xi32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64]]>} {
          ^bb0(%in: i32, %out: i32):
            %13 = func.call @external_func(%in) : (i32) -> i32
            linalg.yield %13 : i32
          } -> tensor<?xi32>
          flow.dispatch.tensor.store %12, %9, offsets = [0], sizes = [%7], strides = [1] : tensor<?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?xi32>>{%7}
          return
        }
      }
    }
  }
  util.func public @test(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @test(%input0: tensor<?xi32>) -> (%output0: tensor<?xi32>)"}} {
    %c32_i64 = arith.constant 32 : i64
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %element_type_i32 = hal.element_type<i32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%0]) type(%element_type_i32) encoding(%dense_row_major)
    %1 = arith.muli %0, %c4 : index
    %2 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<?xi32>{%0} in !stream.resource<external>{%1}
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%1} => !stream.timepoint
    %3 = arith.index_castui %0 : index to i64
    %4 = arith.trunci %3 : i64 to i32
    %5 = arith.shrui %3, %c32_i64 : i64
    %6 = arith.trunci %5 : i64 to i32
    %7 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => with(%2 as %arg1: !stream.resource<external>{%1}, %result as %arg2: !stream.resource<external>{%1}) {
      stream.cmd.dispatch @test_dispatch_0::@rocm_hsaco_fb::@test_dispatch_0_elementwise_D_i32[%0](%4, %6 : i32, i32) {
        ro %arg1[%c0 for %1] : !stream.resource<external>{%1},
        wo %arg2[%c0 for %1] : !stream.resource<external>{%1}
      }
    } => !stream.timepoint
    %8 = stream.timepoint.await %7 => %result : !stream.resource<external>{%1}
    %9 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %8 : tensor<?xi32>{%0} in !stream.resource<external>{%1} -> !hal.buffer_view
    util.return %9 : !hal.buffer_view
  }
}
