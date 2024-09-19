// RUN: iree-opt --iree-hal-post-configuration-transformation-pipeline --verify-diagnostics %s -o -

// The final bitcode validation should error out on any external functions that
// remain in the final bitcode (post device bitcode linking).

module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #hal.device.target<"hip", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>, ukernels = "none"}>]> : !hal.device
  // expected-error @+1 {{failed to serialize executables}}
  hal.executable private @test {
    // expected-error @+2 {{found an unresolved external function 'external_func' in the final bitcode}}
    // expected-error @+1 {{failed to serialize executable for target backend rocm}}
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>, ukernels = "none"}>) {
      hal.executable.export public @test ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) {
      ^bb0(%arg0: !hal.device):
        %c128 = arith.constant 128 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        hal.return %c128, %c2, %c1 : index, index, index
      }
      builtin.module {
        func.func private @external_func(vector<1xf16>) -> vector<1xf16>

        func.func @test() attributes {translation_info = #iree_codegen.translation_info<None workgroup_size = [128, 2, 1] subgroup_size = 64>} {
          %c0 = arith.constant 0 : index
          %thread_id_x = gpu.thread_id  x
          %thread_id_y = gpu.thread_id  y
          %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>
          %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>
          %2 = vector.load %0[%thread_id_x, %thread_id_y] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<1xf16>
          %3 = func.call @external_func(%2) : (vector<1xf16>) -> vector<1xf16>
          vector.store %3, %1[%thread_id_x, %thread_id_y] : memref<4096x4096xf16, strided<[4096, 1], offset: ?>>, vector<1xf16>
          return
        }
      }
    }
  }
  util.func public @isolated_benchmark(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @isolated_benchmark(%input0: tensor<4096x4096xf16>, %input1: tensor<4096x4096xf16>) -> (%output0: tensor<4096x4096xf16>)"}} {
    %c33554432 = arith.constant 33554432 : index
    %c0 = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    %element_type_f16 = hal.element_type<f16> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c4096, %c4096]) type(%element_type_f16) encoding(%dense_row_major)
    %0 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<4096x4096xf16> in !stream.resource<external>{%c33554432}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%c4096, %c4096]) type(%element_type_f16) encoding(%dense_row_major)
    %1 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<4096x4096xf16> in !stream.resource<external>{%c33554432}
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%c33554432} => !stream.timepoint
    %2 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c33554432}, %1 as %arg3: !stream.resource<external>{%c33554432}, %result as %arg4: !stream.resource<external>{%c33554432}) {
      stream.cmd.dispatch @test::@rocm_hsaco_fb::@test {
        ro %arg2[%c0 for %c33554432] : !stream.resource<external>{%c33554432},
        ro %arg3[%c0 for %c33554432] : !stream.resource<external>{%c33554432},
        wo %arg4[%c0 for %c33554432] : !stream.resource<external>{%c33554432}
      }
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %result : !stream.resource<external>{%c33554432}
    %4 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %3 : tensor<4096x4096xf16> in !stream.resource<external>{%c33554432} -> !hal.buffer_view
    util.return %4 : !hal.buffer_view
  }
}
