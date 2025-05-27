// RUN: iree-compile --compile-from=stream --compile-to=hal %s | FileCheck %s

module attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
  util.global private @device_a = #hal.device.target<"local", [#hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_cpu.vmvx_encoding_layout<>, ukernels = "none"}>]> : !hal.device
  util.global private @device_b = #hal.device.target<"metal", [#hal.executable.target<"metal-spirv", "metal-msl-fb", {iree.gpu.target = #iree_gpu.target<arch = "apple", features = "spirv:v1.3,cap:Shader", wgp = <compute =  fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [], subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 32768, max_workgroup_counts = [65535, 65535, 65535]>>}>]> : !hal.device
  util.func public @multi_device_mul(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "async func @multi_device_mul(%input0: tensor<4xf32> {iree.abi.affinity = #hal.device.promise<@device_a>}) -> (%output0: tensor<4xf32> {iree.abi.affinity = #hal.device.promise<@device_a>})", iree.abi.model = "coarse-fences"}} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c4]) type(%element_type_f32) encoding(%dense_row_major)
    %0 = stream.tensor.import on(#hal.device.affinity<@device_a>) %arg0 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %1 = stream.timepoint.import on(#hal.device.affinity<@device_a>) %arg1 : (!hal.fence) => !stream.timepoint
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>) await(%1) => !stream.resource<external>{%c16} => !stream.timepoint
    stream.timepoint.chain_external on(#hal.device.affinity<@device_a>) %result_timepoint => (%arg2 : !hal.fence)
    %5 = stream.tensor.export on(#hal.device.affinity<@device_a>) %result : tensor<4xf32> in !stream.resource<external>{%c16} -> !hal.buffer_view
    util.return %5 : !hal.buffer_view
  }
}
// Check that for local interop we have a mapping persistent buffer flags.
// CHECK: util.func public @multi_device_mul(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view
// CHECK: hal.device.queue.alloca
// CHECK-SAME: MappingPersistent
