// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-strip-config-info)))))' %s | FileCheck %s

#config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 64], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [64, 128, 0]}>
#translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>
util.global private @__device_0 = #hal.device.target<"hip", {legacy_sync}, [#hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none", waves_per_eu = 2 : i64}>]> : !hal.device
hal.executable private @main$async_dispatch_41 {
hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none", waves_per_eu = 2 : i64}>) {
    hal.executable.export public @main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32 ordinal(0) layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) {
    ^bb0(%arg0: !hal.device):
    %x, %y, %z = flow.dispatch.workgroup_count_from_slice
    hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
    func.func @matmul_transpose_b_8192x640x640_f16xf16xf32() attributes {translation_info = #translation_info} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4:2 = util.assume.int
            %2[<umin = 80010752, umax = 80010752, udiv = 80010752>, <umin = 80010752, umax = 80010752, udiv = 80010752>, <umin = 80010752, umax = 80010752, udiv = 80010752>, <umin = 80010752, umax = 80010752, udiv = 80010752>, <umin = 74767872, umax = 74767872, udiv = 74767872>],
            %3[<umin = 90496512, umax = 90496512, udiv = 90496512>, <umin = 90496512, umax = 90496512, udiv = 90496512>, <umin = 90496512, umax = 90496512, udiv = 90496512>, <umin = 90496512, umax = 90496512, udiv = 90496512>, <umin = 85253632, umax = 85253632, udiv = 85253632>]
        : index, index
        %5 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%4#0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>>
        %6 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640x640xf16>>
        %7 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<640xf16>>
        %8 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%4#1) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<8192x640xf16>>
        %9 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8192x640xf16>> -> tensor<8192x640xf16>
        %10 = flow.dispatch.tensor.load %6, offsets = [0, 0], sizes = [640, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<640x640xf16>> -> tensor<640x640xf16>
        %11 = flow.dispatch.tensor.load %7, offsets = [0], sizes = [640], strides = [1] : !flow.dispatch.tensor<readonly:tensor<640xf16>> -> tensor<640xf16>
        %12 = tensor.empty() : tensor<8192x640xf16>
        %13 = tensor.empty() : tensor<8192x640xf32>
        %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<8192x640xf32>) -> tensor<8192x640xf32>
        %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%9, %10 : tensor<8192x640xf16>, tensor<640x640xf16>) outs(%14 : tensor<8192x640xf32>) attrs =  {lowering_config = #config} {
        ^bb0(%in: f16, %in_0: f16, %out: f32):
        %17 = arith.extf %in : f16 to f32
        %18 = arith.extf %in_0 : f16 to f32
        %19 = arith.mulf %17, %18 : f32
        %20 = arith.addf %out, %19 : f32
        linalg.yield %20 : f32
        } -> tensor<8192x640xf32>
        %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%15, %11 : tensor<8192x640xf32>, tensor<640xf16>) outs(%12 : tensor<8192x640xf16>) {
        ^bb0(%in: f32, %in_0: f16, %out: f16):
        %17 = arith.truncf %in : f32 to f16
        %18 = arith.addf %17, %in_0 : f16
        linalg.yield %18 : f16
        } -> tensor<8192x640xf16>
        flow.dispatch.tensor.store %16, %8, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : tensor<8192x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<8192x640xf16>>
        return
    }
    }
}
}
util.global private mutable @main$async_dispatch_41_rocm_hsaco_fb_main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32_buffer : !hal.buffer
util.initializer {
%c807078144 = arith.constant 807078144 : index
%device, %queue_affinity = hal.device.resolve on(<@__device_0>) : !hal.device, i64
%allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
%buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%queue_affinity) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c807078144}
util.global.store %buffer, @main$async_dispatch_41_rocm_hsaco_fb_main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32_buffer : !hal.buffer
util.return
}
util.func public @main$async_dispatch_41_rocm_hsaco_fb_main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
%c-1_i32 = arith.constant -1 : i32
%0 = util.null : !hal.fence
%c1 = arith.constant 1 : index
%c403949312 = arith.constant 403949312 : index
%c1280 = arith.constant 1280 : index
%c403948032 = arith.constant 403948032 : index
%c819200 = arith.constant 819200 : index
%c403128832 = arith.constant 403128832 : index
%c0 = arith.constant 0 : index
%c85253632_i32 = arith.constant 85253632 : i32
%c74767872_i32 = arith.constant 74767872 : i32
%1 = arith.index_cast %arg0 : i32 to index
%device, %queue_affinity = hal.device.resolve on(<@__device_0>) : !hal.device, i64
%cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) affinity(%queue_affinity) : !hal.command_buffer
%main$async_dispatch_41_rocm_hsaco_fb_main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32_buffer = util.global.load @main$async_dispatch_41_rocm_hsaco_fb_main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32_buffer : !hal.buffer
%workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device : !hal.device) target(@main$async_dispatch_41::@rocm_hsaco_fb::@main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32) : index, index, index
%exe = hal.executable.lookup device(%device : !hal.device) executable(@main$async_dispatch_41) : !hal.executable
%ordinal = hal.executable.export.ordinal target(@main$async_dispatch_41::@rocm_hsaco_fb::@main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32) : index
scf.for %arg1 = %c0 to %1 step %c1 {
    hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z]) constants([%c74767872_i32, %c85253632_i32]) bindings([
    (%main$async_dispatch_41_rocm_hsaco_fb_main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32_buffer : !hal.buffer)[%c0, %c403128832],
    (%main$async_dispatch_41_rocm_hsaco_fb_main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32_buffer : !hal.buffer)[%c403128832, %c819200],
    (%main$async_dispatch_41_rocm_hsaco_fb_main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32_buffer : !hal.buffer)[%c403948032, %c1280],
    (%main$async_dispatch_41_rocm_hsaco_fb_main$async_dispatch_41_matmul_transpose_b_8192x640x640_f16xf16xf32_buffer : !hal.buffer)[%c403949312, %c403128832]
    ]) flags("None")
    hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
}
hal.command_buffer.finalize<%cmd : !hal.command_buffer>
%fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
hal.device.queue.execute<%device : !hal.device> affinity(%queue_affinity) wait(%0) signal(%fence) commands([%cmd])
%status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
util.status.check_ok %status, "failed to wait on timepoint"
util.return
}

// CHECK-LABEL: func.func @matmul_transpose_b
// CHECK-NOT:   #translation_info =
// CHECK-NOT:   LLVMGPUVectorDistribute
// CHECK-NOT:   #lowering_config =
