// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942  --iree-codegen-llvmgpu-use-vector-distribution --iree-llvmgpu-enable-prefetch=true --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))"

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>

#qk_config = #iree_gpu.lowering_config<
  {
    mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
    promote_operands = [1],
    subgroup_m_count = 1,
    subgroup_n_count = 1
  }
>

#pv_config = #iree_gpu.lowering_config<
  {
    workgroup = [1, 32, 0, 0],
    reduction = [0, 0, 0, 32],
    promote_operands = [1],
    mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
    subgroup_m_count = 1,
    subgroup_n_count = 1
  }
>

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>], flags = None>
#device_target_hip = #hal.device.target<"hip", [#executable_target_rocm_hsaco_fb]> : !hal.device
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #device_target_hip
  hal.executable public @main {
    hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @main ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @main() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64>} {
          %cst = arith.constant -3.40282347E+38 : f32
          %cst_0 = arith.constant 0.000000e+00 : f32
          %cst_1 = arith.constant 1.250000e-01 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) offset(%c0) flags("None") : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
          %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) offset(%c0) flags("None") : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
          %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) offset(%c0) flags("None") : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
          %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) offset(%c0) flags("None") : !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
          %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
          %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
          %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
          %7 = tensor.empty() : tensor<20x4096x4096xf32>
          %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<20x4096x4096xf32>) -> tensor<20x4096x4096xf32>
          %9 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"], lowering_config = #qk_config, attention_qk_matmul} ins(%4, %5 : tensor<20x4096x64xf16>, tensor<20x4096x64xf16>) outs(%8 : tensor<20x4096x4096xf32>) {
          ^bb0(%in: f16, %in_2: f16, %out: f32):
            %19 = arith.extf %in : f16 to f32
            %20 = arith.extf %in_2 : f16 to f32
            %21 = arith.mulf %19, %20 : f32
            %22 = arith.addf %21, %out : f32
            linalg.yield %22 : f32
          } -> tensor<20x4096x4096xf32>
          %10 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<20x4096x4096xf32>) outs(%7 : tensor<20x4096x4096xf32>) {
          ^bb0(%in: f32, %out: f32):
            %19 = arith.mulf %in, %cst_1 : f32
            linalg.yield %19 : f32
          } -> tensor<20x4096x4096xf32>
          %11 = tensor.empty() : tensor<20x4096xf32>
          %12 = tensor.empty() : tensor<20x4096x64xf32>
          %13 = linalg.fill ins(%cst : f32) outs(%11 : tensor<20x4096xf32>) -> tensor<20x4096xf32>
          %14 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<20x4096xf32>) -> tensor<20x4096xf32>
          %15 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<20x4096x64xf32>) -> tensor<20x4096x64xf32>
          %16:3 = iree_linalg_ext.exp_reduction{indexing_maps = [#map, #map4, #map5, #map5, #map2], iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>]} attributes {lowering_config = #pv_config, attention_pv_matmul} ins(%10, %6 : tensor<20x4096x4096xf32>, tensor<20x4096x64xf16>) outs(%13, %14, %15 : tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>) {
          ^bb0(%arg0: f32, %arg1: f16, %arg2: f32, %arg3: f32, %arg4: f32):
            %19 = arith.addf %arg0, %arg3 : f32
            %201 = arith.truncf %arg0 : f32 to f16
            %202 = arith.extf %201 : f16 to f32
            %20 = arith.extf %arg1 : f16 to f32
            %21 = arith.mulf %202, %20 : f32
            %22 = arith.addf %21, %arg4 : f32
            linalg.yield %arg2, %19, %22 : f32, f32, f32
          } -> tensor<20x4096xf32>, tensor<20x4096xf32>, tensor<20x4096x64xf32>
          %17 = tensor.empty() : tensor<20x4096x64xf16>
          %18 = linalg.generic {indexing_maps = [#map3, #map6, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16#2, %16#1 : tensor<20x4096x64xf32>, tensor<20x4096xf32>) outs(%17 : tensor<20x4096x64xf16>) {
          ^bb0(%in: f32, %in_2: f32, %out: f16):
            %19 = arith.divf %in, %in_2 : f32
            %20 = arith.truncf %19 : f32 to f16
            linalg.yield %20 : f16
          } -> tensor<20x4096x64xf16>
          flow.dispatch.tensor.store %18, %3, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : tensor<20x4096x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
          return
        }
      }
    }
  }
}
