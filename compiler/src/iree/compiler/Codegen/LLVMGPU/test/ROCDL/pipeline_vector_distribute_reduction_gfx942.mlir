// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution --iree-llvmgpu-enable-prefetch=true \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   %s | FileCheck %s

#config = #iree_gpu.lowering_config< {workgroup = [0, 4, 0],
                                      reduction = [0, 0, 128],
                                      thread = [0, 0, 8],
                                      subgroup_basis = [[1, 1, 1], [0, 1, 2]],
                                      thread_basis   = [[1, 4, 16], [0, 1, 2]]}
>
#translation = #iree_codegen.translation_info< pipeline = LLVMGPUVectorDistribute
                                               workgroup_size = [64, 1, 1]
                                               subgroup_size = 64, {}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @matvec_fp16 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matvec_fp16 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_fp16() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
        %5 = tensor.empty() : tensor<1x32000xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) attrs = { lowering_config = #config } {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<1x32000xf16>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        return
      }
    }
  }
}

//     CHECK-LABEL: func.func @matvec_fp16
//          CHECK:    scf.for {{.*}} = %c0 to %c4096 step %c128
//          CHECK:      %[[OUT:.+]] = vector.contract
//     CHECK-SAME:      vector<1x1x8xf16>, vector<1x1x1x1x1x8xf16> into vector<1x1x1xf16>
//          CHECK:      %[[SCALAR:.+]] = vector.extract %[[OUT]]
//          CHECK:      gpu.subgroup_reduce  add %[[SCALAR]]

//          CHECK:      scf.yield
//          CHECK:    vector.transfer_write

// -----

#config = #iree_gpu.lowering_config< {workgroup = [1, 4, 0],
                                      reduction = [0, 0, 512],
                                      thread = [0, 0, 8],
                                      subgroup_basis = [[1, 4, 1], [0, 1, 2]],
                                      thread_basis   = [[1, 1, 64], [0, 1, 2]]}
>
#translation = #iree_codegen.translation_info< pipeline = LLVMGPUVectorDistribute
                                               workgroup_size = [256, 1, 1]
                                               subgroup_size = 64, {}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @matvec_fp16_parallel_subgroup {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matvec_fp16_parallel_subgroup ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_fp16_parallel_subgroup() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
        %5 = tensor.empty() : tensor<1x32000xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) attrs = { lowering_config = #config } {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<1x32000xf16>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        return
      }
    }
  }
}

//     CHECK-LABEL: func.func @matvec_fp16_parallel_subgroup
//          CHECK:    scf.for {{.*}} = %c0 to %c4096 step %c512
//          CHECK:      %[[OUT:.+]] = vector.contract
//     CHECK-SAME:      vector<1x1x8xf16>, vector<1x1x1x1x1x8xf16> into vector<1x1x1xf16>
//          CHECK:      %[[SCALAR:.+]] = vector.extract %[[OUT]]
//          CHECK:      gpu.subgroup_reduce  add %[[SCALAR]]

//          CHECK:      scf.yield
//          CHECK:    vector.transfer_write

// -----

#config = #iree_gpu.lowering_config< {workgroup = [0, 4, 0],
                                      reduction = [0, 0, 512],
                                      thread = [0, 0, 8],
                                      subgroup_basis = [[1, 4, 1], [0, 1, 2]],
                                      thread_basis   = [[1, 1, 64], [0, 1, 2]],
                                      promote_operands = [1]}
>
#translation = #iree_codegen.translation_info< pipeline = LLVMGPUVectorDistribute
                                               workgroup_size = [256, 1, 1]
                                               subgroup_size = 64, {}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @matvec_fp16_promote_rhs {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matvec_fp16_promote_rhs ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_fp16_promote_rhs() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
        %5 = tensor.empty() : tensor<1x32000xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) attrs = { lowering_config = #config } {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<1x32000xf16>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        return
      }
    }
  }
}

//     CHECK-LABEL: func.func @matvec_fp16_promote_rhs
//          CHECK:    %[[ALLOC:.+]] = memref.alloc() : memref<4x516xf16, #gpu.address_space<workgroup>>
//          CHECK:    scf.for {{.*}} = %c0 to %c4096 step %c512
//          CHECK:      %[[RHS_SHARED_READ:.+]] = vector.transfer_read %alloc
//          CHECK:      %[[RHS_INSERT:.+]] = vector.insert_strided_slice %[[RHS_SHARED_READ]]
//          CHECK:      %[[OUT:.+]] = vector.contract
//     CHECK-SAME:      %{{.*}}, %[[RHS_INSERT]], %{{.*}} : vector<1x1x8xf16>, vector<1x1x1x1x1x8xf16> into vector<1x1x1xf16>
//          CHECK:      %[[SCALAR:.+]] = vector.extract %[[OUT]]
//          CHECK:      gpu.subgroup_reduce  add %[[SCALAR]]

//          CHECK:      scf.yield
//          CHECK:    vector.transfer_write

// -----

// Tile Sizes:
// workgroup = [1, 1,  0,   0, 32]
// reduction = [0, 0,  0, 128,  0]
// subgroup  = [0, 0,  0,   0,  8]
// thread    = [0, 0, 32,   4,  4]

// Counts:
// subgroup  = [1, 1,  1,   1,  4]
// threads   = [1, 1,  2,  32,  1]

#config = #iree_gpu.lowering_config<{workgroup = [1, 1, 0, 0, 32],
                                     reduction = [0, 0, 0, 128, 0],
                                     promote_operands = [1, 2]}>

#qk_config = #iree_gpu.lowering_config<{subgroup_basis = [[1, 1, 1, 1, 4], [0, 1, 2, 3]],
                                        thread_basis   = [[1, 1, 2, 32, 1], [0, 1, 2, 3]],
                                        thread         = [0, 0, 32, 4],
                                        promote_operands = [1]}>

#pv_config = #iree_gpu.lowering_config<{subgroup_basis = [[1, 1, 1, 1, 4], [0, 1, 3, 4]],
                                        thread_basis   = [[1, 1, 2, 32, 1], [0, 1, 3, 4]],
                                        thread         = [0, 0, 4, 4],
                                        promote_operands = [1]}>

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [256, 1, 1]
                                              subgroup_size = 64>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @attention_20x1x64x4096x64 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention_20x1x64x4096x64 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_20x1x64x4096x64() attributes {translation_info = #translation} {
        %cst = arith.constant 1.250000e-01 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x1x64xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x1x64xf16>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 1, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x1x64xf16>> -> tensor<20x1x64xf16>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %7 = tensor.empty() : tensor<20x1x64xf16>
        %8 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>],
                     lowering_config = #config,
                     decomposition_config = {
                      qk_attrs = {lowering_config = #qk_config},
                      pv_attrs = {lowering_config = #pv_config}
                     }}
                     ins(%4, %5, %6, %cst : tensor<20x1x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, f16) outs(%7 : tensor<20x1x64xf16>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score : f32
                     } -> tensor<20x1x64xf16>
        iree_tensor_ext.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [20, 1, 64], strides = [1, 1, 1] : tensor<20x1x64xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x1x64xf16>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @attention_20x1x64x4096x64
// CHECK:         scf.for %{{.*}} = %c0 to %c4096 step %c128
// QK Matmul
// CHECK:           vector.contract
// CHECK-SAME:      vector<1x1x32xf16>, vector<1x1x1x1x4x32xf16> into vector<1x1x4xf32>
// CHECK-COUNT-4:   gpu.subgroup_reduce  add

// QK Max
// CHECK-COUNT-1:   gpu.subgroup_reduce  maximumf

// PV Sum
// CHECK-COUNT-1:   gpu.subgroup_reduce  add

// PV Matmul
// CHECK:           vector.contract
// CHECK-COUNT-8:   gpu.subgroup_reduce  add

// CHECK:           scf.yield

// -----

// Tile Sizes:
// workgroup         = [1, 1,  0,   0, 32]
// partial_reduction = [0, 0,  0, 128,  0]
// subgroup          = [0, 0,  0,   0,  8]
// thread            = [0, 0, 32,   4,  4]

// Counts:
// subgroup          = [1, 1,  1,   1,  4]
// threads           = [1, 1,  2,  32,  1]

#config = #iree_gpu.lowering_config<{workgroup = [1, 1, 0, 0, 32],
                                     partial_reduction = [0, 0, 0, 128, 0],
                                     promote_operands = [1, 2]}>

#qk_config = #iree_gpu.lowering_config<{subgroup_basis = [[1, 1, 1, 1, 4], [0, 1, 2, 3]],
                                        thread_basis   = [[1, 1, 2, 32, 1], [0, 1, 2, 3]],
                                        thread         = [0, 0, 32, 4],
                                        promote_operands = [1]}>

#pv_config = #iree_gpu.lowering_config<{subgroup_basis = [[1, 1, 1, 1, 4], [0, 1, 3, 4]],
                                        thread_basis   = [[1, 1, 2, 32, 1], [0, 1, 3, 4]],
                                        thread         = [0, 0, 4, 4],
                                        promote_operands = [1]}>

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
                                              workgroup_size = [256, 1, 1]
                                              subgroup_size = 64>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @attention_20x1x64x4096x64 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @attention_20x1x64x4096x64 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_20x1x64x4096x64() attributes {translation_info = #translation} {
        %cst = arith.constant 1.250000e-01 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x1x64xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x1x64xf16>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 1, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x1x64xf16>> -> tensor<20x1x64xf16>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %7 = tensor.empty() : tensor<20x1x64xf16>
        %8 = iree_linalg_ext.attention  {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>],
                     lowering_config = #config,
                     decomposition_config = {
                      qk_attrs = {lowering_config = #qk_config},
                      pv_attrs = {lowering_config = #pv_config}
                     }}
                     ins(%4, %5, %6, %cst : tensor<20x1x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, f16) outs(%7 : tensor<20x1x64xf16>) {
                      ^bb0(%score: f32):
                        iree_linalg_ext.yield %score : f32
                     } -> tensor<20x1x64xf16>
        iree_tensor_ext.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [20, 1, 64], strides = [1, 1, 1] : tensor<20x1x64xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<20x1x64xf16>>
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @attention_20x1x64x4096x64
// CHECK:         scf.for %{{.*}} = %c0 to %c4096 step %c128
// QK Matmul
// CHECK:           vector.contract
// CHECK-SAME:      vector<1x1x32xf16>, vector<1x1x1x1x4x32xf16> into vector<1x1x4xf32>
// CHECK-COUNT-4:   gpu.subgroup_reduce  add

// No subgroup reduction in the loop other than QK reductions
// CHECK-NOT: gpu.subgroup_reduce

// CHECK:           scf.yield

// CHECK:           vector.multi_reduction <maximumf>
// CHECK-COUNT-1:   gpu.subgroup_reduce  maximumf

// PV Matmul
// CHECK:           vector.contract
// CHECK-SAME:      vector<1x1x4xf32>, vector<1x1x4xf32> into f32
// CHECK-COUNT-1:   gpu.subgroup_reduce  add

// CHECK:           vector.multi_reduction <add>
// CHECK-COUNT-8:   gpu.subgroup_reduce  add


// -----

#config = #iree_gpu.lowering_config< {workgroup = [0, 4, 0],
                                      reduction = [0, 0, 128],
                                      thread = [0, 0, 4],
                                      subgroup_basis = [[1, 1, 2], [0, 1, 2]],
                                      thread_basis   = [[1, 4, 16], [0, 1, 2]]}
>
#translation = #iree_codegen.translation_info< pipeline = LLVMGPUVectorDistribute
                                               workgroup_size = [64, 1, 1]
                                               subgroup_size = 64, {}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @matvec_fp16 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matvec_fp16 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_fp16_subgroup_reduction() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
        %5 = tensor.empty() : tensor<1x32000xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) attrs = { lowering_config = #config } {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<1x32000xf16>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        return
      }
    }
  }
}
//     CHECK-LABEL: func.func @matvec_fp16_subgroup_reduction
//          CHECK:    scf.for {{.*}} = %c0 to %c4096 step %c128
//          CHECK:      %[[OUT:.+]] = vector.contract
//     CHECK-SAME:      vector<1x1x4xf16>, vector<1x1x1x1x1x4xf16> into vector<1x1x1xf16>
//          CHECK:      %[[SCALAR:.+]] = vector.extract %[[OUT]]
//          CHECK:      gpu.subgroup_reduce  add %[[SCALAR]]
//          CHECK:        gpu.barrier
                       /// Second round of reduction i.e., across subgroups.
//          CHECK:      gpu.subgroup_reduce add {{.*}} cluster(size = 2)
//          CHECK:      scf.yield
//          CHECK:    vector.transfer_write

// -----

#config = #iree_gpu.lowering_config< {workgroup = [0, 4, 0],
                                      partial_reduction = [0, 0, 128],
                                      thread = [0, 0, 8],
                                      subgroup_basis = [[1, 1, 1], [0, 1, 2]],
                                      thread_basis   = [[1, 4, 16], [0, 1, 2]]}
>
#translation = #iree_codegen.translation_info< pipeline = LLVMGPUVectorDistribute
                                               workgroup_size = [64, 1, 1]
                                               subgroup_size = 64, {}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @matvec_fp16_unaligned {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @matvec_fp16_unaligned ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_fp16_unaligned() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4099xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4099xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4099], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4099xf32>> -> tensor<1x4099xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4099], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32000x4099xf16>> -> tensor<32000x4099xf16>
        %tempty = tensor.empty() : tensor<1x4099xf16>
        %trunc = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"] } ins(%3 : tensor<1x4099xf32>) outs(%tempty : tensor<1x4099xf16>) {
        ^bb0(%in : f32, %out : f16):
          %trunced = arith.truncf %in : f32 to f16
          linalg.yield %trunced : f16
        } -> tensor<1x4099xf16>
        %5 = tensor.empty() : tensor<1x32000xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%trunc, %4 : tensor<1x4099xf16>, tensor<32000x4099xf16>) outs(%6 : tensor<1x32000xf16>) attrs = { lowering_config = #config } {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<1x32000xf16>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        return
      }
    }
  }
}

// Test that we don't emit spurious roundtrips to (shared) memory to perform masked reads for unaligned cases.
//
//   MEMORY-LABEL: func.func @matvec_fp16
//    CHECK-LABEL: func.func @matvec_fp16
//      CHECK-NOT:   vector.transfer_write
//          CHECK:   gpu.subgroup_reduce
//          CHECK:   vector.transfer_write
