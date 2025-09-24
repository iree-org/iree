// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN:   --iree-codegen-llvmgpu-use-vector-distribution --iree-llvmgpu-enable-prefetch=true \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target)))))" \
// RUN:   %s | FileCheck %s

#config = #iree_gpu.lowering_config< {workgroup = [0, 4, 0],
                                      reduction = [0, 0, 128],
                                      thread = [0, 0, 8],
                                      subgroup_basis = [[1, 1, 1], [0, 1, 2]],
                                      lane_basis = [[1, 4, 16], [0, 1, 2]]}
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
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
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
                                      lane_basis = [[1, 1, 64], [0, 1, 2]]}
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
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
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
                                      lane_basis = [[1, 1, 64], [0, 1, 2]],
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
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
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
                                        lane_basis = [[1, 1, 2, 32, 1], [0, 1, 2, 3]],
                                        thread         = [0, 0, 32, 4],
                                        promote_operands = [1]}>

#pv_config = #iree_gpu.lowering_config<{subgroup_basis = [[1, 1, 1, 1, 4], [0, 1, 3, 4]],
                                        lane_basis = [[1, 1, 2, 32, 1], [0, 1, 3, 4]],
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
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
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
                                        lane_basis = [[1, 1, 2, 32, 1], [0, 1, 2, 3]],
                                        thread         = [0, 0, 32, 4],
                                        promote_operands = [1]}>

#pv_config = #iree_gpu.lowering_config<{subgroup_basis = [[1, 1, 1, 1, 4], [0, 1, 3, 4]],
                                        lane_basis = [[1, 1, 2, 32, 1], [0, 1, 3, 4]],
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
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
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

// Sum
// CHECK:           vector.contract
// CHECK-SAME:      vector<1x1x4xf32>, vector<1x1x4xf32> into f32
// CHECK-COUNT-1:   gpu.subgroup_reduce  add

// PV Matmul
// CHECK:           vector.contract
// CHECK-SAME:      vector<1x1x4xf32>, vector<1x2x1x1x4x4xf32> into vector<2x1x4xf32>
// CHECK-COUNT-8:   gpu.subgroup_reduce  add


// -----

#config = #iree_gpu.lowering_config< {workgroup = [0, 4, 0],
                                      reduction = [0, 0, 128],
                                      thread = [0, 0, 4],
                                      subgroup_basis = [[1, 1, 2], [0, 1, 2]],
                                      lane_basis = [[1, 4, 16], [0, 1, 2]]}
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
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
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
                                      lane_basis = [[1, 4, 16], [0, 1, 2]]}
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
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
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

// -----

/// Paged attention reduction distribution to multiple subgroups.
/// Distribute 8x32 reduction dims across 4 subbroups with 2x32 threads shape per subgroup.
#translation = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64>
#pv_attrs_config = #iree_gpu.lowering_config<{subgroup_basis = [[1, 1, 1, 1, 1, 4, 1], [4, 3, 2, 1, 5, 6]], thread = [0, 0, 0, 8, 0, 0], lane_basis = [[1, 1, 1, 1, 1, 2, 32], [2, 1, 0, 4, 5, 6]]}>
#qk_attrs_config = #iree_gpu.lowering_config<{subgroup_basis = [[1, 1, 1, 1, 1, 4, 1], [4, 3, 2, 1, 5, 6]], thread = [0, 0, 0, 8, 0, 0], lane_basis = [[1, 1, 1, 1, 1, 2, 32], [2, 1, 0, 4, 5, 6]]}>
#attention_lowering_config = #iree_gpu.lowering_config<{partial_reduction = [0, 0, 0, 0, 0, 8, 0], workgroup = [1, 1, 1, 0, 0, 0, 0]}>

hal.executable private @attention_4xDx1x32x128xf16 {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
  // hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {abi = "hip", iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">, ukernels = "none"}>) {
    hal.executable.export public @attention_4xDx1x32x128xf16 ordinal(0) layout(#hal.pipeline.layout<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg1)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_4xDx1x32x128xf16() attributes {translation_info = #translation} {
        %cst = arith.constant 8.837890e-02 : f16
        %cst_0 = arith.constant 0.000000e+00 : f16
        %cst_1 = arith.constant 0xFC00 : f16
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
        %1 = arith.index_castui %0 : i32 to index
        %2 = util.assume.int %1<umin = 1, umax = 512> : index
        %3 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x1x2x1x32x128xf16>>
        %4 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1x1x128xf16>>
        %5 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(4) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x1x1x128xf16>>
        %6 = iree_tensor_ext.dispatch.workload.ordinal %2, 0 : index
        %7 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x?xi64>>{%6}
        %8 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x?xi64>>{%6}
        %9 = iree_tensor_ext.dispatch.tensor.load %7, offsets = [0, 0], sizes = [4, %6], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x?xi64>>{%6} -> tensor<4x?xi64>
        %10 = iree_tensor_ext.dispatch.tensor.load %8, offsets = [0, 0], sizes = [4, %6], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x?xi64>>{%6} -> tensor<4x?xi64>
        %11 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [4, 1, 1, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1x1x128xf16>> -> tensor<4x1x1x128xf16>
        %12 = tensor.empty() : tensor<4x1x1x128xf16>
        %13 = tensor.empty(%6) : tensor<4x1x1x?x32xf16>
        %14 = tensor.empty(%6) : tensor<4x?x1x32x128xf16>
        %15 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0, 0, 0, 0], sizes = [4096, 1, 1, 1, 32, 128], strides = [1, 1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x1x2x1x32x128xf16>> -> tensor<4096x1x32x128xf16>
        %16 = iree_linalg_ext.gather dimension_map = [0] ins(%15, %9 : tensor<4096x1x32x128xf16>, tensor<4x?xi64>) outs(%14 : tensor<4x?x1x32x128xf16>) -> tensor<4x?x1x32x128xf16>
        %17 = iree_linalg_ext.gather dimension_map = [0] ins(%15, %10 : tensor<4096x1x32x128xf16>, tensor<4x?xi64>) outs(%14 : tensor<4x?x1x32x128xf16>) -> tensor<4x?x1x32x128xf16>
        %18 = linalg.generic {
            indexing_maps = [
              affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
            ],
            iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
          }
          outs(%13 : tensor<4x1x1x?x32xf16>) {
            ^bb0(%out: f16):
              %20 = linalg.index 4 : index
              %21 = linalg.index 3 : index
              %22 = arith.muli %21, %c32 overflow<nsw> : index
              %23 = arith.addi %20, %22 : index
              %24 = arith.cmpi sle, %23, %c0 : index
              %25 = arith.select %24, %cst_0, %cst_1 : f16
            linalg.yield %25 : f16
        } -> tensor<4x1x1x?x32xf16>
        %19 = iree_linalg_ext.attention {
            decomposition_config = {
              pv_attrs = {lowering_config = #pv_attrs_config},
              qk_attrs = {lowering_config = #qk_attrs_config}
            },
            indexing_maps = [
              affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4)>,
              affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d1, d6, d4)>,
              affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d1, d6, d3)>,
              affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>,
              affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d5, d6)>,
              affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>],
            lowering_config = #attention_lowering_config
          }
           ins(%11, %16, %17, %cst, %18 : tensor<4x1x1x128xf16>, tensor<4x?x1x32x128xf16>, tensor<4x?x1x32x128xf16>, f16, tensor<4x1x1x?x32xf16>)
          outs(%12 : tensor<4x1x1x128xf16>) {
            ^bb0(%arg0: f32):
              iree_linalg_ext.yield %arg0 : f32
        } -> tensor<4x1x1x128xf16>
        iree_tensor_ext.dispatch.tensor.store %19, %5, offsets = [0, 0, 0, 0], sizes = [4, 1, 1, 128], strides = [1, 1, 1, 1] : tensor<4x1x1x128xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x1x1x128xf16>>
        return
      }
    }
  }
}

//     CHECK-LABEL: func.func @attention_4xDx1x32x128xf16
//           CHECK:   scf.forall ({{.*}}) in (4)
//           CHECK:     scf.for {{.*}} -> (vector<1x1x1x1x1x1xf32>, vector<1x1x1x1x1x1xf32>, vector<1x1x16x1x1x1x1x1x8xf32>) {
//       CHECK-NOT:       gpu.subgroup_reduce
//           CHECK:       scf.yield
//
// Warning: the above layout_config for vector distribution on attention goes overboard on the tail part.
// CHECK-COUNT-390:     gpu.subgroup_reduce {{.*}} : (f32) -> f32
