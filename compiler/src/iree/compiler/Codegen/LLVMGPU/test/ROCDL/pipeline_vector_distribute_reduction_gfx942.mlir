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
    hal.executable.export public @matvec_fp16 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_fp16() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
        %5 = tensor.empty() : tensor<1x32000xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) attrs = { lowering_config = #config } {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<1x32000xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
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
    hal.executable.export public @matvec_fp16_parallel_subgroup ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_fp16_parallel_subgroup() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
        %5 = tensor.empty() : tensor<1x32000xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) attrs = { lowering_config = #config } {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<1x32000xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
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
    hal.executable.export public @matvec_fp16_promote_rhs ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_fp16_promote_rhs() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x4096xf16>> -> tensor<1x4096xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32000, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32000x4096xf16>> -> tensor<32000x4096xf16>
        %5 = tensor.empty() : tensor<1x32000xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<1x32000xf16>) -> tensor<1x32000xf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<1x4096xf16>, tensor<32000x4096xf16>) outs(%6 : tensor<1x32000xf16>) attrs = { lowering_config = #config } {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %8 = arith.mulf %in, %in_0 : f16
          %9 = arith.addf %out, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<1x32000xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 32000], strides = [1, 1] : tensor<1x32000xf16> -> !flow.dispatch.tensor<writeonly:tensor<1x32000xf16>>
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
