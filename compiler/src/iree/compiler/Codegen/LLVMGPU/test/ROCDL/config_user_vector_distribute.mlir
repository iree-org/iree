// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --iree-codegen-reorder-workgroups-strategy=transpose \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target)))))" %s | FileCheck %s --check-prefix=OPT-OUT

// Check that applying `ReorderWorkgroups*` enables or disables workgroup reordering.

// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-codegen-llvmgpu-use-vector-distribution \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target)))))" %s | FileCheck %s --check-prefix=OPT-IN

// Check that applying the `no_reduce_shared_memory_bank_conflicts` pipeline option attribute disables shared memory padding.

// OPT-OUT:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64
// OPT-OUT-SAME:    gpu_pipeline_options = #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = true>

// OPT-IN:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64
// OPT-IN-SAME:    gpu_pipeline_options = #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = true>
#config = #iree_gpu.lowering_config<{workgroup = [128, 128, 0], reduction = [0, 0, 32], promote_operands = [0, 1],
                                    mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                    subgroup_m_count = 2, subgroup_n_count = 2}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @main_0_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @main_0_dispatch_0_matmul_transpose_b_2048x10240x1280_f16xf16xf32 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // OPT-OUT-LABEL: func.func @main_0_dispatch_0_matmul_transpose_b
      // OPT-OUT:         memref.alloc() : memref<128x32xf16, #gpu.address_space<workgroup>>
      // OPT-OUT:         memref.alloc() : memref<128x32xf16, #gpu.address_space<workgroup>>
      // OPT-OUT:         scf.forall
      // OPT-OUT:          scf.for
      // OPT-OUT:         } {mapping = [#iree_codegen.workgroup_mapping<x>, #iree_codegen.workgroup_mapping<y>]}

      // OPT-IN-LABEL: func.func @main_0_dispatch_0_matmul_transpose_b
      // OPT-IN:         memref.alloc() : memref<128x32xf16, #gpu.address_space<workgroup>>
      // OPT-IN:         memref.alloc() : memref<128x32xf16, #gpu.address_space<workgroup>>
      // OPT-IN:         scf.forall
      // OPT-IN:          scf.for
      // OPT-IN:         } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}

      func.func @main_0_dispatch_0_matmul_transpose_b_2048x10240x1280_f16xf16xf32()
        attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64, {
          gpu_pipeline_options = #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = true>  // Disable the 'reduceSharedMemoryBankConflicts' pass.
        }>} {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>> -> tensor<10240x1280xf16>
        %5 = tensor.empty() : tensor<2048x10240xf32>
        %6 = linalg.fill  ins(%cst : f16) outs(%5 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                              affine_map<(d0, d1, d2) -> (d1, d2)>,
                                              affine_map<(d0, d1, d2) -> (d0, d1)>],
                             iterator_types = ["parallel", "parallel", "reduction"],
                             lowering_config = #config}
          ins(%3, %4 : tensor<2048x1280xf16>, tensor<10240x1280xf16>)
          outs(%6 : tensor<2048x10240xf32>) {
        ^bb0(%in: f16, %in_0: f16, %out: f32):
          %8 = arith.extf %in : f16 to f32
          %9 = arith.extf %in_0 : f16 to f32
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %out, %10 : f32
          linalg.yield %11 : f32
        } -> tensor<2048x10240xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        return
      }
    }
  }
}

// -----

// Check that applying the `reorder_workgroups_strategy = <Transpose>` pipeline option attribute enables workgroup reordering.

// OPT-OUT:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64
// OPT-OUT-SAME:    gpu_pipeline_options = #iree_gpu.pipeline_options<reorder_workgroups_strategy = <Transpose>>

// OPT-IN:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64
// OPT-IN-SAME:    gpu_pipeline_options = #iree_gpu.pipeline_options<reorder_workgroups_strategy = <Transpose>>
#config = #iree_gpu.lowering_config<{workgroup = [128, 128, 0], reduction = [0, 0, 32], promote_operands = [0, 1],
                                    mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                    subgroup_m_count = 2, subgroup_n_count = 2}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @main_0_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @main_0_dispatch_0_matmul_transpose_b_2048x10240x1280_f16xf16xf32 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // OPT-OUT-LABEL: func.func @main_0_dispatch_0_matmul_transpose_b
      // OPT-OUT:         memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
      // OPT-OUT:         memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
      // OPT-OUT:         scf.forall
      // OPT-OUT:          scf.for
      // OPT-OUT:         } {mapping = [#iree_codegen.workgroup_mapping<x>, #iree_codegen.workgroup_mapping<y>]}

      // OPT-IN-LABEL: func.func @main_0_dispatch_0_matmul_transpose_b
      // OPT-IN:         memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
      // OPT-IN:         memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
      // OPT-IN:         scf.forall
      // OPT-IN:          scf.for
      // OPT-IN:         } {mapping = [#iree_codegen.workgroup_mapping<x>, #iree_codegen.workgroup_mapping<y>]}
      func.func @main_0_dispatch_0_matmul_transpose_b_2048x10240x1280_f16xf16xf32()
        attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64, {
          gpu_pipeline_options = #iree_gpu.pipeline_options<reorder_workgroups_strategy = <Transpose>>  // enable the 'reorderWorkgroups' pass.
        }>} {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>> -> tensor<10240x1280xf16>
        %5 = tensor.empty() : tensor<2048x10240xf32>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                              affine_map<(d0, d1, d2) -> (d1, d2)>,
                                              affine_map<(d0, d1, d2) -> (d0, d1)>],
                             iterator_types = ["parallel", "parallel", "reduction"],
                             lowering_config = #config}
          ins(%3, %4 : tensor<2048x1280xf16>, tensor<10240x1280xf16>)
          outs(%6 : tensor<2048x10240xf32>) {
        ^bb0(%in: f16, %in_0: f16, %out: f32):
          %8 = arith.extf %in : f16 to f32
          %9 = arith.extf %in_0 : f16 to f32
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %out, %10 : f32
          linalg.yield %11 : f32
        } -> tensor<2048x10240xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        return
      }
    }
  }
}

// -----
// Check that applying the `reorder_workgroups_strategy = <None>` pipeline option disables workgroup reordering.

// OPT-OUT:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64
// OPT-OUT-SAME:    gpu_pipeline_options = #iree_gpu.pipeline_options<reorder_workgroups_strategy = <None>>
#config = #iree_gpu.lowering_config<{workgroup = [128, 128, 0], reduction = [0, 0, 32], promote_operands = [0, 1],
                                    mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                                    subgroup_m_count = 2, subgroup_n_count = 2}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable public @main_0_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @main_0_dispatch_0_matmul_transpose_b_2048x10240x1280_f16xf16xf32 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // OPT-OUT-LABEL: func.func @main_0_dispatch_0_matmul_transpose_b
      // OPT-OUT:         memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
      // OPT-OUT:         memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
      // OPT-OUT:         scf.forall
      // OPT-OUT:          scf.for
      // OPT-OUT:         } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
      func.func @main_0_dispatch_0_matmul_transpose_b_2048x10240x1280_f16xf16xf32()
        attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64, {
          gpu_pipeline_options = #iree_gpu.pipeline_options<reorder_workgroups_strategy = <None>>  // Disable the 'reorderWorkgroups' pass.
        }>} {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1280xf16>> -> tensor<2048x1280xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [10240, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<10240x1280xf16>> -> tensor<10240x1280xf16>
        %5 = tensor.empty() : tensor<2048x10240xf32>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x10240xf32>) -> tensor<2048x10240xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                              affine_map<(d0, d1, d2) -> (d1, d2)>,
                                              affine_map<(d0, d1, d2) -> (d0, d1)>],
                             iterator_types = ["parallel", "parallel", "reduction"],
                             lowering_config = #config}
          ins(%3, %4 : tensor<2048x1280xf16>, tensor<10240x1280xf16>)
          outs(%6 : tensor<2048x10240xf32>) {
        ^bb0(%in: f16, %in_0: f16, %out: f32):
          %8 = arith.extf %in : f16 to f32
          %9 = arith.extf %in_0 : f16 to f32
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %out, %10 : f32
          linalg.yield %11 : f32
        } -> tensor<2048x10240xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1] : tensor<2048x10240xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
        return
      }
    }
  }
}
