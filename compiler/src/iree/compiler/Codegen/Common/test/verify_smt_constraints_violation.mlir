// RUN: iree-opt --split-input-file \
// RUN:   --iree-codegen-experimental-verify-pipeline-constraints \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-verify-smt-constraints)))))' \
// RUN:   --verify-diagnostics %s

// Test: Static dim violation emits an error.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#gpu_target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target}>
#translation = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

hal.executable @verify_violation_ex {
  hal.executable.variant public @rocm target(#exec_target) {
    hal.executable.export public @verify_violation ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @verify_violation() attributes {translation_info = #translation} {
        %cst = arith.constant 0.0 : f32
        %lhs = tensor.empty() : tensor<128x64xf32>
        %rhs = tensor.empty() : tensor<64x256xf32>
        %empty = tensor.empty() : tensor<128x256xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
        %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
            ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
            outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c64 = arith.constant 64 : index
        // Assert dim_0 ({}) == 999 -- will fail because dim_0 is 128.
        // expected-error @below {{pipeline constraints violated}}
        iree_codegen.smt.constraints target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {}
            dims(%c128, %c256, %c64) {
        ^bb0(%m: !smt.int, %n: !smt.int, %k: !smt.int):
          %c999 = smt.int.constant 999
          %eq = smt.eq %m, %c999 : !smt.int
          // expected-note @below {{dim_0 (128) should be 999}}
          iree_codegen.smt.assert %eq, "dim_0 ({}) should be 999", %m : !smt.bool, !smt.int
        }
        return
      }
    }
  }
}

// -----

// Test: Knob-based violation -- wg_0 tile (32) does not divide dim_0 (128)
// when dim_0 is 100 (set via problem_dims). The lowering config provides
// the concrete wg_0=32 value.

#pipeline_layout2 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#gpu_target2 = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target2 = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target2}>
#translation2 = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

hal.executable @verify_knob_violation_ex {
  hal.executable.variant public @rocm target(#exec_target2) {
    hal.executable.export public @verify_knob_violation ordinal(0) layout(#pipeline_layout2)
    builtin.module {
      func.func @verify_knob_violation() attributes {translation_info = #translation2} {
        %cst = arith.constant 0.0 : f32
        %lhs = tensor.empty() : tensor<100x64xf32>
        %rhs = tensor.empty() : tensor<64x256xf32>
        %empty = tensor.empty() : tensor<100x256xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<100x256xf32>) -> tensor<100x256xf32>
        // The lowering config has workgroup = [32, 64, 0].
        // 100 % 32 != 0, so the divisibility constraint should fire.
        %result = linalg.matmul {
            lowering_config = #iree_gpu.lowering_config<{workgroup = [32, 64, 0]}>,
            root_op = #iree_codegen.root_op<set = 0>}
            ins(%lhs, %rhs : tensor<100x64xf32>, tensor<64x256xf32>)
            outs(%fill : tensor<100x256xf32>) -> tensor<100x256xf32>
        %c100 = arith.constant 100 : index
        %c256 = arith.constant 256 : index
        %c64 = arith.constant 64 : index
        // expected-error @below {{pipeline constraints violated}}
        iree_codegen.smt.constraints target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_0">,
                                  #iree_codegen.smt.int_knob<"wg_1">,
                                  #iree_codegen.smt.int_knob<"wg_2">]}
            dims(%c100, %c256, %c64) {
        ^bb0(%m: !smt.int, %n: !smt.int, %k: !smt.int):
          %wg_0 = iree_codegen.smt.knob "wg_0" : !smt.int
          %c0 = smt.int.constant 0
          %rem = smt.int.mod %m, %wg_0
          %eq = smt.eq %rem, %c0 : !smt.int
          // expected-note @below {{dim_0 must be divisible by wg_0 (100 % 32 == 0)}}
          iree_codegen.smt.assert %eq, "dim_0 must be divisible by wg_0 ({} % {} == 0)", %m, %wg_0 : !smt.bool, !smt.int, !smt.int
        }
        return
      }
    }
  }
}

// -----

// Test: Verification skipped when some knobs are missing from the config.
// The config has workgroup but not reduction, so not all knobs can be
// resolved. Verification is skipped (no error, constraints just erased).

#pipeline_layout3 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#gpu_target3 = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target3 = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target3}>
#translation3 = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

hal.executable @verify_skip_partial_knobs_ex {
  hal.executable.variant public @rocm target(#exec_target3) {
    hal.executable.export public @verify_skip_partial_knobs ordinal(0) layout(#pipeline_layout3)
    builtin.module {
      func.func @verify_skip_partial_knobs() attributes {translation_info = #translation3} {
        %cst = arith.constant 0.0 : f32
        %lhs = tensor.empty() : tensor<128x64xf32>
        %rhs = tensor.empty() : tensor<64x256xf32>
        %empty = tensor.empty() : tensor<128x256xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
        // Config has workgroup but NOT reduction. The knobs template
        // references both, so extraction fails and verification is skipped.
        %result = linalg.matmul {
            lowering_config = #iree_gpu.lowering_config<{workgroup = [32, 64, 0]}>,
            root_op = #iree_codegen.root_op<set = 0>}
            ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
            outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c64 = arith.constant 64 : index
        // No expected-error -- should be silently erased because
        // reduction knob can't be resolved.
        iree_codegen.smt.constraints target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_0">],
                     reduction = [#iree_codegen.smt.int_knob<"red_0">]}
            dims(%c128, %c256, %c64) {
        ^bb0(%m: !smt.int, %n: !smt.int, %k: !smt.int):
          %wg_0 = iree_codegen.smt.knob "wg_0" : !smt.int
          %c0 = smt.int.constant 0
          %rem = smt.int.mod %m, %wg_0
          %eq = smt.eq %rem, %c0 : !smt.int
          iree_codegen.smt.assert %eq, "divisible" : !smt.bool
        }
        return
      }
    }
  }
}

// -----

// Test: Array size mismatch between knobs template and config skips
// verification. The template has 3 workgroup knobs but config has 2.

#pipeline_layout4 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#gpu_target4 = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target4 = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target4}>
#translation4 = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

hal.executable @verify_skip_array_mismatch_ex {
  hal.executable.variant public @rocm target(#exec_target4) {
    hal.executable.export public @verify_skip_array_mismatch ordinal(0) layout(#pipeline_layout4)
    builtin.module {
      func.func @verify_skip_array_mismatch() attributes {translation_info = #translation4} {
        %cst = arith.constant 0.0 : f32
        %lhs = tensor.empty() : tensor<128x64xf32>
        %rhs = tensor.empty() : tensor<64x256xf32>
        %empty = tensor.empty() : tensor<128x256xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
        // Config workgroup has 2 elements, template has 3 -- mismatch.
        %result = linalg.matmul {
            lowering_config = #iree_gpu.lowering_config<{workgroup = [32, 64]}>,
            root_op = #iree_codegen.root_op<set = 0>}
            ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
            outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c64 = arith.constant 64 : index
        // No expected-error -- array size mismatch causes skip.
        iree_codegen.smt.constraints target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_0">,
                                  #iree_codegen.smt.int_knob<"wg_1">,
                                  #iree_codegen.smt.int_knob<"wg_2">]}
            dims(%c128, %c256, %c64) {
        ^bb0(%m: !smt.int, %n: !smt.int, %k: !smt.int):
          %wg_0 = iree_codegen.smt.knob "wg_0" : !smt.int
          %c0 = smt.int.constant 0
          %rem = smt.int.mod %m, %wg_0
          %eq = smt.eq %rem, %c0 : !smt.int
          iree_codegen.smt.assert %eq, "divisible" : !smt.bool
        }
        return
      }
    }
  }
}

// -----

// Test: workgroup_size knob resolved from translation_info.
// workgroup_size = [128, 1, 1], assert wg_x == 256 fails (wg_x is 128).

#pipeline_layout5 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#gpu_target5 = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target5 = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target5}>
#translation5 = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [128, 1, 1] subgroup_size = 64>

hal.executable @verify_wg_size_violation_ex {
  hal.executable.variant public @rocm target(#exec_target5) {
    hal.executable.export public @verify_wg_size_violation ordinal(0) layout(#pipeline_layout5)
    builtin.module {
      func.func @verify_wg_size_violation() attributes {translation_info = #translation5} {
        %cst = arith.constant 0.0 : f32
        %lhs = tensor.empty() : tensor<128x64xf32>
        %rhs = tensor.empty() : tensor<64x256xf32>
        %empty = tensor.empty() : tensor<128x256xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
        %result = linalg.matmul {
            lowering_config = #iree_gpu.lowering_config<{workgroup = [32, 64, 0]}>,
            root_op = #iree_codegen.root_op<set = 0>}
            ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
            outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c64 = arith.constant 64 : index
        // wg_x is 128 from translation_info, assert it equals 256 -- fails.
        // expected-error @below {{pipeline constraints violated}}
        iree_codegen.smt.constraints target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_0">,
                                  #iree_codegen.smt.int_knob<"wg_1">,
                                  #iree_codegen.smt.int_knob<"wg_2">],
                     workgroup_size = [#iree_codegen.smt.int_knob<"wg_x">, 1, 1]}
            dims(%c128, %c256, %c64) {
        ^bb0(%m: !smt.int, %n: !smt.int, %k: !smt.int):
          %wg_x = iree_codegen.smt.knob "wg_x" : !smt.int
          %c256_smt = smt.int.constant 256
          %eq = smt.eq %wg_x, %c256_smt : !smt.int
          // expected-note @below {{wg_x (128) must equal 256}}
          iree_codegen.smt.assert %eq, "wg_x ({}) must equal 256", %wg_x : !smt.bool, !smt.int
        }
        return
      }
    }
  }
}

// -----

// Test: OneOfKnobAttr violation -- MMA resolves to index 0 (16x16x4),
// constraint expects mma_m >= 32 but it's 16.

#pipeline_layout6 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#gpu_target6 = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_32x32x8_F16>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target6 = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target6}>
#translation6 = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

hal.executable @verify_one_of_knob_violation_ex {
  hal.executable.variant public @rocm target(#exec_target6) {
    hal.executable.export public @verify_one_of_knob_violation ordinal(0) layout(#pipeline_layout6)
    builtin.module {
      func.func @verify_one_of_knob_violation() attributes {translation_info = #translation6} {
        %cst = arith.constant 0.0 : f32
        %lhs = tensor.empty() : tensor<128x64xf32>
        %rhs = tensor.empty() : tensor<64x256xf32>
        %empty = tensor.empty() : tensor<128x256xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
        // Config has mma_kind = MFMA_F32_16x16x4_F32 (index 0).
        %result = linalg.matmul {
            lowering_config = #iree_gpu.lowering_config<{
                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
                workgroup = [128, 128, 0]}>,
            root_op = #iree_codegen.root_op<set = 0>}
            ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
            outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c64 = arith.constant 64 : index
        // expected-error @below {{pipeline constraints violated}}
        iree_codegen.smt.constraints target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx",
                [#iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
                 #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>]>,
                     workgroup = [#iree_codegen.smt.int_knob<"wg_0">,
                                  #iree_codegen.smt.int_knob<"wg_1">,
                                  #iree_codegen.smt.int_knob<"wg_2">]}
            dims(%c128, %c256, %c64) {
        ^bb0(%m: !smt.int, %n: !smt.int, %k: !smt.int):
          // mma_idx resolves to 0 (MFMA_F32_16x16x4_F32).
          %mma_idx = iree_codegen.smt.knob "mma_idx" : !smt.int
          // Lookup mma_m: index 0 -> 16, index 1 -> 32.
          %mma_m = iree_codegen.smt.lookup %mma_idx [0, 1] -> [16, 32] : !smt.int
          %c32 = smt.int.constant 32
          %ge = smt.int.cmp ge %mma_m, %c32
          // expected-note @below {{mma_m (16) must be >= 32}}
          iree_codegen.smt.assert %ge, "mma_m ({}) must be >= 32", %mma_m : !smt.bool, !smt.int
        }
        return
      }
    }
  }
}

// -----

// Test: OneOfKnobAttr skip -- config MMA matches neither of two options,
// extraction fails, verification silently skipped.

#pipeline_layout7 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#gpu_target7 = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_32x32x8_F16>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target7 = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target7}>
#translation7 = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

hal.executable @verify_one_of_knob_skip_ex {
  hal.executable.variant public @rocm target(#exec_target7) {
    hal.executable.export public @verify_one_of_knob_skip ordinal(0) layout(#pipeline_layout7)
    builtin.module {
      func.func @verify_one_of_knob_skip() attributes {translation_info = #translation7} {
        %cst = arith.constant 0.0 : f32
        %lhs = tensor.empty() : tensor<128x64xf32>
        %rhs = tensor.empty() : tensor<64x256xf32>
        %empty = tensor.empty() : tensor<128x256xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
        // Config has MFMA_F32_16x16x16_F16 but options only list
        // MFMA_F32_16x16x4_F32 and MFMA_F32_32x32x8_F16 -- no match.
        %result = linalg.matmul {
            lowering_config = #iree_gpu.lowering_config<{
                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
                workgroup = [128, 128, 0]}>,
            root_op = #iree_codegen.root_op<set = 0>}
            ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
            outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c64 = arith.constant 64 : index
        // No expected-error -- config MMA not in options, skip.
        iree_codegen.smt.constraints target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx",
                [#iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
                 #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>]>,
                     workgroup = [#iree_codegen.smt.int_knob<"wg_0">,
                                  #iree_codegen.smt.int_knob<"wg_1">,
                                  #iree_codegen.smt.int_knob<"wg_2">]}
            dims(%c128, %c256, %c64) {
        ^bb0(%m: !smt.int, %n: !smt.int, %k: !smt.int):
          %mma_idx = iree_codegen.smt.knob "mma_idx" : !smt.int
          %c0 = smt.int.constant 0
          %eq = smt.eq %mma_idx, %c0 : !smt.int
          iree_codegen.smt.assert %eq, "should not fire" : !smt.bool
        }
        return
      }
    }
  }
}
