// RUN: iree-opt --split-input-file \
// RUN:   --iree-codegen-experimental-verify-pipeline-constraints \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-codegen-insert-smt-constraints,iree-codegen-verify-smt-constraints))' %s \
// RUN:   --verify-diagnostics \
// RUN:   | FileCheck %s

// Test: End-to-end failure from generated constraints.
// This ensures constraints are inserted and that verification reports violations.
// It also catches cases where incorrect knob templates skip verification.
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

func.func @matmul_e2e_generated_violation_vd(
    %lhs: tensor<128x64xf32>, %rhs: tensor<64x256xf32>)
    -> tensor<128x256xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
      ins(%cst : f32) outs(%init : tensor<128x256xf32>)
      -> tensor<128x256xf32>
  // expected-error @below {{pipeline constraints violated}}
  // expected-note @below {{dim_0 must be divisible by wg_0 (128 % 48 == 0)}}
  %result = linalg.matmul {
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [48, 64, 0],
          reduction = [0, 0, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>,
      root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
      outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %result : tensor<128x256xf32>
}

// -----

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
    pipeline = #iree_gpu.pipeline<TileAndFuse>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

func.func @matmul_e2e_generated_violation_tf(
    %lhs: tensor<128x64xf32>, %rhs: tensor<64x256xf32>)
    -> tensor<128x256xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
      ins(%cst : f32) outs(%init : tensor<128x256xf32>)
      -> tensor<128x256xf32>
  // expected-error @below {{pipeline constraints violated}}
  // expected-note @below {{dim_0 must be divisible by wg_0 (128 % 48 == 0)}}
  %result = linalg.matmul {
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [48, 64, 0],
          reduction = [0, 0, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup = [1, 1, 0]}>,
      root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
      outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %result : tensor<128x256xf32>
}

// -----

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

func.func @conv_e2e_generated_violation_vd(
    %input: tensor<1x18x130x64xf32>, %filter: tensor<3x3x64x128xf32>)
    -> tensor<1x16x128x128xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<1x16x128x128xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 1>}
      ins(%cst : f32) outs(%init : tensor<1x16x128x128xf32>)
      -> tensor<1x16x128x128xf32>
  // expected-error @below {{pipeline constraints violated}}
  // expected-note @below {{dim_2 must be divisible by wg_2 (128 % 48 == 0)}}
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : tensor<2xi64>,
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [1, 1, 48, 64, 0, 0, 0],
          reduction = [0, 0, 0, 0, 1, 1, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup_basis = [[1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 2, 3, 4, 5, 6]]}>,
      root_op = #iree_codegen.root_op<set = 1>,
      strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x18x130x64xf32>,
                               tensor<3x3x64x128xf32>)
      outs(%fill : tensor<1x16x128x128xf32>) -> tensor<1x16x128x128xf32>
  return %result : tensor<1x16x128x128xf32>
}

// -----

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
    pipeline = #iree_gpu.pipeline<TileAndFuse>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

func.func @conv_e2e_generated_violation_tf(
    %input: tensor<1x18x130x64xf32>, %filter: tensor<3x3x64x128xf32>)
    -> tensor<1x16x128x128xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<1x16x128x128xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 1>}
      ins(%cst : f32) outs(%init : tensor<1x16x128x128xf32>)
      -> tensor<1x16x128x128xf32>
  // expected-error @below {{pipeline constraints violated}}
  // expected-note @below {{dim_2 must be divisible by wg_2 (128 % 48 == 0)}}
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : tensor<2xi64>,
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [1, 1, 48, 64, 0, 0, 0],
          reduction = [0, 0, 0, 0, 1, 1, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup = [1, 1, 1, 1, 0, 0, 0]}>,
      root_op = #iree_codegen.root_op<set = 1>,
      strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x18x130x64xf32>,
                               tensor<3x3x64x128xf32>)
      outs(%fill : tensor<1x16x128x128xf32>) -> tensor<1x16x128x128xf32>
  return %result : tensor<1x16x128x128xf32>
}

// -----

// Test: End-to-end constraint insertion and verification.
// Use the same shapes as above but with divisible workgroup sizes.
// It should pass verification and have constraints erased.
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

func.func @matmul_e2e_constraints_erased_vd(
    %lhs: tensor<128x64xf32>, %rhs: tensor<64x256xf32>)
    -> tensor<128x256xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
      ins(%cst : f32) outs(%init : tensor<128x256xf32>)
      -> tensor<128x256xf32>
  %result = linalg.matmul {
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [32, 64, 0],
          reduction = [0, 0, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup_basis = [[1, 1, 1], [0, 1, 2]]}>,
      root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
      outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %result : tensor<128x256xf32>
}

// CHECK-LABEL: func.func @matmul_e2e_constraints_erased_vd
// CHECK:       linalg.matmul
// CHECK-NOT:   iree_codegen.smt.constraints

func.func @conv_e2e_constraints_erased_vd(
    %input: tensor<1x18x130x64xf32>, %filter: tensor<3x3x64x128xf32>)
    -> tensor<1x16x128x128xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<1x16x128x128xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 1>}
      ins(%cst : f32) outs(%init : tensor<1x16x128x128xf32>)
      -> tensor<1x16x128x128xf32>
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : tensor<2xi64>,
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [1, 1, 64, 64, 0, 0, 0],
          reduction = [0, 0, 0, 0, 1, 1, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup_basis = [[1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 2, 3, 4, 5, 6]]}>,
      root_op = #iree_codegen.root_op<set = 1>,
      strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x18x130x64xf32>,
                               tensor<3x3x64x128xf32>)
      outs(%fill : tensor<1x16x128x128xf32>) -> tensor<1x16x128x128xf32>
  return %result : tensor<1x16x128x128xf32>
}

// CHECK-LABEL: func.func @conv_e2e_constraints_erased_vd
// CHECK:       linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:   iree_codegen.smt.constraints

// -----

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
    pipeline = #iree_gpu.pipeline<TileAndFuse>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

func.func @matmul_e2e_constraints_erased_tf(
    %lhs: tensor<128x64xf32>, %rhs: tensor<64x256xf32>)
    -> tensor<128x256xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
      ins(%cst : f32) outs(%init : tensor<128x256xf32>)
      -> tensor<128x256xf32>
  %result = linalg.matmul {
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [16, 16, 0],
          reduction = [0, 0, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup = [1, 1, 0]}>,
      root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
      outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %result : tensor<128x256xf32>
}

// CHECK-LABEL: func.func @matmul_e2e_constraints_erased_tf
// CHECK:       linalg.matmul
// CHECK-NOT:   iree_codegen.smt.constraints

func.func @conv_e2e_constraints_erased_tf(
    %input: tensor<1x18x130x64xf32>, %filter: tensor<3x3x64x128xf32>)
    -> tensor<1x16x128x128xf32>
    attributes {hal.executable.target = #exec_target,
                translation_info = #translation} {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<1x16x128x128xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 1>}
      ins(%cst : f32) outs(%init : tensor<1x16x128x128xf32>)
      -> tensor<1x16x128x128xf32>
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : tensor<2xi64>,
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [1, 1, 16, 64, 0, 0, 0],
          reduction = [0, 0, 0, 0, 1, 1, 16],
          mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
          subgroup = [0, 1, 0, 1, 1, 0, 0]}>,
      root_op = #iree_codegen.root_op<set = 1>,
      strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x18x130x64xf32>,
                               tensor<3x3x64x128xf32>)
      outs(%fill : tensor<1x16x128x128xf32>) -> tensor<1x16x128x128xf32>
  return %result : tensor<1x16x128x128xf32>
}

// CHECK-LABEL: func.func @conv_e2e_constraints_erased_tf
// CHECK:       linalg.conv_2d_nhwc_hwcf
// CHECK-NOT:   iree_codegen.smt.constraints

// -----

// VectorDistribute attention: a configuration that satisfies all v0
// constraints. m_tile divides M (1024 / 128 = 8), n_tile divides N
// (64 / 64 = 1), red_k2 divides K2 (1024 / 64 = 16), every divisibility
// by the chosen MMA intrinsic shape holds, and total threads fit.
//
// m_tile (128) and n_tile (64) are intentionally asymmetric so that an
// accidental axis swap in the emitter (e.g., binding `wg_m` to the N
// position or vice versa) surfaces as a verification failure here
// rather than silently passing under symmetry.
//
// Implied SMT-internal knob values (not in the user's lowering_config —
// these are SMT-aux, see kSMTAuxKnobKeys in SMTConstraintUtils.h; the
// verifier resolves them as unknown and silently skips constraints
// that reference them. Spelling them out here so a future contributor
// can adjust m_tile / n_tile without re-deriving by hand):
//
//   pv_mma_m = pv_mma_n = pv_mma_k = 16     (MFMA_F32_16x16x16_F16)
//   sg_m_cnt = 4, sg_n_cnt = 1              (from subgroup_basis)
//   sg_m_tcnt = m_tile / (sg_m_cnt × pv_mma_m) = 128 / (4 × 16) = 2
//   sg_n_tcnt = n_tile / (sg_n_cnt × pv_mma_n) = 64 / (1 × 16) = 4
//   sg_k_tcnt = red_k2 / pv_mma_k            = 64 / 16            = 4
//   sg_num    = sg_m_cnt × sg_n_cnt          = 4                  (== Constraint 5 pin)
//   total_threads = sg_num × sg_size         = 4 × 64 = 256       (≤ 1024)
//
// LDS budget (approx): qkShared (Q tile + K tile) + pvShared (P tile +
// V tile) at f16 (2 B/elt) is well under 65536 B for this small fixture.

#gpu_target_attn = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x16_F16>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target_attn = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target_attn}>
#translation_attn = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [256, 1, 1] subgroup_size = 64>

#qmap = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#kmap = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#vmap = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#smap = affine_map<(d0, d1, d2, d3, d4) -> ()>
#omap = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
#stmap = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>

func.func @attention_e2e_constraints_passing(
    %q: tensor<4x1024x64xf16>, %k: tensor<4x1024x64xf16>,
    %v: tensor<4x1024x64xf16>, %scale: f16,
    %out_init: tensor<4x1024x64xf16>,
    %max_init: tensor<4x1024xf16>, %sum_init: tensor<4x1024xf16>)
    -> (tensor<4x1024x64xf16>, tensor<4x1024xf16>, tensor<4x1024xf16>)
    attributes {hal.executable.target = #exec_target_attn,
                translation_info = #translation_attn} {
  %res:3 = iree_linalg_ext.online_attention {
      root_op = #iree_codegen.root_op<set = 0>,
      indexing_maps = [#qmap, #kmap, #vmap, #smap, #omap, #stmap, #stmap],
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [1, 128, 0, 0, 64],
          reduction = [0, 0, 0, 64, 0],
          promote_operands = [0, 1, 2],
          promotion_types = [#iree_gpu.derived_thread_config,
                             #iree_gpu.derived_thread_config,
                             #iree_gpu.derived_thread_config],
          decomposition_config = {
            qk_attrs = {lowering_config = #iree_gpu.lowering_config<{
              mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
              promote_operands = [0, 1],
              promotion_types = [#iree_gpu.derived_thread_config,
                                 #iree_gpu.derived_thread_config],
              subgroup_basis = [[1, 4, 1, 1, 1], [0, 1, 2, 3]]}>},
            pv_attrs = {lowering_config = #iree_gpu.lowering_config<{
              mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
              promote_operands = [1],
              promotion_types = [#iree_gpu.derived_thread_config],
              subgroup_basis = [[1, 4, 1, 1, 1], [0, 1, 3, 4]]}>}}}>}
      ins(%q, %k, %v, %scale : tensor<4x1024x64xf16>, tensor<4x1024x64xf16>,
                                tensor<4x1024x64xf16>, f16)
      outs(%out_init, %max_init, %sum_init :
              tensor<4x1024x64xf16>, tensor<4x1024xf16>, tensor<4x1024xf16>) {
        ^bb0(%score: f32):
          iree_linalg_ext.yield %score : f32
       } -> tensor<4x1024x64xf16>, tensor<4x1024xf16>, tensor<4x1024xf16>
  return %res#0, %res#1, %res#2
      : tensor<4x1024x64xf16>, tensor<4x1024xf16>, tensor<4x1024xf16>
}

// CHECK-LABEL: func.func @attention_e2e_constraints_passing
// CHECK:       iree_linalg_ext.online_attention
// CHECK-NOT:   iree_codegen.smt.constraints

// -----

// Attention configuration with a violating m_tile (60) that does not
// divide M (1024). The verifier must surface the corresponding note.

#gpu_target_attn_v = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x16_F16>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target_attn_v = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target_attn_v}>
#translation_attn_v = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [256, 1, 1] subgroup_size = 64>

#qmap_v = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#kmap_v = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#vmap_v = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#smap_v = affine_map<(d0, d1, d2, d3, d4) -> ()>
#omap_v = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
#stmap_v = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>

func.func @attention_e2e_generated_violation(
    %q: tensor<4x1024x64xf16>, %k: tensor<4x1024x64xf16>,
    %v: tensor<4x1024x64xf16>, %scale: f16,
    %out_init: tensor<4x1024x64xf16>,
    %max_init: tensor<4x1024xf16>, %sum_init: tensor<4x1024xf16>)
    -> (tensor<4x1024x64xf16>, tensor<4x1024xf16>, tensor<4x1024xf16>)
    attributes {hal.executable.target = #exec_target_attn_v,
                translation_info = #translation_attn_v} {
  // expected-error @below {{pipeline constraints violated}}
  // expected-note @below {{dim_1 must be divisible by m_tile (1024 % 96 == 0)}}
  %res:3 = iree_linalg_ext.online_attention {
      root_op = #iree_codegen.root_op<set = 0>,
      indexing_maps = [#qmap_v, #kmap_v, #vmap_v, #smap_v, #omap_v, #stmap_v, #stmap_v],
      lowering_config = #iree_gpu.lowering_config<{
          workgroup = [1, 96, 0, 0, 64],
          reduction = [0, 0, 0, 64, 0],
          promote_operands = [0, 1, 2],
          promotion_types = [#iree_gpu.derived_thread_config,
                             #iree_gpu.derived_thread_config,
                             #iree_gpu.derived_thread_config],
          decomposition_config = {
            qk_attrs = {lowering_config = #iree_gpu.lowering_config<{
              mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
              promote_operands = [0, 1],
              promotion_types = [#iree_gpu.derived_thread_config,
                                 #iree_gpu.derived_thread_config],
              subgroup_basis = [[1, 4, 1, 1, 1], [0, 1, 2, 3]]}>},
            pv_attrs = {lowering_config = #iree_gpu.lowering_config<{
              mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
              promote_operands = [1],
              promotion_types = [#iree_gpu.derived_thread_config],
              subgroup_basis = [[1, 4, 1, 1, 1], [0, 1, 3, 4]]}>}}}>}
      ins(%q, %k, %v, %scale : tensor<4x1024x64xf16>, tensor<4x1024x64xf16>,
                                tensor<4x1024x64xf16>, f16)
      outs(%out_init, %max_init, %sum_init :
              tensor<4x1024x64xf16>, tensor<4x1024xf16>, tensor<4x1024xf16>) {
        ^bb0(%score: f32):
          iree_linalg_ext.yield %score : f32
       } -> tensor<4x1024x64xf16>, tensor<4x1024xf16>, tensor<4x1024xf16>
  return %res#0, %res#1, %res#2
      : tensor<4x1024x64xf16>, tensor<4x1024xf16>, tensor<4x1024xf16>
}
