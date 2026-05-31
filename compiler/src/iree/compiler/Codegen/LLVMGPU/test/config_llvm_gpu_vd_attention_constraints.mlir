// RUN: iree-opt --split-input-file \
// RUN:   --iree-codegen-experimental-verify-pipeline-constraints \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-codegen-insert-smt-constraints))' %s \
// RUN:   | FileCheck %s

// Per-constraint coverage for the VectorDistribute attention emitter.
// Locks in the attention knob template (workgroup, reduction,
// promote_operands, decomposition_config carrying per-matmul lowering_config
// templates for QK / PV, workgroup_size, subgroup_size) and verifies the
// v0 assert families.

#gpu_target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target}>

#qmap = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#kmap = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#vmap = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#smap = affine_map<(d0, d1, d2, d3, d4) -> ()>
#omap = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
#maxmap = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
#summap = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>

func.func @attention_f16(%q: tensor<4x1024x64xf16>, %k: tensor<4x1024x64xf16>,
                         %v: tensor<4x1024x64xf16>, %scale: f16,
                         %out_init: tensor<4x1024x64xf16>,
                         %max_init: tensor<4x1024xf16>,
                         %sum_init: tensor<4x1024xf16>)
    -> (tensor<4x1024x64xf16>, tensor<4x1024xf16>, tensor<4x1024xf16>)
    attributes {hal.executable.target = #exec_target} {
  %res:3 = iree_linalg_ext.online_attention {
      root_op = #iree_codegen.root_op<set = 0>,
      indexing_maps = [#qmap, #kmap, #vmap, #smap, #omap, #maxmap, #summap]}
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

// CHECK-LABEL: func.func @attention_f16
// CHECK:       iree_linalg_ext.online_attention {{.+}} #iree_codegen.root_op<set = [[SET:[0-9]+]]>
// CHECK:       iree_codegen.smt.constraints target = <set = [[SET]]>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
// CHECK-NEXT:  knobs = {
// Top-level lowering_config mirrors KernelConfig.cpp's attention output:
// workgroup (1 at batch dims, m_tile/n_tile knobs at innermost M/N), reduction
// (red_k2 knob at K2), promote_operands + matching promotion_types, and
// decomposition_config carrying per-matmul LoweringConfigAttr templates.
// CHECK-DAG:   workgroup = [1, #iree_codegen.smt.int_knob<"m_tile">, 0, 0, #iree_codegen.smt.int_knob<"n_tile">]
// CHECK-DAG:   reduction = [0, 0, 0, #iree_codegen.smt.int_knob<"red_k2">, 0]
// CHECK-DAG:   promote_operands = [0, 1, 2]
// CHECK-DAG:   promotion_types = [#iree_gpu.derived_thread_config, #iree_gpu.derived_thread_config, #iree_gpu.derived_thread_config]
// CHECK-DAG:   decomposition_config = {
// Per-matmul lowering_configs are typed LoweringConfigAttr (not raw DictAttr)
// so attention codegen can cast them; they carry mma_kind (knob),
// promote_operands + promotion_types, and subgroup_basis whose counts
// reference the top-level sg_m_cnt / sg_n_cnt knobs at the M / N positions.
// CHECK-DAG:     qk_attrs = {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_codegen.smt.one_of_knob<"qk_mma_idx",
// CHECK-DAG:     pv_attrs = {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_codegen.smt.one_of_knob<"pv_mma_idx",
// CHECK-DAG:     subgroup_basis = {{\[\[}}1, #iree_codegen.smt.int_knob<"sg_m_cnt">, 1, 1, #iree_codegen.smt.int_knob<"sg_n_cnt">{{\]}}, {{\[}}0, 1, 2, 3{{\]\]}}
// CHECK-DAG:     subgroup_basis = {{\[\[}}1, #iree_codegen.smt.int_knob<"sg_m_cnt">, 1, 1, #iree_codegen.smt.int_knob<"sg_n_cnt">{{\]}}, {{\[}}0, 1, 3, 4{{\]\]}}
// CHECK-DAG:   workgroup_size = [#iree_codegen.smt.int_knob<"wg_size_x">, 1, 1]
// CHECK-DAG:   subgroup_size = #iree_codegen.smt.int_knob<"sg_size">
// CHECK-SAME:  }

// Constraint families. CHECK-DAG so the order inside the region doesn't
// matter; one per assert message family that the emitter must produce.
//
// Ordering anchor -- keep the single-intrinsic_mn asserts near the
// layout-binding asserts so the emitted template stays easy to audit. The
// three ordered CHECK lines below pin only that local grouping; all CHECK-DAGs
// further down are otherwise unconstrained except that they must match after
// the last anchor.
// CHECK:     "qk_mma_m == pv_mma_m (single intrinsic_mn)"
// CHECK:     "qk_mma_n == pv_mma_n (single intrinsic_mn)"
// CHECK:     "qk_acc.element_x == pv_acc.element_x"
//
// Top-level (sg_size pin, tile bounds, factorization, sg counts):
// CHECK-DAG: "sg_size == preferred_subgroup_size"
// CHECK-DAG: "dim_1 must be divisible by m_tile ({} % {} == 0)"
// CHECK-DAG: "dim_4 must be divisible by n_tile ({} % {} == 0)"
// CHECK-DAG: "dim_3 must be divisible by red_k2 ({} % {} == 0)"
// CHECK-DAG: "m_tile >= pv_mma_m"
// CHECK-DAG: "n_tile >= pv_mma_n"
// CHECK-DAG: "red_k2 >= pv_mma_k"
// CHECK-DAG: "m_tile <= 512 (max VGPRs)"
// CHECK-DAG: "n_tile <= 512 (max VGPRs)"
// CHECK-DAG: "red_k2 <= 512 (max VGPRs)"
// CHECK-DAG: "m_tile must be divisible by sg_m_cnt * pv_mma_m"
// CHECK-DAG: "n_tile must be divisible by sg_n_cnt * pv_mma_n"
// CHECK-DAG: "red_k2 must be divisible by pv_mma_k"
// CHECK-DAG: "dim_k1 must be divisible by qk_mma_k ({} % {} == 0)"
// (Single-intrinsic_mn invariant is locked in by the ordered CHECK
// lines above; no CHECK-DAG mirror here would consume the same
// occurrence twice.)
// CHECK-DAG: "sg_m_cnt >= 1"
// CHECK-DAG: "sg_m_cnt <= 32"
// CHECK-DAG: "sg_n_cnt == 1"
// CHECK-DAG: "sg_m_cnt * sg_n_cnt == 4"
// CHECK-DAG: "sg_m_cnt * sg_n_cnt * sg_size <= max_threads"
// CHECK-DAG: "wg_size_x == sg_m_cnt * sg_n_cnt * sg_size"
// CHECK-DAG: "per-matmul shared memory must fit in workgroup memory"
// qk_acc == pv_acc match invariant (6 field equalities):
// CHECK-DAG: "qk_acc.element_y == pv_acc.element_y"
// CHECK-DAG: "qk_acc.thread_x == pv_acc.thread_x"
// CHECK-DAG: "qk_acc.thread_y == pv_acc.thread_y"
// CHECK-DAG: "qk_acc.tstrides_x == pv_acc.tstrides_x"
// CHECK-DAG: "qk_acc.tstrides_y == pv_acc.tstrides_y"
// Schedule-validity (per is_valid_vector_distribute_mma_schedule):
// CHECK-DAG: "qk_schedule.m must be divisible by the derived M workgroup tile ({} % {} == 0)"
// CHECK-DAG: "qk_schedule.n must be divisible by the derived N workgroup tile ({} % {} == 0)"
// CHECK-DAG: "qk_schedule.k must be divisible by the derived K workgroup tile ({} % {} == 0)"
// CHECK-DAG: "qk_schedule.lhs inner dim distributable across wg_threads"
// CHECK-DAG: "qk_schedule.rhs inner dim distributable across wg_threads"
// CHECK-DAG: "pv_schedule.m must be divisible by the derived M workgroup tile ({} % {} == 0)"
// CHECK-DAG: "pv_schedule.n must be divisible by the derived N workgroup tile ({} % {} == 0)"
// CHECK-DAG: "pv_schedule.k must be divisible by the derived K workgroup tile ({} % {} == 0)"
// CHECK-DAG: "pv_schedule.lhs inner dim distributable across wg_threads"
// CHECK-DAG: "pv_schedule.rhs inner dim distributable across wg_threads"
