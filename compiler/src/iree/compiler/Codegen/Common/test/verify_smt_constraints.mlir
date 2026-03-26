// RUN: iree-opt --split-input-file \
// RUN:   --iree-codegen-experimental-verify-pipeline-constraints \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-verify-smt-constraints)))))' %s \
// RUN:   | FileCheck %s

// Test: Matching pipeline -- constraints are evaluated and erased.

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

hal.executable @verify_match_ex {
  hal.executable.variant public @rocm target(#exec_target) {
    hal.executable.export public @verify_match ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @verify_match() attributes {translation_info = #translation} {
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
        iree_codegen.smt.constraints target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_0">,
                                  #iree_codegen.smt.int_knob<"wg_1">,
                                  #iree_codegen.smt.int_knob<"wg_2">]}
            dims(%c128, %c256, %c64) {
        ^bb0(%m: !smt.int, %n: !smt.int, %k: !smt.int):
          %c128_smt = smt.int.constant 128
          %eq = smt.eq %m, %c128_smt : !smt.int
          iree_codegen.smt.assert %eq, "dim_0 ({}) == 128", %m : !smt.bool, !smt.int
        }
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable public @verify_match_ex
// CHECK:         linalg.matmul
// CHECK-NOT:     iree_codegen.smt.constraints

// -----

// Test: Mismatching pipeline -- constraints are silently erased.

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
// Chosen pipeline is TileAndFuse, but constraints are for VectorDistribute.
#translation2 = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<TileAndFuse>
    workgroup_size = [64, 1, 1] subgroup_size = 64>

hal.executable @verify_mismatch_ex {
  hal.executable.variant public @rocm target(#exec_target2) {
    hal.executable.export public @verify_mismatch ordinal(0) layout(#pipeline_layout2)
    builtin.module {
      func.func @verify_mismatch() attributes {translation_info = #translation2} {
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
        iree_codegen.smt.constraints target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_0">]}
            dims(%c128, %c256, %c64) {
        ^bb0(%m: !smt.int, %n: !smt.int, %k: !smt.int):
          %c128_smt = smt.int.constant 128
          %eq = smt.eq %m, %c128_smt : !smt.int
          iree_codegen.smt.assert %eq, "dim_0 ({}) == 128", %m : !smt.bool, !smt.int
        }
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable public @verify_mismatch_ex
// CHECK:         linalg.matmul
// CHECK-NOT:     iree_codegen.smt.constraints

// -----

// Test: and(false, unknown) short-circuits to false -- no violation because
// the assert condition is or(true_static, unknown_knob) which is true.

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

hal.executable @verify_or_short_circuit_ex {
  hal.executable.variant public @rocm target(#exec_target3) {
    hal.executable.export public @verify_or_short_circuit ordinal(0) layout(#pipeline_layout3)
    builtin.module {
      func.func @verify_or_short_circuit() attributes {translation_info = #translation3} {
        %cst = arith.constant 0.0 : f32
        %lhs = tensor.empty() : tensor<128x64xf32>
        %rhs = tensor.empty() : tensor<64x128xf32>
        %empty = tensor.empty() : tensor<128x128xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
        %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
            ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x128xf32>)
            outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
        %c128 = arith.constant 128 : index
        %c64 = arith.constant 64 : index
        iree_codegen.smt.constraints target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_0">]}
            dims(%c128, %c128, %c64) {
        ^bb0(%m: !smt.int, %n: !smt.int, %k: !smt.int):
          // Static check: dim_0 == 128 (true).
          %c128_smt = smt.int.constant 128
          %static_ok = smt.eq %m, %c128_smt : !smt.int
          // Knob check: unknown (knob not resolved).
          %wg = iree_codegen.smt.knob "wg_0" : !smt.int
          %c0 = smt.int.constant 0
          %rem = smt.int.mod %m, %wg
          %knob_ok = smt.eq %rem, %c0 : !smt.int
          // or(true, unknown) should short-circuit to true.
          %combined = smt.or %static_ok, %knob_ok
          iree_codegen.smt.assert %combined, "or short-circuit" : !smt.bool
        }
        return
      }
    }
  }
}

// or(true, unknown) = true, so no violation -- constraints erased cleanly.
// CHECK-LABEL: hal.executable public @verify_or_short_circuit_ex
// CHECK:         linalg.matmul
// CHECK-NOT:     iree_codegen.smt.constraints

// -----

// Test: exercises ALL evaluator ops in a single constraint region with
// static dims and no knobs resolved. All assertions pass.

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

hal.executable @verify_all_ops_ex {
  hal.executable.variant public @rocm target(#exec_target4) {
    hal.executable.export public @verify_all_ops ordinal(0) layout(#pipeline_layout4)
    builtin.module {
      func.func @verify_all_ops() attributes {translation_info = #translation4} {
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
        iree_codegen.smt.constraints target = <set = 0>,
            pipeline = #iree_gpu.pipeline<VectorDistribute>,
            knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_0">]}
            dims(%c128, %c256, %c64) {
        ^bb0(%m: !smt.int, %n: !smt.int, %k: !smt.int):
          // IntConstantOp
          %c2 = smt.int.constant 2
          %c4 = smt.int.constant 4
          %c8 = smt.int.constant 8
          %c32 = smt.int.constant 32
          %c64_smt = smt.int.constant 64
          %c128_smt = smt.int.constant 128
          %c0 = smt.int.constant 0

          // KnobOp (unresolved -- propagates unknown)
          %wg = iree_codegen.smt.knob "wg_0" : !smt.int

          // IntMulOp: 2 * 64 = 128
          %mul = smt.int.mul %c2, %c64_smt

          // IntAddOp: 64 + 64 = 128
          %add = smt.int.add %c64_smt, %c64_smt

          // IntSubOp: 128 - 0 = 128
          %sub = smt.int.sub %c128_smt, %c0

          // IntDivOp: 128 / 4 = 32
          %div = smt.int.div %c128_smt, %c4

          // IntModOp: 128 % 64 = 0
          %mod = smt.int.mod %m, %c64_smt

          // EqOp: 128 == 128
          %eq_mul = smt.eq %mul, %m : !smt.int

          // EqOp: mod == 0
          %eq_mod = smt.eq %mod, %c0 : !smt.int

          // IntCmpOp: 128 >= 8
          %ge = smt.int.cmp ge %m, %c8

          // IntCmpOp: div (32) == 32
          %eq_div = smt.eq %div, %c32 : !smt.int

          // AndOp: eq_mul && eq_mod && ge && eq_div
          %and1 = smt.and %eq_mul, %eq_mod, %ge, %eq_div

          // NotOp: !false = true (128 < 8 is false, so not gives true)
          %lt = smt.int.cmp lt %m, %c8
          %not_lt = smt.not %lt

          // OrOp: and1 || false (and1 is true, so true)
          %false_val = smt.int.cmp lt %m, %c0
          %or1 = smt.or %and1, %false_val

          // IteOp: if eq_mul then 128 else 0 -> 128
          %ite_val = smt.ite %eq_mul, %c128_smt, %c0 : !smt.int
          %eq_ite = smt.eq %ite_val, %c128_smt : !smt.int

          // LookupOp: lookup key 2 in [1,2,3] -> [10,20,30] = 20
          %lookup_val = iree_codegen.smt.lookup %c2 [1, 2, 3] -> [10, 20, 30] : !smt.int
          %c20 = smt.int.constant 20
          %eq_lookup = smt.eq %lookup_val, %c20 : !smt.int

          // Combine everything: and1 && not_lt && or1 && eq_ite && eq_lookup
          %all = smt.and %and1, %not_lt, %or1, %eq_ite, %eq_lookup

          // Also check add and sub via eq
          %eq_add = smt.eq %add, %c128_smt : !smt.int
          %eq_sub = smt.eq %sub, %c128_smt : !smt.int
          %final = smt.and %all, %eq_add, %eq_sub

          // AssertOp
          iree_codegen.smt.assert %final, "all evaluator ops pass" : !smt.bool
        }
        return
      }
    }
  }
}

// All assertions pass with static dims -- constraints erased cleanly.
// CHECK-LABEL: hal.executable public @verify_all_ops_ex
// CHECK:         linalg.matmul
// CHECK-NOT:     iree_codegen.smt.constraints

// -----

// Test: OneOfKnobAttr resolves MMA attr to index, LookupOp derives shape,
// assertion passes.

#pipeline_layout5 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#gpu_target5 = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_32x32x8_F16>],
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
    workgroup_size = [64, 1, 1] subgroup_size = 64>

hal.executable @verify_one_of_knob_ex {
  hal.executable.variant public @rocm target(#exec_target5) {
    hal.executable.export public @verify_one_of_knob ordinal(0) layout(#pipeline_layout5)
    builtin.module {
      func.func @verify_one_of_knob() attributes {translation_info = #translation5} {
        %cst = arith.constant 0.0 : f32
        %lhs = tensor.empty() : tensor<128x64xf32>
        %rhs = tensor.empty() : tensor<64x256xf32>
        %empty = tensor.empty() : tensor<128x256xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
        // Config has mma_kind matching option index 1 (MFMA_F32_32x32x8_F16).
        %result = linalg.matmul {
            lowering_config = #iree_gpu.lowering_config<{
                mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
                workgroup = [128, 128, 0]}>,
            root_op = #iree_codegen.root_op<set = 0>}
            ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
            outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c64 = arith.constant 64 : index
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
          // mma_idx resolves to 1 (MFMA_F32_32x32x8_F16 is at index 1).
          %mma_idx = iree_codegen.smt.knob "mma_idx" : !smt.int
          // Lookup mma_m: index 0 -> 16, index 1 -> 32.
          %mma_m = iree_codegen.smt.lookup %mma_idx [0, 1] -> [16, 32] : !smt.int
          %c32 = smt.int.constant 32
          %eq = smt.eq %mma_m, %c32 : !smt.int
          iree_codegen.smt.assert %eq, "mma_m should be 32" : !smt.bool
        }
        return
      }
    }
  }
}

// OneOfKnob matches index 1, lookup gives 32, assertion passes.
// CHECK-LABEL: hal.executable public @verify_one_of_knob_ex
// CHECK:         linalg.matmul
// CHECK-NOT:     iree_codegen.smt.constraints
