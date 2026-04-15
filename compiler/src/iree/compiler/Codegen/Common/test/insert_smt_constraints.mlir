// RUN: iree-opt --split-input-file \
// RUN:   --iree-codegen-experimental-verify-pipeline-constraints \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-insert-smt-constraints)))))' %s \
// RUN:   | FileCheck %s

// Test: Static f32 matmul 128x256x64 with workgroup tile divisibility.

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

hal.executable @matmul_f32_ex {
  hal.executable.variant public @rocm target(#exec_target) {
    hal.executable.export public @matmul_f32 ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_f32() {
        %cst = arith.constant 0.0 : f32
        %lhs = tensor.empty() : tensor<128x64xf32>
        %rhs = tensor.empty() : tensor<64x256xf32>
        %empty = tensor.empty() : tensor<128x256xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
        %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
            ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
            outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
        return
      }
    }
  }
}

// Only the root op matmul gets constraints.
// CHECK-LABEL: hal.executable public @matmul_f32_ex
// CHECK:         func.func @matmul_f32
// CHECK:           linalg.fill
// CHECK-NOT:       iree_codegen.smt.constraints
// CHECK:           linalg.matmul
//
// CHECK:           iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
// CHECK-NEXT:          knobs = {mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", [#iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>]>, reduction = [0, 0, #iree_codegen.smt.int_knob<"red_2">], subgroup_basis = {counts = [#iree_codegen.smt.int_knob<"sg_m_cnt">, #iree_codegen.smt.int_knob<"sg_n_cnt">, 1], mapping = [0, 1, 2]}, subgroup_size = #iree_codegen.smt.int_knob<"sg_size">, workgroup = [#iree_codegen.smt.int_knob<"wg_0">, #iree_codegen.smt.int_knob<"wg_1">, 0], workgroup_size = [#iree_codegen.smt.int_knob<"wg_x">, #iree_codegen.smt.int_knob<"wg_y">, #iree_codegen.smt.int_knob<"wg_z">]}
// CHECK:           ^bb0(%[[M:.+]]: !smt.int, %[[N:.+]]: !smt.int, %[[K:.+]]: !smt.int):
//
// Common: static dims.
// CHECK:             iree_codegen.smt.assert {{.*}}, "dim_0 ({}) == 128", %[[M]]
// CHECK:             iree_codegen.smt.assert {{.*}}, "dim_1 ({}) == 256", %[[N]]
// CHECK:             iree_codegen.smt.assert {{.*}}, "dim_2 ({}) == 64", %[[K]]
//
// Divisibility: dim % tile == 0.
// CHECK:             iree_codegen.smt.assert {{.*}}, "dim_0 must be divisible by wg_0 ({} % {} == 0)"
// CHECK:             iree_codegen.smt.assert {{.*}}, "dim_1 must be divisible by wg_1 ({} % {} == 0)"
// CHECK:             iree_codegen.smt.assert {{.*}}, "dim_2 must be divisible by red_2 ({} % {} == 0)"

// -----

// Test: Target without TargetPipelineProviderAttrInterface produces no
// constraints (e.g., CPU targets that don't implement the interface).

#pipeline_layout_cpu = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#exec_target_cpu = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">

hal.executable @cpu_target_ex {
  hal.executable.variant public @cpu target(#exec_target_cpu) {
    hal.executable.export public @cpu_target ordinal(0) layout(#pipeline_layout_cpu)
    builtin.module {
      func.func @cpu_target() {
        %cst = arith.constant 0.0 : f32
        %empty = tensor.empty() : tensor<64x64xf32>
        %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
            ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
        return
      }
    }
  }
}

// CPU target has no pipeline provider; no constraints emitted.
// CHECK-LABEL: hal.executable public @cpu_target_ex
// CHECK-NOT:     iree_codegen.smt.constraints

// -----

// Test: SPIRV target skips LLVMGPU constraint generation.

#pipeline_layout_spirv = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#gpu_target_spirv = #iree_gpu.target<arch = "", features = "spirv:v1.6,cap:Shader", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle|arithmetic,
  subgroup_size_choices = [32],
  max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [65535, 65535, 65535]
>>
#exec_target_spirv = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
    iree.spirv.features = ["vulkan-spirv"],
    iree_codegen.target_info = #gpu_target_spirv}>

hal.executable @spirv_target_ex {
  hal.executable.variant public @vulkan target(#exec_target_spirv) {
    hal.executable.export public @spirv_matmul ordinal(0) layout(#pipeline_layout_spirv)
    builtin.module {
      func.func @spirv_matmul() {
        %cst = arith.constant 0.0 : f32
        %lhs = tensor.empty() : tensor<64x32xf32>
        %rhs = tensor.empty() : tensor<32x64xf32>
        %empty = tensor.empty() : tensor<64x64xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
        %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
            ins(%lhs, %rhs : tensor<64x32xf32>, tensor<32x64xf32>)
            outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
        return
      }
    }
  }
}

// SPIRV target: LLVMGPU constraints are skipped.
// CHECK-LABEL: hal.executable public @spirv_target_ex
// CHECK:         linalg.matmul
// CHECK-NOT:     iree_codegen.smt.constraints

// -----

// Test: Non-linalg root ops are silently skipped (no crash).

#pipeline_layout3 = #hal.pipeline.layout<bindings = [
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

hal.executable @non_linalg_root_ex {
  hal.executable.variant public @rocm target(#exec_target3) {
    hal.executable.export public @non_linalg_root ordinal(0) layout(#pipeline_layout3)
    builtin.module {
      func.func @non_linalg_root() {
        %cst = arith.constant {root_op = #iree_codegen.root_op<set = 0>} 0.0 : f32
        return
      }
    }
  }
}

// Non-linalg ops are skipped; no constraints emitted.
// CHECK-LABEL: hal.executable public @non_linalg_root_ex
// CHECK-NOT:     iree_codegen.smt.constraints
