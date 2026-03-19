// RUN: iree-opt --split-input-file \
// RUN:   --iree-codegen-add-tuner-attributes \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-insert-smt-constraints)))))' %s \
// RUN:   | FileCheck %s

// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-insert-smt-constraints)))))' %s \
// RUN:   | FileCheck %s --check-prefix=NO_TUNER_ATTRS

// -----

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
        %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
            ins(%cst : f32) outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
        %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
            ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
            outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
        return
      }
    }
  }
}

// The fill is in the same set but skipped; only the matmul gets constraints.
// CHECK-LABEL: hal.executable public @matmul_f32_ex
// CHECK:         func.func @matmul_f32
// CHECK:           linalg.fill
// CHECK-NOT:       iree_codegen.smt.constraints
// CHECK:           linalg.matmul
//
// CHECK:           iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
// CHECK-NEXT:          knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_0">, #iree_codegen.smt.int_knob<"wg_1">, #iree_codegen.smt.int_knob<"wg_2">]}
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
// CHECK:             iree_codegen.smt.assert {{.*}}, "dim_2 must be divisible by wg_2 ({} % {} == 0)"

// No tuner attrs flag: pass is gated.
// NO_TUNER_ATTRS-LABEL: hal.executable public @matmul_f32_ex
// NO_TUNER_ATTRS:         linalg.matmul
// NO_TUNER_ATTRS-NOT:     iree_codegen.smt.constraints

// -----

// Test: Fill-only dispatch still gets constraints (fill is the sole root).

#pipeline_layout2 = #hal.pipeline.layout<bindings = [
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

hal.executable @fill_only_ex {
  hal.executable.variant public @rocm target(#exec_target2) {
    hal.executable.export public @fill_only ordinal(0) layout(#pipeline_layout2)
    builtin.module {
      func.func @fill_only() {
        %cst = arith.constant 0.0 : f32
        %empty = tensor.empty() : tensor<64x64xf32>
        %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
            ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
        return
      }
    }
  }
}

// Fill is the only root, so it gets constraints (2 dims: M, N).
// CHECK-LABEL: hal.executable public @fill_only_ex
// CHECK:         linalg.fill
// CHECK:         iree_codegen.smt.constraints target = <set = 0>
// CHECK:         ^bb0(%{{.+}}: !smt.int, %{{.+}}: !smt.int):

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
