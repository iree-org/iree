// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-insert-smt-constraints)))))' %s \
// RUN:   | FileCheck %s

// Test: Without --iree-codegen-add-tuner-attributes the pass is a no-op.

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

hal.executable @matmul_gated_ex {
  hal.executable.variant public @rocm target(#exec_target) {
    hal.executable.export public @matmul_gated ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_gated() {
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

// CHECK-LABEL: hal.executable public @matmul_gated_ex
// CHECK:         linalg.matmul
// CHECK-NOT:     iree_codegen.smt.constraints
