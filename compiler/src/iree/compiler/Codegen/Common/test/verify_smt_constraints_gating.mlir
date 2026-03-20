// RUN: iree-opt \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-verify-smt-constraints)))))' %s \
// RUN:   | FileCheck %s

// Test: Without --iree-codegen-experimental-verify-pipeline-constraints the pass erases all
// constraints ops unconditionally.

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

hal.executable @verify_gated_ex {
  hal.executable.variant public @rocm target(#exec_target) {
    hal.executable.export public @verify_gated ordinal(0) layout(#pipeline_layout)
    builtin.module {
      func.func @verify_gated() attributes {translation_info = #translation} {
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
            knobs = {}
            dims(%c128, %c256, %c64) {
        ^bb0(%m: !smt.int, %n: !smt.int, %k: !smt.int):
          %c128_smt = smt.int.constant 128
          %eq = smt.eq %m, %c128_smt : !smt.int
          iree_codegen.smt.assert %eq, "dim_0 == 128" : !smt.bool
        }
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable public @verify_gated_ex
// CHECK:         linalg.matmul
// CHECK-NOT:     iree_codegen.smt.constraints
