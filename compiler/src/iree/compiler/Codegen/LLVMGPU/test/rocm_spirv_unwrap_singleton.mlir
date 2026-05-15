// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(iree-hal-serialize-target-executables{target=rocm dump-intermediates-path=%t}))' \
// RUN: --iree-rocm-target=gfx1201 %s
// RUN: FileCheck %s < %t/module_rocm_spirv_unwrap_test_rocm_spirv_fb.unwrapped.ll

// Two single-element array values flowing through the same control-flow merge
// should become two PHI nodes after LLVM export. 

#pipeline_layout = #hal.pipeline.layout<bindings = []>
#target = #iree_gpu.target<arch = "gfx1201", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = none,
  subgroup_size_choices = [64],
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>

hal.executable public @rocm_spirv_unwrap_test {
  hal.executable.variant public @rocm_spirv_fb target(<"rocm", "rocm-spirv-fb", {
    iree_codegen.target_info = #target,
    ukernels = "none"
  }>) {
    hal.executable.export public @empty ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    } attributes {subgroup_size = 64 : index, workgroup_size = [1 : index, 1 : index, 1 : index]}
    builtin.module attributes {
      llvm.data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n32:64-S32-G1-P4-A0",
      llvm.target_triple = "spirv64-amd-amdhsa"
    } {
      llvm.func spir_kernelcc @empty() attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
        llvm.return
      }

      llvm.func spir_funccc @two_singletons_through_cond(
          %cond: i1,
          %inner_a: vector<16xf32>,
          %inner_b: vector<16xf32>,
          %inner_c: vector<16xf32>,
          %inner_d: vector<16xf32>) -> vector<16xf32> {
        // CHECK-LABEL: @two_singletons
        // CHECK-NOT: phi [1 x {{.*}}]
        %init_a = llvm.mlir.poison : !llvm.array<1 x vector<16xf32>>
        %init_b = llvm.mlir.poison : !llvm.array<1 x vector<16xf32>>
        %initial_a = llvm.insertvalue %inner_a, %init_a[0] : !llvm.array<1 x vector<16xf32>>
        %initial_b = llvm.insertvalue %inner_b, %init_b[0] : !llvm.array<1 x vector<16xf32>>

        llvm.cond_br %cond, ^then, ^else

      ^then:
        %then_a = llvm.insertvalue %inner_c, %init_a[0] : !llvm.array<1 x vector<16xf32>>
        %then_b = llvm.insertvalue %inner_d, %init_b[0] : !llvm.array<1 x vector<16xf32>>
        llvm.br ^join(%then_a, %then_b : !llvm.array<1 x vector<16xf32>>, !llvm.array<1 x vector<16xf32>>)

      ^else:
        llvm.br ^join(%initial_a, %initial_b : !llvm.array<1 x vector<16xf32>>, !llvm.array<1 x vector<16xf32>>)

      ^join(%joined_a: !llvm.array<1 x vector<16xf32>>, %joined_b: !llvm.array<1 x vector<16xf32>>):
        %result_a = llvm.extractvalue %joined_a[0] : !llvm.array<1 x vector<16xf32>>
        %result_b = llvm.extractvalue %joined_b[0] : !llvm.array<1 x vector<16xf32>>
        %result = llvm.fadd %result_a, %result_b : vector<16xf32>
        llvm.return %result : vector<16xf32>
      }
    }
  }
}
