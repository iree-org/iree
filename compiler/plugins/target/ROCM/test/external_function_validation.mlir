// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(iree-hal-serialize-target-executables{target=rocm}))' \
// RUN:   --verify-diagnostics %s -o -

// The final bitcode validation should error out on any external functions that
// remain in the final bitcode (post device bitcode linking).

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "",
                                      wgp = <compute = fp16, storage =  b16,
                                      subgroup =  none, dot =  none, mma = [],
                                      subgroup_size_choices = [64],
                                      max_workgroup_sizes = [1024, 1024, 1024],
                                      max_thread_count_per_workgroup = 1024,
                                      max_workgroup_memory_bytes = 65536,
                                      max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>,
   ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, Indirect>],
                                        flags = Indirect>
builtin.module {
  hal.executable public @test {
    // expected-error @+2 {{found an unresolved external function 'external_func' in the final bitcode}}
    // expected-error @+1 {{failed to serialize executable for target backend rocm}}
    hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @test ordinal(0) layout(#pipeline_layout)
        attributes {subgroup_size = 64 : index, workgroup_size = [128 : index, 2 : index, 1 : index]} {
      ^bb0(%arg0: !hal.device):
        %c128 = arith.constant 128 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        hal.return %c128, %c2, %c1 : index, index, index
      }
      builtin.module {
        llvm.func @external_func() attributes {sym_visibility = "private"}
        llvm.func @test() attributes { rocdl.kernel } {
          llvm.call @external_func() : () -> ()
          llvm.return
        }
      }
    }
  }
}
