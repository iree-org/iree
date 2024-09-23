// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(llvm.func(iree-rocdl-annotate-kernel-for-translation)))))' \
// RUN:   --split-input-file %s | FileCheck %s

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "",
                                      wgp = <compute = int32, storage =  b32,
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
        llvm.func @test() {
          llvm.return
        }
        llvm.func @test_not_exported() {
          llvm.return
        }
      }
    }
  }
}

// CHECK-LABEL: llvm.func @test() attributes {
// CHECK-SAME:    rocdl.flat_work_group_size = "256,256"
// CHECK-SAME:    rocdl.kernel
// CHECK-SAME:    rocdl.reqd_work_group_size = array<i32: 128, 2, 1>
//
// CHECK-LABEL: llvm.func @test_not_exported() {

// -----

// Check that we annotate kernel arguments on gfx940-series.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree.gpu.target = #iree_gpu.target<arch = "gfx940", features = "",
                                      wgp = <compute = int32, storage =  b32,
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
  hal.executable public @test_kern_arg {
    hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @test_kern_arg ordinal(0) layout(#pipeline_layout)
        attributes {subgroup_size = 64 : index, workgroup_size = [128 : index, 2 : index, 1 : index]} {
      ^bb0(%arg0: !hal.device):
        %c128 = arith.constant 128 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        hal.return %c128, %c2, %c1 : index, index, index
      }
      builtin.module {
        llvm.func @test_kern_arg(%arg0: i32) {
          llvm.return
        }
      }
    }
  }
}

// CHECK-LABEL: llvm.func @test_kern_arg
// CHECK-SAME:    (%{{.+}}: i32 {llvm.inreg})

// -----

// Check that we *do not* annotate kernel arguments on gfx90a (not supported by the firmware).

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree.gpu.target = #iree_gpu.target<arch = "gfx90a", features = "",
                                      wgp = <compute = int32, storage =  b32,
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
  hal.executable public @test_no_kern_arg {
    hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @test_no_kern_arg ordinal(0) layout(#pipeline_layout)
        attributes {subgroup_size = 64 : index, workgroup_size = [128 : index, 2 : index, 1 : index]} {
      ^bb0(%arg0: !hal.device):
        %c128 = arith.constant 128 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        hal.return %c128, %c2, %c1 : index, index, index
      }
      builtin.module {
        llvm.func @test_no_kern_arg(%arg0: i32) {
          llvm.return
        }
      }
    }
  }
}

// CHECK-LABEL: llvm.func @test_no_kern_arg
// CHECK-SAME:    (%{{.+}}: i32)
