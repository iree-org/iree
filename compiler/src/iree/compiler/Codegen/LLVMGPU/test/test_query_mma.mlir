// RUN: iree-opt --split-input-file --iree-test-llvmgpu-query-mma %s | FileCheck %s

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
{iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "",
wgp = <compute = int32, storage =  b32,
subgroup = arithmetic, dot = dp4xi8toi32,
mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>],
subgroup_size_choices = [64], max_workgroup_sizes = [1024],
max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
max_workgroup_counts = [2147483647]>>}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
module {
  hal.executable private @main {
    hal.executable.variant public @main target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @entry_point layout(#pipeline_layout)
      builtin.module {
        func.func @fn() {
          return
        }
      }
    }
  }
}

// CHECK:       Executable Variant Name
// CHECK-SAME:  main
// CHECK: MMA   Intrinsics
// CHECK-SAME:  MFMA_F32_16x16x4_F32
// CHECK-SAME:  MFMA_F32_16x16x16_F16
// CHECK-LABEL: func.func @fn

// -----

#executable_target_rocm_hsaco_fb0 = #hal.executable.target<"rocm", "rocm-hsaco-fb",
{iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "",
wgp = <compute = int32, storage =  b32,
subgroup = arithmetic, dot = dp4xi8toi32,
mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>],
subgroup_size_choices = [64], max_workgroup_sizes = [1024],
max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
max_workgroup_counts = [2147483647]>>}>
#executable_target_rocm_hsaco_fb1 = #hal.executable.target<"rocm", "rocm-hsaco-fb",
{iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "",
wgp = <compute = int32, storage =  b32,
subgroup = arithmetic, dot = dp4xi8toi32,
mma = [<MFMA_F32_32x32x8_F16>, <MFMA_F32_16x16x16_BF16>],
subgroup_size_choices = [64], max_workgroup_sizes = [1024],
max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
max_workgroup_counts = [2147483647]>>}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
module {
  hal.executable private @main_0 {
    hal.executable.variant public @main_0 target(#executable_target_rocm_hsaco_fb0) {
      hal.executable.export public @entry_point_0 layout(#pipeline_layout)
      builtin.module {
        func.func @fn_0() {
          return
        }
      }
    }
  }
  hal.executable private @main_1 {
    hal.executable.variant public @main_1 target(#executable_target_rocm_hsaco_fb1) {
      hal.executable.export public @entry_point layout(#pipeline_layout)
      builtin.module {
        func.func @fn_1() {
          return
        }
      }
    }
  }
}

// CHECK-DAG: main_0
// CHECK-DAG: MMA Intrinsics: MFMA_F32_16x16x4_F32 MFMA_F32_16x16x16_F16
// CHECK-DAG: main_1
// CHECK-DAG: MMA Intrinsics: MFMA_F32_32x32x8_F16 MFMA_F32_16x16x16_BF16

// -----

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
module {
  hal.executable private @main {
    hal.executable.variant public @main target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @entry_point layout(#pipeline_layout)
      builtin.module {
        func.func @fn_empty() {
          return
        }
      }
    }
  }
}

// CHECK-NOT:   Executable Variant Name
// CHECK-NOT:   MMA Intrinsics
// CHECK-LABEL: func.func @fn
