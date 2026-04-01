// RUN: iree-opt %s --split-input-file \
// RUN:     --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-propagate-dispatch-size-bounds)))))" \
// RUN:  | FileCheck %s
// RUN: iree-opt %s --split-input-file \
// RUN:     --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-propagate-dispatch-size-bounds{use-dispatch-config=true})))))" \
// RUN:  | FileCheck %s --check-prefix=DISPATCH-CONFIG

// Note: not the real target definition, missing types
#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "gfx1100", features = "",
  wgp = <compute = fp32,
    storage = b32,
    subgroup = arithmetic,
    subgroup_size_choices = [32, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @static {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target) {
    hal.executable.export public @static ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      hal.return %c32, %c8, %c1 : index, index, index
    } attributes {workgroup_size = [64 : index, 2 : index, 1 : index]}
    builtin.module {
// CHECK-LABEL: func.func @static()
// CHECK-SAME: gpu.known_block_size = array<i32: 64, 2, 1>
      func.func @static() {
// CHECK-NEXT: gpu.lane_id upper_bound 64
        %lane_id = gpu.lane_id

// CHECK-NEXT: gpu.subgroup_id upper_bound 4 : index
        %subgroup_id = gpu.subgroup_id : index

// CHECK-NEXT: gpu.thread_id x upper_bound 64
// CHECK-NEXT: gpu.thread_id y upper_bound 2
// CHECK-NEXT: gpu.thread_id z upper_bound 1
        %thread_id_x = gpu.thread_id x
        %thread_id_y = gpu.thread_id y
        %thread_id_z = gpu.thread_id z

// CHECK-NEXT: arith.constant 64 : index
// CHECK-NEXT: arith.constant 2 : index
// CHECK-NEXT: arith.constant 1 : index
        %block_dim_x = gpu.block_dim x
        %block_dim_y = gpu.block_dim y
        %block_dim_z = gpu.block_dim z

// CHECK-NEXT: arith.constant 64 : index
// CHECK-NEXT: arith.constant 2 : index
// CHECK-NEXT: arith.constant 1 : index
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index

// CHECK-NEXT: hal.interface.workgroup.id[0] upper_bound 32
// CHECK-NEXT: hal.interface.workgroup.id[1] upper_bound 8
// CHECK-NEXT: hal.interface.workgroup.id[2] upper_bound 1
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index

// CHECK-NEXT: arith.constant 32 : index
// CHECK-NEXT: arith.constant 8 : index
// CHECK-NEXT: arith.constant 1 : index
        %workgroup_conut_x = hal.interface.workgroup.count[0] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index

        return
      }
    }
  }
}

// -----

// Note: not the real target definition, missing types
#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "gfx1100", features = "",
  wgp = <compute = fp32,
    storage = b32,
    subgroup = arithmetic,
    subgroup_size_choices = [32, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @manual_subgroup_size {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target) {
    hal.executable.export public @manual_subgroup_size ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      hal.return %c32, %c8, %c1 : index, index, index
    } attributes {subgroup_size = 32 : index}
    builtin.module {
// CHECK-LABEL: func.func @manual_subgroup_size()
      func.func @manual_subgroup_size() {
// CHECK-NEXT: gpu.lane_id upper_bound 32
        %lane_id = gpu.lane_id

// No workgroup_size; bound from max_thread_count_per_workgroup (1024) / subgroup_size (32) = 32.
// CHECK-NEXT: gpu.subgroup_id upper_bound 32 : index
        %subgroup_id = gpu.subgroup_id : index

// CHECK-NEXT: arith.constant 32 : index
        %subgroup_size = gpu.subgroup_size : index

        return
      }
    }
  }
}

// -----

// Test variable subgroup sizes on gfx1100 (subgroup_size_choices = [32, 64])
// with static workgroup sizes but no explicit subgroup_size selection.
#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx1100", features = "",
  wgp = <compute = fp32,
    storage = b32,
    subgroup = arithmetic,
    subgroup_size_choices = [32, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @gfx1100_variable_subgroup {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target) {
    hal.executable.export public @gfx1100_variable_subgroup ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      hal.return %c128, %c1, %c1 : index, index, index
    } attributes {workgroup_size = [128 : index, 1 : index, 1 : index]}
    builtin.module {
// CHECK-LABEL: func.func @gfx1100_variable_subgroup()
      func.func @gfx1100_variable_subgroup() {
// CHECK-NEXT: gpu.lane_id upper_bound 64
        %lane_id = gpu.lane_id

// CHECK-NEXT: gpu.subgroup_id upper_bound 4 : index
        %subgroup_id = gpu.subgroup_id : index

// CHECK-NEXT: gpu.subgroup_size upper_bound 64 : index
        %subgroup_size = gpu.subgroup_size : index

        return
      }
    }
  }
}

// -----

// Test pseudo-variable subgroup sizes on gfx942 (subgroup_size_choices = [64])
// with static workgroup sizes but no explicit subgroup_size selection in case
// that ever comes up.
#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "",
  wgp = <compute = fp32,
    storage = b32,
    subgroup = arithmetic,
    subgroup_size_choices = [64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @gfx942_not_really_variable_subgroup {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target) {
    hal.executable.export public @gfx942_not_really_variable_subgroup ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      hal.return %c128, %c1, %c1 : index, index, index
    } attributes {workgroup_size = [128 : index, 1 : index, 1 : index]}
    builtin.module {
// CHECK-LABEL: func.func @gfx942_not_really_variable_subgroup()
// CHECK-SAME: gpu.known_block_size = array<i32: 128, 1, 1>
      func.func @gfx942_not_really_variable_subgroup() {
// CHECK-NEXT: gpu.lane_id upper_bound 64
        %lane_id = gpu.lane_id

// CHECK-NEXT: gpu.subgroup_id upper_bound 2 : index
        %subgroup_id = gpu.subgroup_id : index

// CHECK-NEXT: arith.constant 64 : index
        %subgroup_size = gpu.subgroup_size : index

        return
      }
    }
  }
}

// -----

#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<arch = "gfx1100", features = "",
  wgp = <compute = fp32,
    storage = b32,
    subgroup = arithmetic,
    subgroup_size_choices = [32, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @dynamic {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target) {
    hal.executable.export public @dynamic ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      %count_x = affine.apply affine_map<()[s0] -> (s0 ceildiv 32)>()[%arg1]
      %count_y = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg2]
      %count_z = arith.constant 1 : index
      hal.return %count_x, %count_y, %count_z : index, index, index
    }
    builtin.module {
// CHECK-LABEL: func.func @dynamic()
      func.func @dynamic() {
// CHECK-NEXT: gpu.lane_id upper_bound 64
        %lane_id = gpu.lane_id

// CHECK-NEXT: gpu.subgroup_id upper_bound 32 : index
        %subgroup_id = gpu.subgroup_id : index

// CHECK-NEXT: gpu.subgroup_size upper_bound 64 : index
        %subgroup_size = gpu.subgroup_size : index

// CHECK-NEXT: gpu.thread_id x upper_bound 1024
// CHECK-NEXT: gpu.thread_id y upper_bound 1024
// CHECK-NEXT: gpu.thread_id z upper_bound 1024
        %thread_id_x = gpu.thread_id x
        %thread_id_y = gpu.thread_id y
        %thread_id_z = gpu.thread_id z

// CHECK-NEXT: gpu.block_dim x upper_bound 1024
// CHECK-NEXT: gpu.block_dim y upper_bound 1024
// CHECK-NEXT: gpu.block_dim z upper_bound 1024
        %block_dim_x = gpu.block_dim x
        %block_dim_y = gpu.block_dim y
        %block_dim_z = gpu.block_dim z

// CHECK-NEXT: hal.interface.workgroup.size[0] upper_bound 1024
// CHECK-NEXT: hal.interface.workgroup.size[1] upper_bound 1024
// CHECK-NEXT: hal.interface.workgroup.size[2] upper_bound 1024
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index

// CHECK-NEXT: hal.interface.workgroup.id[0] upper_bound 2147483647
// CHECK-NEXT: hal.interface.workgroup.id[1] upper_bound 2147483647
// CHECK-NEXT: hal.interface.workgroup.id[2] upper_bound 1
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index

// CHECK-NEXT: hal.interface.workgroup.count[0] upper_bound 2147483647
// CHECK-NEXT: hal.interface.workgroup.count[1] upper_bound 2147483647
// CHECK-NEXT: arith.constant 1 : index
        %workgroup_conut_x = hal.interface.workgroup.count[0] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index

        return
      }
    }
  }
}

// -----

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @static_cpu {
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target) {
    hal.executable.export public @static_cpu ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      hal.return %c32, %c8, %c1 : index, index, index
    } attributes {workgroup_size = [64 : index, 2 : index, 1 : index]}
    builtin.module {
// CHECK-LABEL: func.func @static_cpu()
      func.func @static_cpu() {
// CHECK-NEXT: hal.interface.workgroup.id[0] upper_bound 32
// CHECK-NEXT: hal.interface.workgroup.id[1] upper_bound 8
// CHECK-NEXT: hal.interface.workgroup.id[2] upper_bound 1
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index

// CHECK-NEXT: arith.constant 32 : index
// CHECK-NEXT: arith.constant 8 : index
// CHECK-NEXT: arith.constant 1 : index
        %workgroup_conut_x = hal.interface.workgroup.count[0] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index

        return
      }
    }
  }
}

// -----

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu_features = "+avx512f", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-linux-gnu"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @dynamic_cpu {
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target) {
    hal.executable.export public @dynamic_cpu ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      %count_x = affine.apply affine_map<()[s0] -> (s0 ceildiv 32)>()[%arg1]
      %count_y = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg2]
      %count_z = arith.constant 1 : index
      hal.return %count_x, %count_y, %count_z : index, index, index
    }
    builtin.module {
// CHECK-LABEL: func.func @dynamic_cpu()
      func.func @dynamic_cpu() {
// CHECK-NEXT: hal.interface.workgroup.id[0] : index
// CHECK-NEXT: hal.interface.workgroup.id[1] : index
// CHECK-NEXT: hal.interface.workgroup.id[2] upper_bound 1 : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index

// CHECK-NEXT: hal.interface.workgroup.count[0] : index
// CHECK-NEXT: hal.interface.workgroup.count[1] : index
// CHECK-NEXT: arith.constant 1 : index
        %workgroup_conut_x = hal.interface.workgroup.count[0] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index

        return
      }
    }
  }
}

// -----

// Test that use-dispatch-config reads bounds from dispatch_config instead of
// hal.executable.export.

// Note: not the real target definition, missing types
#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "gfx1100", features = "",
  wgp = <compute = fp32,
    storage = b32,
    subgroup = arithmetic,
    subgroup_size_choices = [32, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @dispatch_config_static {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target) {
    hal.executable.export public @dispatch_config_static ordinal(0) layout(#pipeline_layout)
    builtin.module {
// DISPATCH-CONFIG-LABEL: func.func @dispatch_config_static()
// DISPATCH-CONFIG-SAME: gpu.known_block_size = array<i32: 64, 2, 1>
      func.func @dispatch_config_static() {
// DISPATCH-CONFIG-NEXT: gpu.thread_id x upper_bound 64
// DISPATCH-CONFIG-NEXT: gpu.thread_id y upper_bound 2
// DISPATCH-CONFIG-NEXT: gpu.thread_id z upper_bound 1
        %thread_id_x = gpu.thread_id x
        %thread_id_y = gpu.thread_id y
        %thread_id_z = gpu.thread_id z

// DISPATCH-CONFIG-NEXT: hal.interface.workgroup.id[0] upper_bound 32
// DISPATCH-CONFIG-NEXT: hal.interface.workgroup.id[1] upper_bound 8
// DISPATCH-CONFIG-NEXT: hal.interface.workgroup.id[2] upper_bound 1
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index

        return
      }
      iree_codegen.dispatch_config @dispatch_config_static workgroup_size = [64, 2, 1] subgroup_size = 64 {
        %c32 = arith.constant 32 : index
        %c8 = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        iree_codegen.yield %c32, %c8, %c1 : index, index, index
      }
    }
  }
}
