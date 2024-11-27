// RUN: iree-opt %s --split-input-file \
// RUN:     --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-gpu-propagate-dispatch-size-bounds)))))" \
// RUN:  | FileCheck %s

// Note: not the real target definition, missing types
#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx1100", features = "",
  wgp = <compute =  fp32,
    storage =  b32,
    subgroup =  arithmetic,
    dot =  none, mma = [],
    subgroup_size_choices = [32, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @static {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target) {
    hal.executable.export public @static ordinal(0) layout(#pipeline_layout) attributes {workgroup_size = [64 : index, 2 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      hal.return %c32, %c8, %c1 : index, index, index
    }
    builtin.module {
// CHECK-LABEL: func.func @static
      func.func @static() {
// CHECK: gpu.thread_id x upper_bound 64
// CHECK: gpu.thread_id y upper_bound 2
// CHECK: gpu.thread_id z upper_bound 1
        %thread_id_x = gpu.thread_id x
        %thread_id_y = gpu.thread_id y
        %thread_id_z = gpu.thread_id z

// CHECK: hal.interface.workgroup.size[0] upper_bound 64
// CHECK: hal.interface.workgroup.size[1] upper_bound 2
// CHECK: hal.interface.workgroup.size[2] upper_bound 1
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index

// CHECK: hal.interface.workgroup.id[0] upper_bound 32
// CHECK: hal.interface.workgroup.id[1] upper_bound 8
// CHECK: hal.interface.workgroup.id[2] upper_bound 1
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index

// CHECK: hal.interface.workgroup.count[0] upper_bound 32
// CHECK: hal.interface.workgroup.count[1] upper_bound 8
// CHECK: hal.interface.workgroup.count[2] upper_bound 1
        %workgroup_conut_x = hal.interface.workgroup.count[0] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index

        return
      }
    }
  }
}

// -----

#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree.gpu.target = #iree_gpu.target<arch = "gfx1100", features = "",
  wgp = <compute =  fp32,
    storage =  b32,
    subgroup = arithmetic,
    dot =  none, mma = [],
    subgroup_size_choices = [32, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>

hal.executable private @dynamic {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target) {
    hal.executable.export public @dynamic ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %count_x = affine.apply affine_map<()[s0] -> (s0 ceildiv 32)>()[%arg1]
      %count_y = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)>()[%arg2]
      %count_z = arith.constant 1 : index
      hal.return %count_x, %count_y, %count_z : index, index, index
    }
    builtin.module {
      func.func @dynamic() {
// CHECK: gpu.thread_id x upper_bound 1024
// CHECK: gpu.thread_id y upper_bound 1024
// CHECK: gpu.thread_id z upper_bound 1024
        %thread_id_x = gpu.thread_id x
        %thread_id_y = gpu.thread_id y
        %thread_id_z = gpu.thread_id z

// CHECK: hal.interface.workgroup.size[0] upper_bound 1024
// CHECK: hal.interface.workgroup.size[1] upper_bound 1024
// CHECK: hal.interface.workgroup.size[2] upper_bound 1024
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index

// CHECK: hal.interface.workgroup.id[0] upper_bound 2147483647
// CHECK: hal.interface.workgroup.id[1] upper_bound 2147483647
// CHECK: hal.interface.workgroup.id[2] upper_bound 1
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index

// CHECK: hal.interface.workgroup.count[0] upper_bound 2147483647
// CHECK: hal.interface.workgroup.count[1] upper_bound 2147483647
// CHECK: hal.interface.workgroup.count[2] upper_bound 1
        %workgroup_conut_x = hal.interface.workgroup.count[0] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index

        return
      }
    }
  }
}
