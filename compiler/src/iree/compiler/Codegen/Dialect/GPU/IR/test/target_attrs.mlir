// RUN: iree-opt %s | FileCheck %s

// CHECK-LABEL: func.func @test_target_core()
func.func @test_target_core() attributes {
  // CHECK: #iree_gpu.target_core<
  // CHECK-SAME: compute =  fp32|fp16|int8,
  // CHECK-SAME: storage =  b32|b16,
  // CHECK-SAME: subgroup =  shuffle|arithmetic,
  // CHECK-SAME: dot =  dp4xi8toi32,
  // CHECK-SAME: mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>],
  // CHECK-SAME: subgroup_size_choices = [32, 64],
  // CHECK-SAME: max_workgroup_sizes = [1024, 1024, 1024],
  // CHECK-SAME: max_thread_size = 1024,
  // CHECK-SAME: max_workgroup_memory_bytes = 65536>
  core = #iree_gpu.target_core<
    compute = fp16|fp32|int8, storage = b16|b32,
    subgroup = shuffle|arithmetic, dot = dp4xi8toi32,
    mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>],
    subgroup_size_choices = [32, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_size = 1024,
    max_workgroup_memory_bytes = 65536
  >
} { return }


// CHECK-LABEL: func.func @test_target_core_none()
func.func @test_target_core_none() attributes {
  // CHECK: #iree_gpu.target_core<
  // CHECK-SAME: subgroup =  none,
  // CHECK-SAME: dot =  none,
  // CHECK-SAME: mma = [],
  core = #iree_gpu.target_core<
    compute = fp16|fp32|int8, storage = b16|b32,
    subgroup = none, dot = none,
    mma = [],
    subgroup_size_choices = [32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_size = 1024,
    max_workgroup_memory_bytes = 65536
  >
} { return }

// CHECK-LABEL: func.func @test_target_chip()
func.func @test_target_chip() attributes {
  // CHECK: #iree_gpu.target_chip<
  // CHECK-SAME: core_count = 304>
  chip = #iree_gpu.target_chip<
    core_count = 304
  >
} { return }

// CHECK-LABEL: func.func @test_target()
func.func @test_target() attributes {
  // CHECK: #iree_gpu.target<
  // CHECK-SAME: core = <
  // CHECK-SAME: chip = <
  target = #iree_gpu.target<
    arch="gfx942",
    core = <
      compute = fp16|fp32|int8, storage = b16|b32,
      subgroup = shuffle|arithmetic, dot = dp4xi8toi32,
      mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>],
      subgroup_size_choices = [32, 64],
      max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_size = 1024,
      max_workgroup_memory_bytes = 65536>,
    chip = <core_count = 304>
  >
} { return }

// CHECK-LABEL: func.func @test_alias_target_cuda()
func.func @test_alias_target_cuda() attributes {
  // CHECK: #iree_gpu.alias_target<"sm_80">
  target = #iree_gpu.alias_target<"sm_80">
} { return }

// CHECK-LABEL: func.func @test_alias_target_hip()
func.func @test_alias_target_hip() attributes {
  // CHECK: #iree_gpu.alias_target<"mi300x">
  target = #iree_gpu.alias_target<"mi300x">
} { return }
