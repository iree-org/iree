// RUN: iree-opt %s | FileCheck %s

// CHECK-LABEL: func.func @test_target_wgp()
func.func @test_target_wgp() attributes {
  // CHECK: #iree_gpu.target_wgp<
  // CHECK-SAME: compute =  fp32|fp16|int8,
  // CHECK-SAME: storage =  b32|b16,
  // CHECK-SAME: subgroup =  shuffle|arithmetic,
  // CHECK-SAME: dot =  dp4xi8toi32,
  // CHECK-SAME: mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>],
  // CHECK-SAME: subgroup_size_choices = [32, 64],
  // CHECK-SAME: max_workgroup_sizes = [1024, 1024, 1024],
  // CHECK-SAME: max_thread_count_per_workgroup = 1024,
  // CHECK-SAME: max_workgroup_memory_bytes = 65536>
  wgp = #iree_gpu.target_wgp<
    compute = fp16|fp32|int8, storage = b16|b32,
    subgroup = shuffle|arithmetic, dot = dp4xi8toi32,
    mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>],
    subgroup_size_choices = [32, 64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536
  >
} { return }


// CHECK-LABEL: func.func @test_target_wgp_none()
func.func @test_target_wgp_none() attributes {
  // CHECK: #iree_gpu.target_wgp<
  // CHECK-SAME: subgroup =  none,
  // CHECK-SAME: dot =  none,
  // CHECK-SAME: mma = [],
  wgp = #iree_gpu.target_wgp<
    compute = fp16|fp32|int8, storage = b16|b32,
    subgroup = none, dot = none,
    mma = [],
    subgroup_size_choices = [32],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536
  >
} { return }

// CHECK-LABEL: func.func @test_target_chip()
func.func @test_target_chip() attributes {
  // CHECK: #iree_gpu.target_chip<
  // CHECK-SAME: wgp_count = 304, chiplet_count = 8>
  chip = #iree_gpu.target_chip<
    wgp_count = 304,
    chiplet_count = 8
  >
} { return }

// CHECK-LABEL: func.func @test_target()
func.func @test_target() attributes {
  // CHECK: #iree_gpu.target<
  // CHECK-SAME: arch = "gfx942"
  // CHECK-SAME: features = "+sramecc,+xnack"
  // CHECK-SAME: wgp = <
  // CHECK-SAME: chip = <
  target = #iree_gpu.target<
    arch="gfx942",
    features="+sramecc,+xnack",
    wgp = <
      compute = fp16|fp32|int8, storage = b16|b32,
      subgroup = shuffle|arithmetic, dot = dp4xi8toi32,
      mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>],
      subgroup_size_choices = [32, 64],
      max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024,
      max_workgroup_memory_bytes = 65536>,
    chip = <wgp_count = 304, chiplet_count = 8>
  >
} { return }
