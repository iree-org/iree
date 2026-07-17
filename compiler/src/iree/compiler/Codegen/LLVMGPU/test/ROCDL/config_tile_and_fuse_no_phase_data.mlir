// Negative test: targets without phase data should not get XOR swizzles.
//
// This constructs a synthetic target with gfx950's architecture (so DMA
// hardware support checks pass) but overrides shared_mem_model to cdna3,
// which has no empirical phase group data. getPhaseGroups returns nullopt
// for SharedMemoryModel::CDNA3, causing hasNoBankConflicts to return failure,
// and thus getXorShuffleParams returns nullopt — no XOR swizzle is applied.

// RUN: iree-opt --mlir-print-local-scope --split-input-file \
// RUN:   --iree-codegen-llvmgpu-use-tile-and-fuse-matmul=true \
// RUN:   --iree-codegen-llvmgpu-test-tile-and-fuse-vectorize=true \
// RUN:   --iree-codegen-llvmgpu-use-igemm=false \
// RUN:   --iree-llvmgpu-use-direct-load=true \
// RUN:   --pass-pipeline="builtin.module(iree-llvmgpu-select-lowering-strategy)" %s \
// RUN:   | FileCheck %s

// Synthetic gfx950 target with SharedMemoryModel::CDNA3 (no phase data).
#gpu_target_no_phases = #iree_gpu.target<arch = "gfx950", features = "", wgp = <
  compute = fp64|fp32|fp16|int64|int32|int16|int8,
  storage = b64|b32|b16|b8,
  subgroup = shuffle|arithmetic, dot = dp4xi8toi32,
  mma = [<MFMA_F32_16x16x16_F16>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 163840,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647],
  simds_per_wgp = 4,
  vgpr_space_bits = 16384,
  dma_sizes = [32, 128],
  shared_mem_model = cdna3
>>
#exec_target_no_phases = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target_no_phases}>

// The DMA path is entered (arch is gfx950), but getXorShuffleParams fails
// because getPhaseGroups(CDNA3, ...) returns nullopt. Verify no xor_shuffle.
//
// CHECK-LABEL: func.func @matmul_f16_no_swizzle_cdna_model
// CHECK:         promotion_types =
// CHECK-NOT:     xor_shuffle
// CHECK:         return
func.func @matmul_f16_no_swizzle_cdna_model(
    %arg0: tensor<4096x4096xf16>, %arg1: tensor<4096x4096xf16>,
    %arg2: tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    attributes {hal.executable.target = #exec_target_no_phases} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<4096x4096xf16>, tensor<4096x4096xf16>)
      outs(%arg2 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
  return %0 : tensor<4096x4096xf32>
}
