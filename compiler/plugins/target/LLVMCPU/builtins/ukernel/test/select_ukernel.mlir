// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy,func.func(iree-codegen-lower-bitcode-ukernels))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=CHAIN

// Drives the LLVMCPU `LLVMCPUSelectLoweringStrategy` pass on
// `inner_tiled` rooted dispatches carrying a CPU `DataTiledMMAAttr`. The
// new-style C-bitcode ukernel framework is selected when the target
// config has `llvm_ukernels = "inner_tiled"` (set by the
// `--iree-llvmcpu-enable-llvm-ukernels=inner_tiled` CL flag at
// `iree-compile` time; we set it directly in the test since `iree-opt`
// doesn't run target-option serialization).
//
// Two cases:
//   (1) enabled:  `selectUKernel` matches the BF16 1x16x2 intrinsic and
//                 annotates the op with `iree_codegen.ukernel = …`.
//   (2) disabled: target config lacks `llvm_ukernels`, so `selectUKernel`
//                 returns null and no descriptor is added — the default,
//                 keeping the new framework off in plain builds.

// The `iree_codegen.ukernel_provider = #iree_cpu.ukernel_provider` here is
// what `LLVMCPUTarget.cpp` adds automatically whenever `llvm_ukernels` is
// non-empty; we set it directly because `iree-opt` doesn't run target-option
// serialization.
#executable_target_enabled = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  cpu_features = "+avx512f,+avx512bf16",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  iree_codegen.ukernel_provider = #iree_cpu.ukernel_provider,
  llvm_ukernels = "inner_tiled",
  native_vector_size = 64 : index,
  target_triple = "x86_64-unknown-unknown-eabi-elf"
}>

func.func @bf16_inner_tiled_ukernel_enabled(
    %lhs: tensor<2x4x1x2xbf16>, %rhs: tensor<2x4x16x2xbf16>, %acc: tensor<2x2x1x16xf32>
  ) -> tensor<2x2x1x16xf32> attributes {hal.executable.target = #executable_target_enabled} {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512BF16_1x16x2_F32_BF16>,
    semantics = #iree_cpu.mma_semantics<>
  } : tensor<2x4x1x2xbf16>, tensor<2x4x16x2xbf16> into tensor<2x2x1x16xf32>
  return %0 : tensor<2x2x1x16xf32>
}
// CHECK-LABEL: func.func @bf16_inner_tiled_ukernel_enabled
// CHECK:         iree_codegen.inner_tiled
// CHECK-SAME:      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16", bitcode>

// CHAIN-LABEL: func.func @bf16_inner_tiled_ukernel_enabled
// Once SelectLoweringStrategy has set the descriptor, the very next pass in
// the pipeline (`LowerBitcodeUKernelsPass`, also installed into the
// `Mmt4dTilingExpert` pipeline) is expected to rewrite the `inner_tiled`
// into a `ukernel.generic` and attach the matching bitcode resolved by the
// `#iree_cpu.ukernel_provider`. The CHAIN-NOT line guards against the
// rewrite silently skipping the op (the original `inner_tiled` must be gone
// from the rewritten function body).
// CHAIN:         iree_codegen.ukernel.generic
// CHAIN-SAME:      hal.executable.objects = [{{.*}}"iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16.x86_64_avx512bf16.bc"
// CHAIN-SAME:      "iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16"
// CHAIN-NOT:     iree_codegen.inner_tiled

// -----

#executable_target_disabled = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  cpu_features = "+avx512f,+avx512bf16",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 64 : index,
  target_triple = "x86_64-unknown-unknown-eabi-elf"
}>

func.func @bf16_inner_tiled_ukernel_disabled(
    %lhs: tensor<2x4x1x2xbf16>, %rhs: tensor<2x4x16x2xbf16>, %acc: tensor<2x2x1x16xf32>
  ) -> tensor<2x2x1x16xf32> attributes {hal.executable.target = #executable_target_disabled} {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512BF16_1x16x2_F32_BF16>,
    semantics = #iree_cpu.mma_semantics<>
  } : tensor<2x4x1x2xbf16>, tensor<2x4x16x2xbf16> into tensor<2x2x1x16xf32>
  return %0 : tensor<2x2x1x16xf32>
}
// CHECK-LABEL: func.func @bf16_inner_tiled_ukernel_disabled
// CHECK:         iree_codegen.inner_tiled
// CHECK-NOT:     iree_codegen.ukernel = #iree_codegen.ukernel_descriptor

// Same with the chained pipeline: SelectLoweringStrategy doesn't attach a
// descriptor, so LowerBitcodeUKernels has nothing to rewrite and the
// `inner_tiled` survives untouched.
// CHAIN-LABEL: func.func @bf16_inner_tiled_ukernel_disabled
// CHAIN:         iree_codegen.inner_tiled
// CHAIN-NOT:     iree_codegen.ukernel.generic
// CHAIN-NOT:     hal.executable.objects
