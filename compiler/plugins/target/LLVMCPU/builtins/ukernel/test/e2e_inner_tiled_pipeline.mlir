// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-configuration-preprocessing-pipeline, builtin.module(iree-codegen-llvmcpu-configuration-pipeline, iree-codegen-llvmcpu-lowering-pipeline), iree-codegen-translation-postprocessing-pipeline)))" %s \
// RUN: | FileCheck %s

// End-to-end-through-codegen test for the new C-bitcode ukernel framework.
// Drives the LLVMCPU configuration + lowering pipelines on an
// `iree_codegen.inner_tiled` rooted dispatch wrapped in the standard
// `hal.executable.variant` + binding-subspan structure (the same wrapper
// shape `iree-compile` produces after dispatch creation). Asserts the
// whole chain:
//
//   - SelectUKernels saw the `llvm_ukernels = "inner_tiled"` flag in the
//     target config and the matching MMA intrinsic, so it annotated the
//     `inner_tiled` and attached the matching bitcode to the parent
//     `hal.executable.variant`'s `objects(...)` (the long-lived home; not
//     a discardable attr that intermediate passes might strip).
//   - LowerBitcodeUKernels rewrote the `inner_tiled` into a
//     `ukernel.generic` with `fn_def_attrs = {hal.import.bitcode = true}`.
//   - LowerUKernelOpsToCalls turned that into a `func.call` against a
//     `func.func` declaration carrying `hal.import.bitcode`.
//   - Final LLVM lowering emits `llvm.func @iree_uk_mma_…` with
//     `hal.import.bitcode = true` and a direct `llvm.call` to it. The
//     `hal.import.bitcode` flag is what makes
//     `RewriteExternCallOpToDynamicImportCallOp` in `ConvertToLLVM` skip
//     the runtime-import-table indirection it applies to every other
//     external call — letting our call resolve directly against the
//     linked bitcode at LLVM optimization time.

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  cpu_features = "+avx512f,+avx512bf16",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  iree_codegen.ukernel_provider = #iree_cpu.ukernel_provider,
  llvm_ukernels = "inner_tiled",
  native_vector_size = 64 : index,
  target_triple = "x86_64-unknown-unknown-eabi-elf"
}>

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable private @bf16_inner_tiled_ukernel {
  hal.executable.variant public @variant target(#executable_target) {
    hal.executable.export public @bf16_inner_tiled_ukernel ordinal(0) layout(#pipeline_layout)
        count(%device: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @bf16_inner_tiled_ukernel() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4x1x2xbf16>>
        %rhs = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
            alignment(64) offset(%c0) flags(ReadOnly)
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4x16x2xbf16>>
        %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
            alignment(64) offset(%c0)
            : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x2x1x16xf32>>
        %lhs_t = iree_tensor_ext.dispatch.tensor.load %lhs,
            offsets = [0, 0, 0, 0], sizes = [2, 4, 1, 2], strides = [1, 1, 1, 1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4x1x2xbf16>> -> tensor<2x4x1x2xbf16>
        %rhs_t = iree_tensor_ext.dispatch.tensor.load %rhs,
            offsets = [0, 0, 0, 0], sizes = [2, 4, 16, 2], strides = [1, 1, 1, 1]
            : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4x16x2xbf16>> -> tensor<2x4x16x2xbf16>
        %acc_t = iree_tensor_ext.dispatch.tensor.load %out,
            offsets = [0, 0, 0, 0], sizes = [2, 2, 1, 16], strides = [1, 1, 1, 1]
            : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x2x1x16xf32>> -> tensor<2x2x1x16xf32>
        %res = iree_codegen.inner_tiled ins(%lhs_t, %rhs_t) outs(%acc_t) {
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
        iree_tensor_ext.dispatch.tensor.store %res, %out,
            offsets = [0, 0, 0, 0], sizes = [2, 2, 1, 16], strides = [1, 1, 1, 1]
            : tensor<2x2x1x16xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x2x1x16xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable private @bf16_inner_tiled_ukernel
//
// The bitcode for the matched ukernel is attached as `objects(...)` on
// the variant — the self-contained-IR property of the new framework.
// CHECK:         hal.executable.variant public @variant
// CHECK-SAME:      objects([
// CHECK-SAME:        path = "iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16.x86_64_avx512bf16.bc"
//
// The ukernel function is declared as `llvm.func` carrying
// `hal.import.bitcode = true` — the flag that tells `ConvertToLLVM` to
// emit a direct call instead of routing through the runtime import table.
// CHECK:         llvm.func @iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16
// CHECK-SAME:      attributes {hal.import.bitcode = true
//
// And the dispatch function emits a direct `llvm.call` to that symbol.
// No `inner_tiled`, no `ukernel.generic`, and no `__import_ordinal_`
// indirection (which would mean we accidentally re-used the legacy
// runtime-resolved-import path).
// CHECK:         llvm.call @iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16
// CHECK-NOT:     iree_codegen.inner_tiled
// CHECK-NOT:     iree_codegen.ukernel.generic
// CHECK-NOT:     __import_ordinal_iree_uk_mma
