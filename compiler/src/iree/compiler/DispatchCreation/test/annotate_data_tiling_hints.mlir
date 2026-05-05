// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-annotate-data-tiling-hints))" --split-input-file %s | FileCheck %s --check-prefixes=CHECK-DEFAULT,CHECK
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-annotate-data-tiling-hints{data-tiling-op-types=matmul,scaled_matmul,convolution}))" --split-input-file %s | FileCheck %s --check-prefixes=CHECK-ALL,CHECK
// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-annotate-data-tiling-hints{data-tiling-op-types=convolution}))" --split-input-file %s | FileCheck %s --check-prefixes=CHECK-CONV-ONLY

util.func public @matmul(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
         ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
         outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: @matmul(
// CHECK:         linalg.matmul
// CHECK-SAME:      iree.opt.data_tiling

// -----

util.func public @matmul_with_preset_hints(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = linalg.matmul {"iree.opt.data_tiling"}
         ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
         outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul
         ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
         outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}
// CHECK-LABEL: @matmul_with_preset_hints(
// CHECK:         linalg.matmul
// CHECK-SAME:      iree.opt.data_tiling
// CHECK-NOT:       iree.opt.data_tiling

// -----

// Conv cases: annotated only when convolution is explicitly enabled.
util.func public @conv_2d_nhwc_hwcf(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x?x?xf32>, %arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
         ins(%arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
         outs(%arg2 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  util.return %0 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: @conv_2d_nhwc_hwcf(
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-DEFAULT-NOT:  iree.opt.data_tiling
// CHECK-ALL-SAME:     iree.opt.data_tiling
// CHECK-CONV-ONLY:    linalg.conv_2d_nhwc_hwcf
// CHECK-CONV-ONLY-SAME: iree.opt.data_tiling

// -----

util.func public @conv_1d_ncw_fcw(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>, %arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.conv_1d_ncw_fcw
         ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
         outs(%arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  util.return %0 : tensor<?x?x?xf32>
}
// CHECK-LABEL: @conv_1d_ncw_fcw(
// CHECK:         linalg.conv_1d_ncw_fcw
// CHECK-DEFAULT-NOT:  iree.opt.data_tiling
// CHECK-ALL-SAME:     iree.opt.data_tiling

// -----

util.func public @conv_3d_ndhwc_dhwcf(%arg0 : tensor<?x?x?x?x?xf32>, %arg1 : tensor<?x?x?x?x?xf32>, %arg2 : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = linalg.conv_3d_ndhwc_dhwcf
         ins(%arg0, %arg1 : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
         outs(%arg2 : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  util.return %0 : tensor<?x?x?x?x?xf32>
}
// CHECK-LABEL: @conv_3d_ndhwc_dhwcf(
// CHECK:         linalg.conv_3d_ndhwc_dhwcf
// CHECK-DEFAULT-NOT:  iree.opt.data_tiling
// CHECK-ALL-SAME:     iree.opt.data_tiling

// -----

// Mixed function: matmul gets the hint by default, conv only with convolution enabled.
util.func public @matmul_and_conv(
    %a : tensor<?x?xf32>, %b : tensor<?x?xf32>, %c : tensor<?x?xf32>,
    %x : tensor<?x?x?x?xf32>, %y : tensor<?x?x?x?xf32>, %z : tensor<?x?x?x?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?x?x?xf32>) {
  %0 = linalg.matmul
         ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>)
         outs(%c : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.conv_2d_nhwc_hwcf
         ins(%x, %y : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
         outs(%z : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  util.return %0, %1 : tensor<?x?xf32>, tensor<?x?x?x?xf32>
}
// CHECK-LABEL: @matmul_and_conv(
// CHECK:         linalg.matmul
// CHECK-SAME:      iree.opt.data_tiling
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-DEFAULT-NOT:  iree.opt.data_tiling
// CHECK-ALL-SAME:     iree.opt.data_tiling

// -----

// `enable_inner_tiled` opt-out: when the module's CPU target advertises
// `enable_inner_tiled` but has no MMA intrinsic matching the matmul's element
// types, we drop the data-tiling hint instead of letting SetEncoding produce
// an `iree_codegen.inner_tiled` no later pass can lower. AVX2 (`+avx2,+fma`)
// has no bf16/f32 MMA, so this matmul stays as a plain `linalg.matmul`.

util.global private @__device_avx2 = #hal.device.target<"local", [
  #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
    target_triple = "x86_64-xyz-xyz",
    cpu_features = "+avx2,+fma",
    enable_inner_tiled = true,
    iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>
  }>
]> : !hal.device

util.func public @matmul_bf16_f32_avx2(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?xbf16>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
         ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
         outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: @matmul_bf16_f32_avx2(
// CHECK:         linalg.matmul
// CHECK-NOT:       iree.opt.data_tiling
// CHECK:         util.return

// -----

// `enable_inner_tiled` AVX2 target: f32/f32 matmul *does* match the AVX2
// 1x8x1 FMA intrinsic, so the hint is still added. Companion to the
// bf16/f32 case above — proves the opt-out is per-element-type, not a
// blanket disable when `enable_inner_tiled` is on.

util.global private @__device_avx2_f32 = #hal.device.target<"local", [
  #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
    target_triple = "x86_64-xyz-xyz",
    cpu_features = "+avx2,+fma",
    enable_inner_tiled = true,
    iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>
  }>
]> : !hal.device

util.func public @matmul_f32_f32_avx2(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
         ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
         outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: @matmul_f32_f32_avx2(
// CHECK:         linalg.matmul
// CHECK-SAME:      iree.opt.data_tiling

// -----

// `enable_inner_tiled` AVX-512-BF16 target: bf16/f32 now matches the
// MMA_X86_AVX512BF16_1x16x2_F32_BF16 intrinsic, so the matmul gets the hint.

util.global private @__device_avx512bf16 = #hal.device.target<"local", [
  #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
    target_triple = "x86_64-xyz-xyz",
    cpu_features = "+avx512f,+avx512bf16",
    enable_inner_tiled = true,
    iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>
  }>
]> : !hal.device

util.func public @matmul_bf16_f32_avx512bf16(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?xbf16>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
         ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
         outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: @matmul_bf16_f32_avx512bf16(
// CHECK:         linalg.matmul
// CHECK-SAME:      iree.opt.data_tiling

// -----

// Targets without `enable_inner_tiled` are not subject to the opt-out: data
// tiling there goes through the legacy mmt4d path and doesn't need an MMA
// intrinsic. So bf16/f32 on AVX2 still gets the hint when the target lacks
// `enable_inner_tiled`.

util.global private @__device_avx2_legacy = #hal.device.target<"local", [
  #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
    target_triple = "x86_64-xyz-xyz",
    cpu_features = "+avx2,+fma",
    iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>
  }>
]> : !hal.device

util.func public @matmul_bf16_f32_avx2_no_inner_tiled(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?xbf16>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
         ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
         outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: @matmul_bf16_f32_avx2_no_inner_tiled(
// CHECK:         linalg.matmul
// CHECK-SAME:      iree.opt.data_tiling

// -----

// Multi-target dispatch: opt-out is conservative across `enable_inner_tiled`
// targets. If at least one such target has no matching MMA, drop the hint
// (otherwise compilation for that target would produce an untranslatable
// `iree_codegen.inner_tiled`). Here AVX-512-BF16 supports bf16/f32 but AVX2
// doesn't, so the matmul opts out.

util.global private @__device_multi = #hal.device.target<"local", [
  #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
    target_triple = "x86_64-xyz-xyz",
    cpu_features = "+avx512f,+avx512bf16",
    enable_inner_tiled = true,
    iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>
  }>,
  #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64-2", {
    target_triple = "x86_64-xyz-xyz",
    cpu_features = "+avx2,+fma",
    enable_inner_tiled = true,
    iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>
  }>
]> : !hal.device

util.func public @matmul_bf16_f32_multi(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?xbf16>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
         ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
         outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: @matmul_bf16_f32_multi(
// CHECK:         linalg.matmul
// CHECK-NOT:       iree.opt.data_tiling
// CHECK:         util.return
