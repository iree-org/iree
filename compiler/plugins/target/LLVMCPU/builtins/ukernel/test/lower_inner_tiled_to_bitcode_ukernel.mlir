// RUN: iree-opt --iree-codegen-lower-bitcode-ukernels --split-input-file %s | FileCheck %s

// Tests that an op carrying a `iree_codegen.ukernel = <"name", bitcode>`
// descriptor is rewritten by `LowerBitcodeUKernels` to an
// `iree_codegen.ukernel.generic` call, and that an attached
// `hal.executable.objects` (carrying the ukernel bitcode bytes) is preserved
// on the new op. The `#iree_cpu.ukernel_provider` on the executable target is
// looked up but currently delegates the rewrite to the default fallback in the
// pass; specialized handling of `iree_codegen.inner_tiled` (threading
// `intrinsics_{m,n,k}` and the outer K count as scalar operands) will be
// added in a follow-up commit alongside the SelectUKernels pass that sets
// the descriptor and attaches the bitcode.
//
// Bitcode bytes here are an opaque placeholder; the pass treats
// `hal.executable.objects` as a discardable attribute and does not parse it.

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  iree_codegen.ukernel_provider = #iree_cpu.ukernel_provider
}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: @bf16_matmul_with_ukernel_descriptor
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<16x2xbf16>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<16x2xbf16>
// CHECK-NOT:     linalg.generic
// CHECK:         %[[OUT:.+]] = linalg.fill
// CHECK:         %[[UKERNEL:.+]] = iree_codegen.ukernel.generic
// CHECK-SAME:        {hal.executable.objects = [{{.*}}"iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16.x86_64_avx512bf16.bc"
// CHECK-SAME:         iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16", bitcode>}
// CHECK-SAME:        "iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16"
// CHECK-SAME:        ins(%[[LHS]], %[[RHS]] : tensor<16x2xbf16>, tensor<16x2xbf16>)
// CHECK-SAME:        outs(%[[OUT]] : tensor<16x16xf32>)
// CHECK:         return %[[UKERNEL]]
module attributes {hal.executable.target = #executable_target} {
  func.func @bf16_matmul_with_ukernel_descriptor(
      %arg0: tensor<16x2xbf16>, %arg1: tensor<16x2xbf16>) -> tensor<16x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %2 = linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%arg0, %arg1 : tensor<16x2xbf16>, tensor<16x2xbf16>)
      outs(%1 : tensor<16x16xf32>) attrs = {
      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<
          "iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16", bitcode>,
      hal.executable.objects = [
        #hal.executable.object<{
          path = "iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16.x86_64_avx512bf16.bc",
          data = dense<[0, 1, 2, 3]> : vector<4xi8>}>
      ]
    } {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<16x16xf32>
    return %2 : tensor<16x16xf32>
  }
}
