// RUN: iree-opt --iree-codegen-lower-bitcode-ukernels --split-input-file %s | FileCheck %s

// Three-way test of the `#iree_cpu.ukernel_provider` bitcode lookup:
//
//   (1) Built-in only. Op carries a ukernel descriptor naming a built-in but
//       no `hal.executable.objects`; the provider must find the matching
//       bitcode in the global `EmbeddedDataDirectory` (populated at LLVMCPU
//       plugin init from the embedded TOC) and attach it.
//
//   (2) Bring-your-own override. Op carries a user-supplied
//       `hal.executable.objects` with a matching filename. The provider must
//       prefer the user's bytes over the built-in — never silently swap in
//       the embedded copy. The user bytes here (`dense<[0xAA, ...]>`) are
//       intentionally distinguishable from the real built-in bitcode.
//
//   (3) Pure BYO with a name that does not match any built-in. The provider
//       must still attach the user's bitcode and the rewrite must succeed.
//
// Together these guard the three branches of `getUKernelBitcode` against
// silent regressions.

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  iree_codegen.ukernel_provider = #iree_cpu.ukernel_provider
}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// (1) Built-in only: no user-supplied hal.executable.objects → expect the
//     embedded bf16 ukernel bitcode to be attached as a `dense_resource`.
// CHECK-LABEL: @builtin_lookup
// CHECK:         %[[UK:.+]] = iree_codegen.ukernel.generic
// CHECK-SAME:        {hal.executable.objects = [
// CHECK-SAME:          #hal.executable.object<{
// CHECK-SAME:            path = "iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16.x86_64_avx512bf16.bc"
// CHECK-SAME:            data = dense_resource<iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16.x86_64_avx512bf16.bc>
// CHECK-SAME:         iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16", bitcode>}
// CHECK-SAME:        "iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16"
module attributes {hal.executable.target = #executable_target} {
  func.func @builtin_lookup(
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
          "iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16", bitcode>
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

// -----

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  iree_codegen.ukernel_provider = #iree_cpu.ukernel_provider
}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// (2) Override: user bytes (0xAA-pattern, all four lanes equal — prints as
//     the signless-i8 collapsed splat `dense<-86>`) must reach the
//     rewritten op unchanged. If the provider silently replaced them with
//     the embedded bitcode, the splat below would be swapped for a
//     `dense_resource<…>` and the CHECK-NOT would fail.
// CHECK-LABEL: @byo_overrides_builtin
// CHECK:         %[[UK:.+]] = iree_codegen.ukernel.generic
// CHECK-SAME:        {hal.executable.objects = [
// CHECK-SAME:          #hal.executable.object<{
// CHECK-SAME:            path = "iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16.x86_64_avx512bf16.bc"
// CHECK-SAME:            data = dense<-86>
// CHECK-NOT:       dense_resource
// CHECK-SAME:        "iree_uk_mma_x86_avx512bf16_1x16x2_f32_bf16"
module attributes {hal.executable.target = #executable_target} {
  func.func @byo_overrides_builtin(
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
          data = dense<[170, 170, 170, 170]> : vector<4xi8>}>
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

// -----

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  iree_codegen.ukernel_provider = #iree_cpu.ukernel_provider
}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// (3) Pure BYO: user-only name with no matching built-in. The provider must
//     still surface the user's bitcode on the rewritten op.
// CHECK-LABEL: @byo_pure_with_custom_name
// CHECK:         %[[UK:.+]] = iree_codegen.ukernel.generic
// CHECK-SAME:        {hal.executable.objects = [
// CHECK-SAME:          path = "my_custom_external_ukernel.bc"
// CHECK-SAME:          data = dense<[10, 20, 30, 40]>
// CHECK-SAME:        "my_custom_external_ukernel"
module attributes {hal.executable.target = #executable_target} {
  func.func @byo_pure_with_custom_name(
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
          "my_custom_external_ukernel", bitcode>,
      hal.executable.objects = [
        #hal.executable.object<{
          path = "my_custom_external_ukernel.bc",
          data = dense<[10, 20, 30, 40]> : vector<4xi8>}>
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
