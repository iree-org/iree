// RUN: iree-opt --iree-codegen-lower-bitcode-ukernels --split-input-file %s | FileCheck %s

// Companion to lower_inner_tiled_to_bitcode_ukernel.mlir, exercising the
// i8 x i8 -> i32 VNNI seed. Same shape of test (user-supplied
// `hal.executable.objects` carrying placeholder bitcode bytes), distinct
// from the bf16 test only in the ukernel name and element types — this
// guards against a regression where the framework would accidentally
// hard-code the bf16 seed.

#executable_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  iree_codegen.ukernel_provider = #iree_cpu.ukernel_provider
}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: @i8_matmul_with_ukernel_descriptor
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<16x2xi8>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<16x2xi8>
// CHECK-NOT:     linalg.generic
// CHECK:         %[[OUT:.+]] = linalg.fill
// CHECK:         %[[UKERNEL:.+]] = iree_codegen.ukernel.generic
// CHECK-SAME:        {hal.executable.objects = [{{.*}}"iree_uk_mma_x86_avx512vnni_16x16x2_i32_i8_casti16.x86_64_avx512vnni.bc"
// CHECK-SAME:         iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<"iree_uk_mma_x86_avx512vnni_16x16x2_i32_i8_casti16", bitcode>}
// CHECK-SAME:        "iree_uk_mma_x86_avx512vnni_16x16x2_i32_i8_casti16"
// CHECK-SAME:        ins(%[[LHS]], %[[RHS]] : tensor<16x2xi8>, tensor<16x2xi8>)
// CHECK-SAME:        outs(%[[OUT]] : tensor<16x16xi32>)
// CHECK:         return %[[UKERNEL]]
module attributes {hal.executable.target = #executable_target} {
  func.func @i8_matmul_with_ukernel_descriptor(
      %arg0: tensor<16x2xi8>, %arg1: tensor<16x2xi8>) -> tensor<16x16xi32> {
    %c0 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<16x16xi32>
    %1 = linalg.fill ins(%c0 : i32) outs(%0 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %2 = linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%arg0, %arg1 : tensor<16x2xi8>, tensor<16x2xi8>)
      outs(%1 : tensor<16x16xi32>) attrs = {
      iree_codegen.ukernel = #iree_codegen.ukernel_descriptor<
          "iree_uk_mma_x86_avx512vnni_16x16x2_i32_i8_casti16", bitcode>,
      hal.executable.objects = [
        #hal.executable.object<{
          path = "iree_uk_mma_x86_avx512vnni_16x16x2_i32_i8_casti16.x86_64_avx512vnni.bc",
          data = dense<[0, 1, 2, 3]> : vector<4xi8>}>
      ]
    } {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %3 = arith.extsi %in : i8 to i32
      %4 = arith.extsi %in_0 : i8 to i32
      %5 = arith.muli %3, %4 : i32
      %6 = arith.addi %out, %5 : i32
      linalg.yield %6 : i32
    } -> tensor<16x16xi32>
    return %2 : tensor<16x16xi32>
  }
}
