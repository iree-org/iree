// RUN: iree-opt --split-input-file --iree-gpu-test-target=vp_android_baseline_2022@vulkan --pass-pipeline='builtin.module(iree-codegen-materialize-user-configs, iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[128, 256], [16, 16]]>
#translation = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [16, 8, 1] subgroup_size = 64>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @matmul_128x1024x256(%3: tensor<128x256xf32>, %4: tensor<256x1024xf32>) -> tensor<128x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<128x1024xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  %7 = linalg.matmul {compilation_info = #compilation} ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%6 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  return %7 : tensor<128x1024xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 256], [16, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseVectorize workgroup_size = [16, 8, 1] subgroup_size = 64>
//      CHECK: func.func @matmul_128x1024x256(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]
