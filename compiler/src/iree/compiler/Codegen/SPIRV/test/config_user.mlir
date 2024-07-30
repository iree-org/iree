// RUN: iree-opt --split-input-file --iree-gpu-test-target=vp_android_baseline_2022@vulkan --pass-pipeline='builtin.module(iree-codegen-materialize-user-configs, iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[128, 256], [16, 16]]>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [16, 8, 1] subgroup_size = 64>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
module {
  func.func @matmul_128x1024x256() {
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>> -> tensor<256x1024xf32>
    %5 = tensor.empty() : tensor<128x1024xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
    %7 = linalg.matmul {compilation_info = #compilation} ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%6 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : tensor<128x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 256], [16, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize workgroup_size = [16, 8, 1] subgroup_size = 64>
//      CHECK: func.func @matmul_128x1024x256()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]
