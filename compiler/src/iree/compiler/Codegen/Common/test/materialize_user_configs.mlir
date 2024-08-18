// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-user-configs)' --split-input-file %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [32, 32, 0], [0, 0, 32], [0, 0, 0]]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {target_triple = "x86_64-xyz-xyz"}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
module {
  func.func @preset_config() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) : !flow.dispatch.tensor<readonly:tensor<256x512xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x512xf32>> -> tensor<256x512xf32>
    %5 = tensor.empty() : tensor<128x512xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %7 = linalg.matmul {compilation_info = #compilation} ins(%3, %4 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%6 : tensor<128x512xf32>) -> tensor<128x512xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [32, 32, 0], [0, 0, 32], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: func.func @preset_config()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]
