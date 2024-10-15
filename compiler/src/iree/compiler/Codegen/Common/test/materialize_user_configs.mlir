// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-user-configs)' --split-input-file %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [32, 32, 0], [0, 0, 32], [0, 0, 0]]>
#executable_target_system_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "system-elf-x86_64", {target_triple = "x86_64-xyz-xyz"}>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
module {
  func.func @preset_config() attributes {hal.executable.target = #executable_target_system_elf_x86_64_} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<256x512xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
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

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<
    "llvm-cpu", "embedded-elf-x86_64",
    {cpu_features = "+avx512f",
     data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
     native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
func.func @custom_op_compilation_info(%arg0 : tensor<384x512xf32>, %arg1 : tensor<512x128xf32>,
    %arg2 : tensor<128xf32>) -> tensor<384x128xf32>
    attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<384x128xf32>
  %1 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1)[s0] -> (d0, s0)>,
                       affine_map<(d0, d1)[s0] -> (s0, d1)>,
                       affine_map<(d0, d1)[s0] -> (d1)>,
                       affine_map<(d0, d1)[s0] -> (d0, d1)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                        #iree_linalg_ext.iterator_type<parallel>]}
      attributes {
        compilation_info = #iree_codegen.compilation_info<
          lowering_config = #iree_codegen.lowering_config<tile_sizes = [[24, 32]]>,
          translation_info = <CPUDefault>>
      }
      ins(%arg0, %arg1, %arg2 : tensor<384x512xf32>, tensor<512x128xf32>, tensor<128xf32>)
      outs(%0 : tensor<384x128xf32>) {
    ^bb0(%t0 : tensor<?x?xf32>, %t1 : tensor<?x?xf32>, %t2 : tensor<?xf32>, %t3 : tensor<?x?xf32>):
      %2 = linalg.fill ins(%cst : f32) outs(%t3 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %3 = linalg.matmul ins(%t0, %t1 : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %4 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%3, %t2 : tensor<?x?xf32>, tensor<?xf32>)
          outs(%t3 : tensor<?x?xf32>) {
        ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
          %5 = arith.addf %b0, %b1 : f32
          linalg.yield %5 : f32
      } -> tensor<?x?xf32>
      iree_linalg_ext.yield %4 : tensor<?x?xf32>
  } -> tensor<384x128xf32>
  return %1 : tensor<384x128xf32>
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[24, 32]]>
//  CHECK-DAG: #[[TRANSLATION_INFO:.+]] = #iree_codegen.translation_info<CPUDefault>
//      CHECK: func @custom_op_compilation_info(
// CHECK-SAME:     translation_info = #translation
//      CHECK:   iree_linalg_ext.custom_op
// CHECK-SAME:       attributes {lowering_config = #[[CONFIG]]}
//  CHECK-NOT:   compilation_info
