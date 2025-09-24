// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --iree-llvmcpu-disable-distribution --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-none-elf"}>
func.func @matmul_static() attributes {hal.executable.target = #executable_target_embedded_elf_x86_64_} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x128xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x128xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x128xf32>> -> tensor<512x128xf32>
  %5 = tensor.empty() : tensor<384x128xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<384x128xf32>) -> tensor<384x128xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<384x512xf32>, tensor<512x128xf32>) outs(%6 : tensor<384x128xf32>) -> tensor<384x128xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [384, 128], strides = [1, 1] : tensor<384x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x128xf32>>
  return
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0, 0, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 4]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_static()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]
