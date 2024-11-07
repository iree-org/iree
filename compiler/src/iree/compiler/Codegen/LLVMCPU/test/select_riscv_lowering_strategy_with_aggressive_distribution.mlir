// RUN: iree-opt --iree-llvmcpu-riscv-aggressive-distribution=true --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

#executable_target_embedded_elf_riscv_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+zvl1024b,+v", data_layout = "e-m:e-p:64:64-i64:64-i256:256-n32:64-S256", native_vector_size = 256 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
builtin.module {
  func.func @matmul_riscv_vl1024() attributes {hal.executable.target = #executable_target_embedded_elf_riscv_64_} {
    %cst = arith.constant 0.0 : f32
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<512x256xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<384x256xf32>>
    %lhs = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
    %rhs = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x256xf32>> -> tensor<512x256xf32>
    %init = tensor.empty() : tensor<384x256xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<384x256xf32>) -> tensor<384x256xf32>
    %res = linalg.matmul ins(%lhs, %rhs : tensor<384x512xf32>, tensor<512x256xf32>) outs(%fill : tensor<384x256xf32>) -> tensor<384x256xf32>
    flow.dispatch.tensor.store %res, %2, offsets = [0, 0], sizes = [384, 256], strides = [1, 1] : tensor<384x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x256xf32>>
    return
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 256], [7, 128], [0, 0], [0, 0]]>
//  CHECK-DAG: #[[CONFIG2:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 256, 0], [32, 256, 0], [0, 0, 0], [7, 128, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert, {{\{}}enable_loop_peeling}>
//      CHECK: func.func @matmul_riscv_vl1024()
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG2]]
