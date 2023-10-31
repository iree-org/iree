// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-select-lowering-strategy)))' --iree-codegen-llvm-disable-distribution --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_static  {
  hal.executable.variant public @embedded_elf_x86_64 target(#hal.executable.target<
    "llvm-cpu",
    "embedded-elf-x86_64", {
      data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      native_vector_size = 16 : index,
      target_triple = "x86_64-none-elf"
    }>) {
    hal.executable.export public @matmul_static layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_static() {
        %cst = arith.constant 0.0 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x128xf32>>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [384, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [512, 128], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<512x128xf32>> -> tensor<512x128xf32>
        %init = tensor.empty() : tensor<384x128xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<384x128xf32>) -> tensor<384x128xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<384x512xf32>, tensor<512x128xf32>)
            outs(%fill : tensor<384x128xf32>) -> tensor<384x128xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [384, 128], strides = [1, 1]
            : tensor<384x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x128xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 0, 0], [8, 32, 0], [0, 0, 16], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPadExpert>
//      CHECK: hal.executable.export public @matmul_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]
