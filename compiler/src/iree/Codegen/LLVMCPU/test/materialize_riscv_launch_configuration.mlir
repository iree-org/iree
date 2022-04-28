// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=true}))' -split-input-file %s | FileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_riscv  {
  hal.executable.variant public @embedded_elf_x86_64, target = #hal.executable.target<
    "llvm",
    "embedded-elf-riscv_32", {
      cpu_features = "+m,+f",
      data_layout = "e-m:e-p:32:32-i64:64-n32-S128",
      native_vector_size = 0 : index,
      target_triple = "riscv32-unknown-unknown-eabi-elf"
    }> {
    hal.executable.entry_point public @matmul_riscv layout(#executable_layout)
    builtin.module {
      func.func @matmul_riscv() {
        %cst = arith.constant 0.0 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:384x512xf32>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:512x128xf32>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:384x128xf32>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [384, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:384x512xf32> -> tensor<384x512xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [512, 128], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:512x128xf32> -> tensor<512x128xf32>
        %init = linalg.init_tensor [384, 128] : tensor<384x128xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<384x128xf32>) -> tensor<384x128xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<384x512xf32>, tensor<512x128xf32>)
            outs(%fill : tensor<384x128xf32>) -> tensor<384x128xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [384, 128], strides = [1, 1]
            : tensor<384x128xf32> -> !flow.dispatch.tensor<writeonly:384x128xf32>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [8, 32, 0], [0, 0, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.entry_point public @matmul_riscv
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]
