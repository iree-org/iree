// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=true})))' --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_riscv  {
  hal.executable.variant public @embedded_elf_x86_64 target(#hal.executable.target<
    "llvm-cpu",
    "embedded-elf-riscv_32", {
      cpu_features = "+m,+f",
      data_layout = "e-m:e-p:32:32-i64:64-n32-S128",
      native_vector_size = 16 : index,
      target_triple = "riscv32-none-elf"
    }>) {
    hal.executable.export public @matmul_riscv layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_riscv() {
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

//  CHECK-DAG: #[[CONFIG:.+]] =  #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 64, 0], [8, 32, 0], [0, 0, 1], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
//      CHECK: hal.executable.export public @matmul_riscv
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @thin_depthwise_conv_static {
  hal.executable.variant public @embedded_elf_x86_64 target(#hal.executable.target<
    "llvm-cpu",
    "embedded-elf-riscv_32", {
      cpu_features = "+m,+f",
      data_layout = "e-m:e-p:32:32-i64:64-n32-S128",
      native_vector_size = 16 : index,
      target_triple = "riscv32-none-elf"
    }>) {
    hal.executable.export public @thin_depthwise_conv_static layout(#pipeline_layout)
    builtin.module {
      func.func @thin_depthwise_conv_static() {
        %cst = arith.constant 0.0 : f32
        %input_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<1x57x57x72xf32>>
        %filter_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<3x3x72xf32>>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<1x28x28x72xf32>>
        %input = flow.dispatch.tensor.load %input_binding, offsets = [0, 0, 0, 0], sizes = [1, 161, 161, 240], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x57x57x72xf32>> -> tensor<1x57x57x72xf32>
        %filter = flow.dispatch.tensor.load %filter_binding, offsets = [0, 0, 0], sizes = [3, 3, 240], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<3x3x72xf32>> -> tensor<3x3x72xf32>
        %init = tensor.empty() : tensor<1x28x28x72xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>
        %conv = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
          ins(%input, %filter : tensor<1x57x57x72xf32>, tensor<3x3x72xf32>)
          outs(%fill : tensor<1x28x28x72xf32>) -> tensor<1x28x28x72xf32>

        flow.dispatch.tensor.store %conv, %result_binding, offsets = [0, 0, 0, 0], sizes = [1, 28, 28, 72], strides = [1, 1, 1, 1]
            : tensor<1x28x28x72xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x28x28x72xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 7, 7, 72, 0, 0], [1, 1, 7, 4, 0, 0], [0, 0, 0, 0, 1, 3], [0, 0, 0, 0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
//      CHECK: hal.executable.export public @thin_depthwise_conv_static
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:     linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:       lowering_config  = #[[CONFIG]]

