// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-materialize-user-configs)))' --split-input-file %s | FileCheck %s

#compilation = #iree_codegen.compilation_info<
    lowering_config = <tile_sizes = [[64, 64, 0], [32, 32, 0], [0, 0, 32], [0, 0, 0]]>,
    translation_info  = <CPUDoubleTilingExpert>>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @preset_config_matmul_tensors  {
  hal.executable.variant @system_elf_x86_64 target(<"llvm-cpu", "system-elf-x86_64", {target_triple="x86_64-xyz-xyz"}>) {
    hal.executable.export @preset_config layout(#pipeline_layout)
    builtin.module {
      func.func @preset_config() {
        %cst = arith.constant 0.000000e+00 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<256x512xf32>>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [128, 256], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [256, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<256x512xf32>> -> tensor<256x512xf32>
        %init = tensor.empty() : tensor<128x512xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x512xf32>) -> tensor<128x512xf32>
        %gemm = linalg.matmul {compilation_info = #compilation}
            ins(%lhs, %rhs : tensor<128x256xf32>, tensor<256x512xf32>)
            outs(%fill : tensor<128x512xf32>) -> tensor<128x512xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [128, 512], strides = [1, 1]
            : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
        return
      }
    }
  }
}
//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 64, 0], [32, 32, 0], [0, 0, 32], [0, 0, 0]]>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//      CHECK: hal.executable.export
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK: func.func @preset_config
//      CHECK:   linalg.matmul
// CHECK-SAME:       lowering_config = #[[CONFIG]]
