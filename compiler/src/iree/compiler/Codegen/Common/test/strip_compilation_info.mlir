// RUN: iree-opt --split-input-file --iree-codegen-strip-compilation-info %s | FileCheck %s

#translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64>
func.func @main() attributes {translation_info = #translation_info} {
    return
}

// CHECK-LABEL: func.func @main
// CHECK-NOT:   #translation_info =
// CHECK-NOT:   LLVMGPUVectorDistribute

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @strip_main {
  hal.executable.variant public @strip_main target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
    builtin.module {
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<None subgroup_size = 32>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<None subgroup_size = 32>} {
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable private @strip_main
// CHECK: @fn1
// CHECK-NOT:   #translation_info =
// CHECK: @fn2
// CHECK-NOT:   #translation_info =
// CHECK: return

#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[128, 256], [16, 16]]>
#translation = #iree_codegen.translation_info<None workgroup_size = [16, 8, 1] subgroup_size = 64>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @matmul_128x1024x256() {
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
  %1 = hal.interface.binding.subspan layout(#layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>>
  %2 = hal.interface.binding.subspan layout(#layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>> -> tensor<256x1024xf32>
  %5 = tensor.empty() : tensor<128x1024xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  %7 = linalg.matmul {compilation_info = #compilation} ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%6 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : tensor<128x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
  return
}

// CHECK-LABEL: func.func @matmul_128x1024x256
// CHECK-NOT:   #translation_info =
// CHECK-NOT:   #config =
// CHECK-NOT:   #compilation
