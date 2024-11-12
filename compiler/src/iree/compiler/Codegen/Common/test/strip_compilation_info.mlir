// RUN: iree-opt --split-input-file --iree-codegen-strip-compilation-info %s | FileCheck %s

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64>
func.func @main() attributes {translation_info = #translation_info} {
  return
}

// CHECK-LABEL: func.func @main
// CHECK-NOT:   iree_codegen.translation_info
// CHECK-NOT:   LLVMGPUVectorDistribute

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @strip_main {
  hal.executable.variant public @strip_main target(#hal.executable.target<"", "", {}>) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
    builtin.module {
      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<pipeline = None subgroup_size = 32>}  {
        return
      }
      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<pipeline = None subgroup_size = 32>} {
        return
      }
    }
  }
}

// CHECK-LABEL: hal.executable private @strip_main
// CHECK: @fn1
// CHECK-NOT:   translation_info =
// CHECK: @fn2
// CHECK-NOT:   translation_info =
// CHECK: return

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[128, 256], [16, 16]]>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [16, 8, 1] subgroup_size = 64>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @matmul_128x1024x256(%lhs : tensor<128x256xf32>, %rhs: tensor<256x1024xf32>, %init: tensor<128x1024xf32>) -> tensor<128x1024xf32> {
  %result =  linalg.matmul {compilation_info = #compilation} ins(%lhs, %rhs : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%init : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  return %result : tensor<128x1024xf32>
}

// CHECK-LABEL: func.func @matmul_128x1024x256
// CHECK-NOT:   iree_codegen.translation_info
// CHECK-NOT:   iree_codegen.lowering_config
// CHECK-NOT:   iree_codegen.compilation_info

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[128, 256], [16, 16]]>
func.func @matmul_128x1024x256_1(%lhs : tensor<128x256xf32>, %rhs: tensor<256x1024xf32>, %init: tensor<128x1024xf32>) -> tensor<128x1024xf32> {
  %result =  linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 256], [16, 16]]>} ins(%lhs, %rhs : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%init : tensor<128x1024xf32>) -> tensor<128x1024xf32>
  return %result : tensor<128x1024xf32>
}

// CHECK-LABEL: func.func @matmul_128x1024x256_1
// CHECK-NOT:   iree_codegen.lowering_config
