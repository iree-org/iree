// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-materialize-user-configs, iree-spirv-lower-executable-target-pass{test-lowering-configuration=true})))' %s | FileCheck %s

#compilation = #iree_codegen.compilation_info<
    lowering_config  = <tile_sizes = [[128, 256], [16, 16]]>,
    translation_info = <SPIRVBaseVectorize>,
    workgroup_size = [16, 8, 1], subgroup_size = 64>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable public @user_config {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
          max_compute_shared_memory_size = 16384,
          max_compute_workgroup_invocations = 128,
          max_compute_workgroup_size = [128, 128, 64],
          subgroup_size = 32>>
    }>) {
    hal.executable.export public @matmul_128x1024x256 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_128x1024x256() {
        %cst = arith.constant 0.000000e+00 : f32
        %c128 = arith.constant 128 : index
        %c1024 = arith.constant 1024 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>> -> tensor<256x1024xf32>
        %15 = tensor.empty() : tensor<128x1024xf32>
        %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
        %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup", compilation_info = #compilation}
            ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%16 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : tensor<128x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 256], [16, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @matmul_128x1024x256
// CHECK-SAME:     subgroup_size = 64 : index
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
// CHECK-SAME:     workgroup_size = [16 : index, 8 : index, 1 : index]
//      CHECK: linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]
