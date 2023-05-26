// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true})))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @batch_matmul_f32_16x4096x40x4096 {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader], []>, AMD:DiscreteGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>>
    }> {
    hal.executable.export @batch_matmul_f32_16x4096x40x4096 layout(#pipeline_layout)
    builtin.module {
      func.func @batch_matmul_f32_16x4096x40x4096() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x4096x4096xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x4096x40xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<16x4096x40xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 4096, 4096], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x4096x4096xf32>> -> tensor<16x4096x4096xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [16, 4096, 40], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x4096x40xf32>> -> tensor<16x4096x40xf32>
        %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [16, 4096, 40], strides = [1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<16x4096x40xf32>> -> tensor<16x4096x40xf32>
        %6 = linalg.batch_matmul ins(%3, %4 : tensor<16x4096x4096xf32>, tensor<16x4096x40xf32>) outs(%5 : tensor<16x4096x40xf32>) -> tensor<16x4096x40xf32>
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [16, 4096, 40], strides = [1, 1, 1] : tensor<16x4096x40xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x4096x40xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 512, 8, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize pipeline_depth = 1>
//      CHECK: hal.executable.export public @batch_matmul_f32_16x4096x40x4096
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [2 : index, 64 : index, 1 : index]
//      CHECK: func.func @batch_matmul_f32_16x4096x40x4096()
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @matmul_f16_64x640x320 {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader, Float16], []>, AMD:DiscreteGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>>
    }> {
    hal.executable.export @matmul_f16_64x640x320 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_f16_64x640x320() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64x320xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<320x640xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x640xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 320], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x320xf16>> -> tensor<64x320xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [320, 640], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<320x640xf16>> -> tensor<320x640xf16>
        %5 = tensor.empty() : tensor<64x640xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<64x640xf16>) -> tensor<64x640xf16>
        %7 = linalg.matmul ins(%3, %4 : tensor<64x320xf16>, tensor<320x640xf16>) outs(%6 : tensor<64x640xf16>) -> tensor<64x640xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [64, 640], strides = [1, 1] : tensor<64x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<64x640xf16>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 128, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize pipeline_depth = 2 store_stage = 0>
//      CHECK: hal.executable.export public @matmul_f16_64x640x320
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [16 : index, 16 : index, 1 : index]
//      CHECK: func.func @matmul_f16_64x640x320()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @batch_matmul_f32_16x4096x40x4096 {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader], []>, AMD:DiscreteGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>>
    }> {
    hal.executable.export @batch_matmul_f32_16x4096x40x4096 layout(#pipeline_layout)
    builtin.module {
      func.func @batch_matmul_f32_16x4096x40x4096() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x4096x4096xf32>>
        %7 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x4096x48xf32>>
        %8 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x4096x48xf32>>
        %9 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0], sizes = [16, 4096, 4096], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x4096x4096xf32>> -> tensor<16x4096x4096xf32>
        %10 = flow.dispatch.tensor.load %7, offsets = [0, 0, 0], sizes = [16, 4096, 48], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x4096x48xf32>> -> tensor<16x4096x48xf32>
        %11 = tensor.empty() : tensor<16x4096x48xf32>
        %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<16x4096x48xf32>) -> tensor<16x4096x48xf32>
        %13 = linalg.batch_matmul ins(%9, %10 : tensor<16x4096x4096xf32>, tensor<16x4096x48xf32>) outs(%12 : tensor<16x4096x48xf32>) -> tensor<16x4096x48xf32>
        flow.dispatch.tensor.store %13, %8, offsets = [0, 0, 0], sizes = [16, 4096, 48], strides = [1, 1, 1] : tensor<16x4096x48xf32> -> !flow.dispatch.tensor<writeonly:tensor<16x4096x48xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 256, 16, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize pipeline_depth = 1>
//      CHECK: hal.executable.export public @batch_matmul_f32_16x4096x40x4096
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [4 : index, 32 : index, 1 : index]
//      CHECK: func.func @batch_matmul_f32_16x4096x40x4096()
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @batch_matmul_f16_1x4096x4096x512 {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader], []>, AMD:DiscreteGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>>
    }> {
    hal.executable.export @batch_matmul_f16_1x4096x4096x512 layout(#pipeline_layout)
    builtin.module {
      func.func @batch_matmul_f16_1x4096x4096x512() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x4096x512xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x512x4096xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x4096x4096xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 4096, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x4096x512xf16>> -> tensor<1x4096x512xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 512, 4096], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x512x4096xf16>> -> tensor<1x512x4096xf16>
        %5 = tensor.empty() : tensor<1x4096x4096xf32>
        %6 = tensor.empty() : tensor<1x4096x4096xf16>
        %7 = linalg.fill ins(%cst : f16) outs(%6 : tensor<1x4096x4096xf16>) -> tensor<1x4096x4096xf16>
        %8 = linalg.batch_matmul ins(%3, %4 : tensor<1x4096x512xf16>, tensor<1x512x4096xf16>) outs(%7 : tensor<1x4096x4096xf16>) -> tensor<1x4096x4096xf16>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<1x4096x4096xf16>) outs(%5 : tensor<1x4096x4096xf32>) {
        ^bb0(%in: f16, %out: f32):
          %10 = arith.extf %in : f16 to f32
          linalg.yield %10 : f32
        } -> tensor<1x4096x4096xf32>
        flow.dispatch.tensor.store %9, %2, offsets = [0, 0, 0], sizes = [1, 4096, 4096], strides = [1, 1, 1] : tensor<1x4096x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x4096x4096xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 256, 32]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize pipeline_depth = 1>
//      CHECK: hal.executable.export public @batch_matmul_f16_1x4096x4096x512
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [32 : index, 8 : index, 1 : index]
//      CHECK: func.func @batch_matmul_f16_1x4096x4096x512()
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]
