// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true})))' %s | FileCheck %s

// Large matmul that can match the best tiling scheme.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @matmul_1024x2048x512 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 64>>
    }>) {
    hal.executable.export @matmul_1024x2048x512 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_1024x2048x512() {
        %c0 = arith.constant 0 : index
        %c2048 = arith.constant 2048 : index
        %c1024 = arith.constant 1024 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x2048xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1024x2048xf32>>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>> -> tensor<1024x512xf32>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 2048], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<512x2048xf32>> -> tensor<512x2048xf32>
        %15 = tensor.empty() : tensor<1024x2048xf32>
        %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
        %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup"}
            ins(%8, %10 : tensor<1024x512xf32>, tensor<512x2048xf32>) outs(%16 : tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [1024, 2048], strides = [1, 1]
            : tensor<1024x2048xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024x2048xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 128], [16, 4], [0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @matmul_1024x2048x512
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [32 : index, 2 : index, 1 : index]
//      CHECK: func.func @matmul_1024x2048x512()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Small matmul N that can still tile to all threads in a workgroup.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @matmul_3136x24x96 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 64>>
    }>) {
    hal.executable.export @matmul_3136x24x96 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_3136x24x96() {
        %c0 = arith.constant 0 : index
        %c24 = arith.constant 24 : index
        %c3136 = arith.constant 3136 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<3136x96xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<96x24xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<3136x24xf32>>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [3136, 96], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<3136x96xf32>> -> tensor<3136x96xf32>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [96, 24], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<96x24xf32>> -> tensor<96x24xf32>
        %15 = tensor.empty() : tensor<3136x24xf32>
        %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<3136x24xf32>) -> tensor<3136x24xf32>
        %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup"}
            ins(%8, %10 : tensor<3136x96xf32>, tensor<96x24xf32>) outs(%16 : tensor<3136x24xf32>) -> tensor<3136x24xf32>
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [3136, 24], strides = [1, 1]
            : tensor<3136x24xf32> -> !flow.dispatch.tensor<writeonly:tensor<3136x24xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[448, 8], [14, 4], [0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @matmul_3136x24x96
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [2 : index, 32 : index, 1 : index]
//      CHECK: func.func @matmul_3136x24x96()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Small matmul M that can still tile to all threads in a workgroup.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @matmul_196x64x192 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 64>>
    }>) {
    hal.executable.export @matmul_196x64x192 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_196x64x192() {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c196 = arith.constant 196 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<196x192xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<192x64xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<196x64xf32>>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [196, 192], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<196x192xf32>> -> tensor<196x192xf32>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [192, 64], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<192x64xf32>> -> tensor<192x64xf32>
        %15 = tensor.empty() : tensor<196x64xf32>
        %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<196x64xf32>) -> tensor<196x64xf32>
        %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup"}
            ins(%8, %10 : tensor<196x192xf32>, tensor<192x64xf32>) outs(%16 : tensor<196x64xf32>) -> tensor<196x64xf32>
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [196, 64], strides = [1, 1]
            : tensor<196x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<196x64xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[28, 64], [7, 4], [0, 0, 8]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @matmul_196x64x192
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [16 : index, 4 : index, 1 : index]
//      CHECK: func.func @matmul_196x64x192()
//      CHECK:   linalg.matmul
// CHECK-SAME:      lowering_config = #[[CONFIG]]

// -----

// Small matmul K that can still tile to all threads in a workgroup.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @matmul_12544x96x16 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 64>>
    }>) {
    hal.executable.export @matmul_12544x96x16 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_12544x96x16() {
        %c0 = arith.constant 0 : index
        %c96 = arith.constant 96 : index
        %c12544 = arith.constant 12544 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<12544x16xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x96xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<12544x96xf32>
        linalg.fill ins(%cst : f32) outs(%2 : memref<12544x96xf32>)
        linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%0, %1 : memref<12544x16xf32>, memref<16x96xf32>) outs(%2 : memref<12544x96xf32>)
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[128, 32], [16, 4], [0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @matmul_12544x96x16
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [8 : index, 8 : index, 1 : index]
//      CHECK: func.func @matmul_12544x96x16()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Odd matmul M and small N that cannot utilize all threads in a workgroup.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @matmul_49x160x576 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 64>>
    }>) {
    hal.executable.export @matmul_49x160x576 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_49x160x576() {
        %c0 = arith.constant 0 : index
        %c160 = arith.constant 160 : index
        %c49 = arith.constant 49 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<49x576xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<576x160xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<49x160xf32>>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [49, 576], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<49x576xf32>> -> tensor<49x576xf32>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [576, 160], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<576x160xf32>> -> tensor<576x160xf32>
        %15 = tensor.empty() : tensor<49x160xf32>
        %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<49x160xf32>) -> tensor<49x160xf32>
        %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup"}
            ins(%8, %10 : tensor<49x576xf32>, tensor<576x160xf32>) outs(%16 : tensor<49x160xf32>) -> tensor<49x160xf32>
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [49, 160], strides = [1, 1]
            : tensor<49x160xf32> -> !flow.dispatch.tensor<writeonly:tensor<49x160xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[7, 32], [7, 4], [0, 0, 8]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @matmul_49x160x576
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [8 : index, 1 : index, 1 : index]
//      CHECK: func.func @matmul_49x160x576()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Large batch matmul.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @batch_matmul_4x384x384 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 64>>
    }>) {
    hal.executable.export @batch_matmul_4x384x384 layout(#pipeline_layout)
    builtin.module {
      func.func @batch_matmul_4x384x384() {
        %c0 = arith.constant 0 : index
        %c384 = arith.constant 384 : index
        %c4 = arith.constant 4 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4x384x32xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4x32x384xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<4x384x384xf32>>
        %11 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 384, 32], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<4x384x32xf32>> -> tensor<4x384x32xf32>
        %14 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4, 32, 384], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<4x32x384xf32>> -> tensor<4x32x384xf32>
        %21 = tensor.empty() : tensor<4x384x384xf32>
        %22 = linalg.fill ins(%cst : f32) outs(%21 : tensor<4x384x384xf32>) -> tensor<4x384x384xf32>
        %23 = linalg.batch_matmul {__internal_linalg_transform__ = "workgroup"}
            ins(%11, %14 : tensor<4x384x32xf32>, tensor<4x32x384xf32>) outs(%22 : tensor<4x384x384xf32>) -> tensor<4x384x384xf32>
        flow.dispatch.tensor.store %23, %2, offsets = [0, 0, 0], sizes = [4, 384, 384], strides = [1, 1, 1]
            : tensor<4x384x384xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x384x384xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 32, 128], [1, 16, 4], [0, 0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @batch_matmul_4x384x384
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [32 : index, 2 : index, 1 : index]
//      CHECK: func.func @batch_matmul_4x384x384()
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Small batch matmul.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @batch_matmul_4x8x8 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Qualcomm:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 64],
        subgroup_size = 64>>
    }>) {
    hal.executable.export @batch_matmul_4x8x8 layout(#pipeline_layout)
    builtin.module {
      func.func @batch_matmul_4x8x8() {
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4x8x32xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4x32x8xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<4x8x8xf32>>
        %11 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 8, 32], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<4x8x32xf32>> -> tensor<4x8x32xf32>
        %14 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4, 32, 8], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<4x32x8xf32>> -> tensor<4x32x8xf32>
        %21 = tensor.empty() : tensor<4x8x8xf32>
        %22 = linalg.fill ins(%cst : f32) outs(%21 : tensor<4x8x8xf32>) -> tensor<4x8x8xf32>
        %23 = linalg.batch_matmul {__internal_linalg_transform__ = "workgroup"}
            ins(%11, %14 : tensor<4x8x32xf32>, tensor<4x32x8xf32>) outs(%22 : tensor<4x8x8xf32>) -> tensor<4x8x8xf32>
        flow.dispatch.tensor.store %23, %2, offsets = [0, 0, 0], sizes = [4, 8, 8], strides = [1, 1, 1]
            : tensor<4x8x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x8x8xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 8, 8], [1, 1, 4], [0, 0, 0, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
//      CHECK: hal.executable.export public @batch_matmul_4x8x8
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [2 : index, 8 : index, 1 : index]
//      CHECK: func.func @batch_matmul_4x8x8()
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]
