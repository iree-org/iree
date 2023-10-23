// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true})))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_1d_sort {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
        subgroup_size = 16>>
    }>) {
    hal.executable.export @static_1d_sort layout(#pipeline_layout)
    builtin.module {
      func.func @static_1d_sort() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<1000xi32>>
        %1 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [1000], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<1000xi32>> -> tensor<1000xi32>
        %2 = iree_linalg_ext.sort {__internal_linalg_transform__ = "workgroup"} dimension(0) outs(%1 : tensor<1000xi32>)  {
        ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
          %3 = arith.cmpi slt, %arg0, %arg1 : i32
          iree_linalg_ext.yield %3 : i1
        } -> tensor<1000xi32>
        flow.dispatch.tensor.store %2, %0, offsets = [0], sizes = [1000], strides = [1] : tensor<1000xi32> -> !flow.dispatch.tensor<readwrite:tensor<1000xi32>>
        return
      }
    }
  }
}

// Check that the workgroup count and size are (1, 1, 1) for serializing the computation.

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = []>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute>
//       CHECK: hal.executable.export public @static_1d_sort
//  CHECK-SAME:   translation_info = #[[TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [1 : index, 1 : index, 1 : index]
//       CHECK: func.func @static_1d_sort()
//       CHECK:   iree_linalg_ext.sort
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_3d_sort {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
        subgroup_size = 16>>
    }>) {
    hal.executable.export @static_3d_sort layout(#pipeline_layout)
    builtin.module {
      func.func @static_3d_sort() {
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x32x128xi32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<64x32x128xi32>
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%0 : memref<64x32x128xi32>) outs(%1 : memref<64x32x128xi32>) {
          ^bb0(%arg4: i32, %s: i32):  // no predecessors
              linalg.yield %arg4 : i32
          }
        iree_linalg_ext.sort {__internal_linalg_transform__ = "workgroup"} dimension(1) outs(%1 : memref<64x32x128xi32>)  {
          ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
            %11 = arith.cmpi slt, %arg2, %arg3 : i32
            iree_linalg_ext.yield %11 : i1
          }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 0, 16], [1, 0, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute>
//      CHECK: hal.executable.export public @static_3d_sort
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [16 : index, 1 : index, 1 : index]
//      CHECK: func.func @static_3d_sort()
//      CHECK:   iree_linalg_ext.sort
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_1d_fft_stage2 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirvfb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
        subgroup_size = 16>>
    }>) {
    hal.executable.export @static_1d_fft_stage2 layout(#pipeline_layout)
    builtin.module {
      func.func @static_1d_fft_stage2() {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
        %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<32xf32>> -> tensor<32xf32>
        %4:2 = iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"} ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
        flow.dispatch.tensor.store %4#0, %0, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        flow.dispatch.tensor.store %4#1, %1, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:tensor<32xf32>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute>
//       CHECK: hal.executable.export public @static_1d_fft_stage2
//  CHECK-SAME:   translation_info = #[[TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [16 : index, 1 : index, 1 : index]
//       CHECK: func.func @static_1d_fft_stage2()
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_3d_fft_stage3 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirvfb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
        subgroup_size = 16>>
    }>) {
    hal.executable.export @static_3d_fft_stage3 layout(#pipeline_layout)
    builtin.module {
      func.func @static_3d_fft_stage3() {
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c32 = arith.constant 32 : index
        %cst = arith.constant dense<[1.000000e+00, 0.707106769, 6.12323426E-17, -0.707106769]> : tensor<4xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -0.707106769, -1.000000e+00, -0.707106769]> : tensor<4xf32>
        %0 = bufferization.to_memref %cst_0 : memref<4xf32>
        %1 = bufferization.to_memref %cst : memref<4xf32>
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x128x32xf32>
        %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<64x128x32xf32>
        iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"}
            ins(%c3, %1, %0 : index, memref<4xf32>, memref<4xf32>)
            outs(%2, %3 : memref<64x128x32xf32>, memref<64x128x32xf32>)
        return
      }
    }
  }
}


//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 8]{{\]}}>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute>
//       CHECK: hal.executable.export public @static_3d_fft_stage3
//  CHECK-SAME:   translation_info = #[[TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [16 : index, 1 : index, 1 : index]
//       CHECK: func.func @static_3d_fft_stage3()
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:     lowering_config = #[[CONFIG]]
