// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | FileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @copy_as_generic {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @copy_as_generic layout(#executable_layout)
    builtin.module {
      func.func @copy_as_generic() {
        %c0 = arith.constant 0 : index
        %d0 = hal.interface.constant.load[0] : index
        %d1 = hal.interface.constant.load[1] : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<?x?xi32>{%d0, %d1}
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<?x?xi32>{%d0, %d1}
        linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%0 : memref<?x?xi32>) outs(%1 : memref<?x?xi32>) {
            ^bb0(%arg4: i32, %s: i32):  // no predecessors
              linalg.yield %arg4 : i32
          }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 16], [1, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVDistribute>
//      CHECK: hal.executable.entry_point public @copy_as_generic
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @tensor_insert {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 64 : i32}>
    }> {
    hal.executable.entry_point @copy layout(#executable_layout)
    builtin.module {
      func.func @copy() {
        %c0 = arith.constant 0 : index
        %c224 = arith.constant 224 : index
        %c3 = arith.constant 3 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1x224x224x3xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<1x224x224x3xf32>
        linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
            ins(%0 : memref<1x224x224x3xf32>) outs(%1 : memref<1x224x224x3xf32>) {
          ^bb0(%arg4: f32, %s: f32):  // no predecessors
            linalg.yield %arg4 : f32
          }
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 2, 32, 1], [0, 1, 1, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVDistribute>
//      CHECK: hal.executable.entry_point public @copy
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Average pooling op with nice tilable input.

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @avg_pool {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 32 : i32}>
    }> {
    hal.executable.entry_point public @avg_pool layout(#executable_layout)
    builtin.module {
      func @avg_pool() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:1x24x24x8xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:1x2x2x8xf32>
        %2 = linalg.init_tensor [12, 12] : tensor<12x12xf32>
        %14 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 24, 24, 8], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:1x24x24x8xf32> -> tensor<1x24x24x8xf32>
        %20 = linalg.init_tensor [1, 2, 2, 8] : tensor<1x2x2x8xf32>
        %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
        %22 = linalg.pooling_nhwc_sum {__internal_linalg_transform__ = "workgroup", dilations = dense<1> : vector<2xi64>, strides = dense<12> : vector<2xi64>}
            ins(%14, %2 : tensor<1x24x24x8xf32>, tensor<12x12xf32>)
            outs(%21 : tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
        flow.dispatch.tensor.store %22, %1, offsets = [0, 0, 0, 0], sizes = [1, 2, 2, 8], strides = [1, 1, 1, 1]
            : tensor<1x2x2x8xf32> -> !flow.dispatch.tensor<writeonly:1x2x2x8xf32>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 2, 2, 8], [0, 1, 1, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVDistribute>
//      CHECK: hal.executable.entry_point public @avg_pool
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.pooling_nhwc_sum
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Max pooling op with odd size-1 dimension sizes.

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @max_pool {
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 32 : i32}>
    }> {
    hal.executable.entry_point public @max_pool layout(#executable_layout)
    builtin.module  {
      func @max_pool() {
        %cst = arith.constant 0xFF800000 : f32
        %c38 = arith.constant 38 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c320 = arith.constant 320 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:1x76x1x1xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:1x38x1x1xf32>
        %2 = linalg.init_tensor [2, 1] : tensor<2x1xf32>
        %13 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 76, 1, 1], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:1x76x1x1xf32> -> tensor<1x76x1x1xf32>
        %18 = linalg.init_tensor [1, 38, 1, 1] : tensor<1x38x1x1xf32>
        %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<1x38x1x1xf32>) -> tensor<1x38x1x1xf32>
        %20 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<[2, 1]> : vector<2xi64>}
            ins(%13, %2 : tensor<1x76x1x1xf32>, tensor<2x1xf32>)
            outs(%19 : tensor<1x38x1x1xf32>) -> tensor<1x38x1x1xf32>
        flow.dispatch.tensor.store %20, %1, offsets = [0, 0, 0, 0], sizes = [1, 38, 1, 1], strides = [1, 1, 1, 1]
            : tensor<1x38x1x1xf32> -> !flow.dispatch.tensor<writeonly:1x38x1x1xf32>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 32], [0, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVDistribute>
//      CHECK: hal.executable.entry_point public @max_pool
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [32 : index, 1 : index, 1 : index]
//      CHECK:   linalg.pooling_nhwc_max
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Element wise op with mismatched input and output rank.

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

hal.executable @elementwise {
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 32 : i32}>
    }> {
    hal.executable.entry_point public @elementwise layout(#executable_layout)
    builtin.module {
      func @elementwise() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c10 = arith.constant 10 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:1x10xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:10xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:10xf32>
        %9 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 10], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:1x10xf32> -> tensor<1x10xf32>
        %10 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [10], strides = [1]
            : !flow.dispatch.tensor<readonly:10xf32> -> tensor<10xf32>
        %11 = linalg.init_tensor [10] : tensor<10xf32>
        %12 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%9, %10 : tensor<1x10xf32>, tensor<10xf32>) outs(%11 : tensor<10xf32>) {
            ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
              %13 = arith.addf %arg2, %arg3 : f32
              linalg.yield %13 : f32
            } -> tensor<10xf32>
        flow.dispatch.tensor.store %12, %2, offsets = [0], sizes = [10], strides = [1] : tensor<10xf32> -> !flow.dispatch.tensor<writeonly:10xf32>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVDistribute>
//      CHECK: hal.executable.entry_point public @elementwise
// CHECK-SAME:   translation_info = #[[TRANSLATION]]

// -----

// Fused depthwise convolution and element wise ops: don't vectorize with partially active subgroups.

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#map22 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

hal.executable @dwconv_elementwise {
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 32 : i32}>
    }> {
    hal.executable.entry_point public @dwconv_elementwise layout(#executable_layout)
    builtin.module  {
      func @dwconv_elementwise() {
        %cst = arith.constant opaque<"_", "0xDEADBEEF"> : tensor<3x3x1x4xf32>
        %cst_8 = arith.constant 1.001000e+00 : f32
        %cst_9 = arith.constant 0.000000e+00 : f32
        %c18 = arith.constant 18 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c4576 = arith.constant 4576 : index
        %c6272 = arith.constant 6272 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:1x21x20x1xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:1x19x18x1x4xf32>
        %11 = linalg.init_tensor [1, 19, 18, 1, 4] : tensor<1x19x18x1x4xf32>
        %14 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 21, 20, 1], strides = [1, 1, 1, 1]
            : !flow.dispatch.tensor<readonly:1x21x20x1xf32> -> tensor<1x21x20x1xf32>
        %18 = linalg.init_tensor [1, 19, 18, 1, 4] : tensor<1x19x18x1x4xf32>
        %19 = linalg.fill ins(%cst_9 : f32) outs(%18 : tensor<1x19x18x1x4xf32>) -> tensor<1x19x18x1x4xf32>
        %20 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
            ins(%14, %cst : tensor<1x21x20x1xf32>, tensor<3x3x1x4xf32>) outs(%19 : tensor<1x19x18x1x4xf32>) -> tensor<1x19x18x1x4xf32>
        %21 = linalg.generic {
            indexing_maps = [#map22, #map22], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
            ins(%20 : tensor<1x19x18x1x4xf32>) outs(%11 : tensor<1x19x18x1x4xf32>) {
          ^bb0(%arg3: f32, %arg4: f32):
            %22 = math.sqrt %cst_8 : f32
            %23 = arith.addf %arg3, %cst_9 : f32
            linalg.yield %23 : f32
          } -> tensor<1x19x18x1x4xf32>
        flow.dispatch.tensor.store %21, %1, offsets = [0, 0, 0, 0, 0], sizes = [1, 19, 18, 1, 14], strides = [1, 1, 1, 1, 1]
            : tensor<1x19x18x1x4xf32> -> !flow.dispatch.tensor<writeonly:1x19x18x1x4xf32>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 4, 2, 0, 4], [0, 1, 1, 0, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVDistribute>
//      CHECK: hal.executable.entry_point public @dwconv_elementwise
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]
