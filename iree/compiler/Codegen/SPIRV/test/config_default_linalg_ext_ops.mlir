// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | FileCheck %s

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_1d_sort {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @static_1d_sort layout(#executable_layout)
    builtin.module {
      builtin.func @static_1d_sort() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:1000xi32>
        %1 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [1000], strides = [1] : !flow.dispatch.tensor<readwrite:1000xi32> -> tensor<1000xi32>
        %2 = iree_linalg_ext.sort {__internal_linalg_transform__ = "workgroup"} dimension(0) outs(%1 : tensor<1000xi32>)  {
        ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
          %3 = arith.cmpi slt, %arg0, %arg1 : i32
          iree_linalg_ext.yield %3 : i1
        } -> tensor<1000xi32>
        flow.dispatch.tensor.store %2, %0, offsets = [0], sizes = [1000], strides = [1] : tensor<1000xi32> -> !flow.dispatch.tensor<readwrite:1000xi32>
        return
      }
    }
  }
}

// Check that the workgroup count and size are (1, 1, 1) for serializing the computation.

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = [], native_vector_size = []>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = []>
//       CHECK: hal.executable.entry_point public @static_1d_sort
//  CHECK-SAME:   translation.info = #[[TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [1 : index, 1 : index, 1 : index]
//       CHECK: func @static_1d_sort()
//       CHECK:   iree_linalg_ext.sort
//  CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_3d_sort {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @static_3d_sort layout(#executable_layout)
    builtin.module {
      builtin.func @static_3d_sort() {
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

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[1, 0, 16], [1, 0, 1]{{\]}}, native_vector_size = []>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = []>
//      CHECK: hal.executable.entry_point public @static_3d_sort
// CHECK-SAME:   translation.info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [16 : index, 1 : index, 1 : index]
//      CHECK: func @static_3d_sort()
//      CHECK:   iree_linalg_ext.sort
// CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_1d_fft_stage2 {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirvfb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @static_1d_fft_stage2 layout(#executable_layout)
    builtin.module {
      builtin.func @static_1d_fft_stage2() {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %cst = arith.constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
        %cst_0 = arith.constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readwrite:32xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readwrite:32xf32>
        %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:32xf32> -> tensor<32xf32>
        %3 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [32], strides = [1] : !flow.dispatch.tensor<readwrite:32xf32> -> tensor<32xf32>
        %4:2 = iree_linalg_ext.fft {__internal_linalg_transform__ = "workgroup"} ins(%c2, %cst, %cst_0 : index, tensor<2xf32>, tensor<2xf32>) outs(%2, %3 : tensor<32xf32>, tensor<32xf32>) : tensor<32xf32>, tensor<32xf32>
        flow.dispatch.tensor.store %4#0, %0, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:32xf32>
        flow.dispatch.tensor.store %4#1, %1, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !flow.dispatch.tensor<readwrite:32xf32>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[4]{{\]}}, native_vector_size = []>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = []>
//       CHECK: hal.executable.entry_point public @static_1d_fft_stage2
//  CHECK-SAME:   translation.info = #[[TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [16 : index, 1 : index, 1 : index]
//       CHECK: func @static_1d_fft_stage2()
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @static_3d_fft_stage3 {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirvfb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @static_3d_fft_stage3 layout(#executable_layout)
    builtin.module {
      builtin.func @static_3d_fft_stage3() {
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


//   CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering.config<tile_sizes = {{\[}}[1, 1, 8]{{\]}}, native_vector_size = []>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistribute", workload_per_wg = []>
//       CHECK: hal.executable.entry_point public @static_3d_fft_stage3
//  CHECK-SAME:   translation.info = #[[TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [16 : index, 1 : index, 1 : index]
//       CHECK: func @static_3d_fft_stage3()
//       CHECK:   iree_linalg_ext.fft
//  CHECK-SAME:     lowering.config = #[[CONFIG]]

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @tensor_insert {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirvfb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @tensor_insert layout(#executable_layout)
    builtin.module {
      builtin.func @tensor_insert() {
        %offset_y = hal.interface.constant.load[0] : index
        %offset_x = hal.interface.constant.load[1] : index
        %source_size_y = hal.interface.constant.load[2] : index
        %source_size_x = hal.interface.constant.load[3] : index
        %dest_size_y = hal.interface.constant.load[4] : index
        %dest_size_x = hal.interface.constant.load[5] : index
        %source_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?xf32>{%source_size_y, %source_size_x}
        %dest_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readwrite:?x?xf32>{%dest_size_y, %dest_size_x}
        %source = flow.dispatch.tensor.load %source_binding, offsets = [0, 0], sizes = [%source_size_y, %source_size_y], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xf32>{%source_size_y, %source_size_x} -> tensor<?x?xf32>
        %dest = flow.dispatch.tensor.load %dest_binding, offsets = [0, 0], sizes = [%dest_size_y, %dest_size_x], strides = [1, 1]
            : !flow.dispatch.tensor<readwrite:?x?xf32>{%dest_size_y, %dest_size_x} -> tensor<?x?xf32>
        %insert = tensor.insert_slice %source into %dest[%offset_y, %offset_x] [%source_size_y, %source_size_x] [1, 1]
            : tensor<?x?xf32> into tensor<?x?xf32>
        flow.dispatch.tensor.store %insert, %dest_binding, offsets = [0, 0], sizes = [%dest_size_y, %dest_size_x], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>{%dest_size_y, %dest_size_x}
        return
      }
    }
  }
}
// Check that the pipeline is set to `SPIRVDistributeAndCopy`

//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistributeCopy", workload_per_wg = []>
//      CHECK: tensor.insert_slice
//  CHECK-NOT:     lowering.config

// -----

#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @tensor_extract {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirvfb", {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    }> {
    hal.executable.entry_point @tensor_extract layout(#executable_layout)
    builtin.module {
      builtin.func @tensor_extract() {
        %offset_y = hal.interface.constant.load[0] : index
        %offset_x = hal.interface.constant.load[1] : index
        %source_size_y = hal.interface.constant.load[2] : index
        %source_size_x = hal.interface.constant.load[3] : index
        %result_size_y = hal.interface.constant.load[4] : index
        %result_size_x = hal.interface.constant.load[5] : index
        %source_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:?x?xf32>{%source_size_y, %source_size_x}
        %result_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:?x?xf32>{%result_size_y, %result_size_x}
        %source = flow.dispatch.tensor.load %source_binding, offsets = [0, 0], sizes = [%source_size_y, %source_size_y], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:?x?xf32>{%source_size_y, %source_size_x} -> tensor<?x?xf32>
        %extract = tensor.extract_slice %source[%offset_y, %offset_x] [%result_size_y, %result_size_x] [1, 1]
            : tensor<?x?xf32> to tensor<?x?xf32>
        flow.dispatch.tensor.store %extract, %result_binding, offsets = [0, 0], sizes = [%result_size_y, %result_size_x], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%result_size_y, %result_size_x}
        return
      }
    }
  }
}
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation.info<"SPIRVDistributeCopy", workload_per_wg = []>
//      CHECK: tensor.extract_slice
//  CHECK-NOT:     lowering.config
