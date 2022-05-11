// RUN: iree-opt --split-input-file --pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | FileCheck %s

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 256)>
#map2 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map3 = affine_map<(d0)[s0] -> (-d0 + 256, s0)>
#map4 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable public @matmul_256x1024x128_div_sub {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.5,
          [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixNV],
          [SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>, NVIDIA:DiscreteGPU,
          {cooperative_matrix_properties_nv = [
            {a_type = i8, b_type = i8, c_type = i32, k_size = 32 : i32,
             m_size = 8 : i32, n_size = 8 : i32, result_type = i32, scope = 3 : i32},
            {a_type = f16, b_type = f16, c_type = f16, k_size = 16 : i32,
             m_size = 16 : i32, n_size = 16 : i32, result_type = f16,
             scope = 3 : i32},
            {a_type = f16, b_type = f16, c_type = f32, k_size = 16 : i32,
             m_size = 16 : i32, n_size = 16 : i32, result_type = f32,
             scope = 3 : i32}],
           max_compute_shared_memory_size = 49152 : i32,
           max_compute_workgroup_invocations = 1024 : i32,
           max_compute_workgroup_size = dense<[2147483647, 65535, 65535]> : vector<3xi32>,
           subgroup_size = 32 : i32}>}> {
    hal.executable.entry_point public @matmul_256x1024x128_div_sub layout(#executable_layout)
    builtin.module {
      func.func @matmul_256x1024x128_div_sub() {
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:256x1024xf16>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:256x1024xf16>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:256x128xf16>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<readonly:128x1024xf16>
        %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : !flow.dispatch.tensor<writeonly:256x1024xf16>
        %11 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:256x1024xf16> -> tensor<256x1024xf16>
        %14 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:256x1024xf16> -> tensor<256x1024xf16>
        %17 = linalg.init_tensor [256, 1024] : tensor<256x1024xf16>
        %19 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [256, 128], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:256x128xf16> -> tensor<256x128xf16>
        %21 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:128x1024xf16> -> tensor<128x1024xf16>
        %24 = linalg.init_tensor [256, 1024] : tensor<256x1024xf16>
        %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %26 = linalg.matmul ins(%19, %21 : tensor<256x128xf16>, tensor<128x1024xf16>) outs(%25 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %27 = linalg.generic {
            indexing_maps = [#map5, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"]}
            ins(%26, %11, %14 : tensor<256x1024xf16>, tensor<256x1024xf16>, tensor<256x1024xf16>)
            outs(%17 : tensor<256x1024xf16>)
            attrs =  {__internal_linalg_transform__ = "workgroup"} {
          ^bb0(%arg2: f16, %arg3: f16, %arg4: f16, %arg5: f16):  // no predecessors
            %28 = arith.divf %arg2, %arg3 : f16
            %29 = arith.subf %28, %arg4 : f16
              linalg.yield %29 : f16
            } -> tensor<256x1024xf16>
        flow.dispatch.tensor.store %27, %4, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
            : tensor<256x1024xf16> -> !flow.dispatch.tensor<writeonly:256x1024xf16>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 16, 16], [16, 16, 16]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVVectorizeToCooperativeOps>
//      CHECK: hal.executable.entry_point public @matmul_256x1024x128_div_sub
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [32 : index, 1 : index, 1 : index]
//      CHECK: func @matmul_256x1024x128_div_sub()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

// Small K - not supported by cooperative matrix.

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 256)>
#map2 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map3 = affine_map<(d0)[s0] -> (-d0 + 256, s0)>
#map4 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable public @matmul_256x1024x8 {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.5,
          [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixNV],
          [SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>, NVIDIA:DiscreteGPU,
          {cooperative_matrix_properties_nv = [
            {a_type = i8, b_type = i8, c_type = i32, k_size = 32 : i32,
             m_size = 8 : i32, n_size = 8 : i32, result_type = i32, scope = 3 : i32},
            {a_type = f16, b_type = f16, c_type = f16, k_size = 16 : i32,
             m_size = 16 : i32, n_size = 16 : i32, result_type = f16,
             scope = 3 : i32},
            {a_type = f16, b_type = f16, c_type = f32, k_size = 16 : i32,
             m_size = 16 : i32, n_size = 16 : i32, result_type = f32,
             scope = 3 : i32}],
           max_compute_shared_memory_size = 49152 : i32,
           max_compute_workgroup_invocations = 1024 : i32,
           max_compute_workgroup_size = dense<[2147483647, 65535, 65535]> : vector<3xi32>,
           subgroup_size = 32 : i32}>}> {
    hal.executable.entry_point public @matmul_256x1024x8 layout(#executable_layout)
    builtin.module {
      func.func @matmul_256x1024x8() {
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:256x8xf16>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:8x1024xf16>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:256x1024xf16>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:256x8xf16> -> tensor<256x8xf16>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:8x1024xf16> -> tensor<8x1024xf16>
        %15 = linalg.init_tensor [256, 1024] : tensor<256x1024xf16>
        %16 = linalg.fill ins(%cst : f16) outs(%15 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup"}
            ins(%8, %10 : tensor<256x8xf16>, tensor<8x1024xf16>) outs(%16 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
            : tensor<256x1024xf16> -> !flow.dispatch.tensor<writeonly:256x1024xf16>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVVectorize
//       CHECK: hal.executable.entry_point public @matmul_256x1024x8
//  CHECK-SAME:   translation_info = #[[TRANSLATION]]
