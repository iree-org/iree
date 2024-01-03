// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-select-lowering-strategy-pass)))' \
// RUN:   %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable public @matmul_256x1024x128_div_add {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64, min_subgroup_size = 32, max_subgroup_size = 64>
       >}>) {
    hal.executable.export public @matmul_256x1024x128_div_add layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_256x1024x128_div_add() {
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x128xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x1024xf16>>
        %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
        %11 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>> -> tensor<256x1024xf16>
        %14 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>> -> tensor<256x1024xf16>
        %17 = tensor.empty() : tensor<256x1024xf16>
        %19 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [256, 128], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<256x128xf16>> -> tensor<256x128xf16>
        %21 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<128x1024xf16>> -> tensor<128x1024xf16>
        %24 = tensor.empty() : tensor<256x1024xf16>
        %25 = linalg.fill ins(%cst : f16) outs(%24 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %26 = linalg.matmul ins(%19, %21 : tensor<256x128xf16>, tensor<128x1024xf16>) outs(%25 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %27 = linalg.generic {
            indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]}
            ins(%26, %11, %14 : tensor<256x1024xf16>, tensor<256x1024xf16>, tensor<256x1024xf16>)
            outs(%17 : tensor<256x1024xf16>) {
          ^bb0(%arg2: f16, %arg3: f16, %arg4: f16, %arg5: f16):  // no predecessors
            %28 = arith.divf %arg2, %arg3 : f16
            %29 = arith.addf %28, %arg4 : f16
              linalg.yield %29 : f16
            } -> tensor<256x1024xf16>
        flow.dispatch.tensor.store %27, %4, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
            : tensor<256x1024xf16> -> !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 128], [32, 64], [0, 0, 32], [16, 16, 16]{{\]}}>
//  CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize pipeline_depth = 1 store_stage = 0>
//CHECK-LABEL: hal.executable.export public @matmul_256x1024x128_div_add
// CHECK-SAME:   subgroup_size = 32 : index
// CHECK-SAME:   translation_info = #[[$TRANSLATION]]
// CHECK-SAME:   workgroup_size = [64 : index, 2 : index, 1 : index]
//      CHECK: func.func @matmul_256x1024x128_div_add()
//      CHECK:   linalg.matmul
// CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable public @batch_matmul_16x128x256x512_div {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64, min_subgroup_size = 32, max_subgroup_size = 64>
       >}>) {
    hal.executable.export public @batch_matmul_16x128x256x512_div layout(#pipeline_layout)
    builtin.module {
      func.func @batch_matmul_16x128x256x512_div() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x128x512xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x512x256xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x128x256xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x128x256xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 128, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x128x512xf16>> -> tensor<16x128x512xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [16, 512, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x512x256xf16>> -> tensor<16x512x256xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [16, 128, 256], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x128x256xf16>> -> tensor<16x128x256xf16>
        %7 = tensor.empty() : tensor<16x128x256xf16>
        %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<16x128x256xf16>) -> tensor<16x128x256xf16>
        %9 = linalg.batch_matmul ins(%4, %5 : tensor<16x128x512xf16>, tensor<16x512x256xf16>) outs(%8 : tensor<16x128x256xf16>) -> tensor<16x128x256xf16>
        %10 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%9, %6 : tensor<16x128x256xf16>, tensor<16x128x256xf16>) outs(%7 : tensor<16x128x256xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %11 = arith.divf %in, %in_0 : f16
          linalg.yield %11 : f16
        } -> tensor<16x128x256xf16>
        flow.dispatch.tensor.store %10, %3, offsets = [0, 0, 0], sizes = [16, 128, 256], strides = [1, 1, 1] : tensor<16x128x256xf16> -> !flow.dispatch.tensor<writeonly:tensor<16x128x256xf16>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 128], [1, 32, 64], [0, 0, 0, 32], [1, 16, 16, 16]{{\]}}>
//  CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize pipeline_depth = 1 store_stage = 0>
//CHECK-LABEL: hal.executable.export public @batch_matmul_16x128x256x512_div
// CHECK-SAME:   subgroup_size = 32 : index
// CHECK-SAME:   translation_info = #[[$TRANSLATION]]
// CHECK-SAME:   workgroup_size = [64 : index, 2 : index, 1 : index]
//      CHECK: func.func @batch_matmul_16x128x256x512_div()
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

// Linalg.generic that is a batch matmul.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @generic_batch_matmul_32x8x512x64 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64, min_subgroup_size = 32, max_subgroup_size = 64>
     >}>) {
    hal.executable.export @generic_batch_matmul_32x8x512x64 layout(#pipeline_layout)
    builtin.module {
      func.func @generic_batch_matmul_32x8x512x64() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x32x64xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<32x64x512xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x128x512xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 32, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<128x32x64xf16>> -> tensor<128x32x64xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [32, 64, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<32x64x512xf16>> -> tensor<32x64x512xf16>
        %5 = tensor.empty() : tensor<32x128x512xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<32x128x512xf16>) -> tensor<32x128x512xf16>
        %7 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%3, %4 : tensor<128x32x64xf16>, tensor<32x64x512xf16>) outs(%6 : tensor<32x128x512xf16>)
        attrs = {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]} {
        ^bb0(%arg0: f16, %arg1: f16, %arg2: f16):
          %8 = arith.mulf %arg0, %arg1 : f16
          %9 = arith.addf %arg2, %8 : f16
          linalg.yield %9 : f16
        } -> tensor<32x128x512xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [32, 128, 512], strides = [1, 1, 1] : tensor<32x128x512xf16> -> !flow.dispatch.tensor<writeonly:tensor<32x128x512xf16>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 128], [1, 32, 64], [0, 0, 0, 32], [1, 16, 16, 16]{{\]}}>
//  CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize pipeline_depth = 1 store_stage = 0>
//CHECK-LABEL: hal.executable.export public @generic_batch_matmul_32x8x512x64
// CHECK-SAME:   subgroup_size = 32 : index
// CHECK-SAME:   translation_info = #[[$TRANSLATION]]
// CHECK-SAME:   workgroup_size = [64 : index, 2 : index, 1 : index]
//      CHECK: func.func @generic_batch_matmul_32x8x512x64()
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

// K dim size not divisble by 32.

#map = affine_map<(d0, d1) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable public @batch_matmul_16x1024x1024x80 {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64, min_subgroup_size = 32, max_subgroup_size = 64>
       >}>) {
    hal.executable.export public @batch_matmul_16x1024x1024x80 layout(#pipeline_layout)
    builtin.module {
      func.func @batch_matmul_16x1024x1024x80() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x1024x80xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x80x1024xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x1024x1024xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 1024, 80], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x1024x80xf16>> -> tensor<16x1024x80xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [16, 80, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x80x1024xf16>> -> tensor<16x80x1024xf16>
        %5 = tensor.empty() : tensor<16x1024x1024xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<16x1024x1024xf16>) -> tensor<16x1024x1024xf16>
        %7 = linalg.batch_matmul ins(%3, %4 : tensor<16x1024x80xf16>, tensor<16x80x1024xf16>) outs(%6 : tensor<16x1024x1024xf16>) -> tensor<16x1024x1024xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [16, 1024, 1024], strides = [1, 1, 1] : tensor<16x1024x1024xf16> -> !flow.dispatch.tensor<writeonly:tensor<16x1024x1024xf16>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 64, 128], [1, 32, 64], [0, 0, 0, 16], [1, 16, 16, 16]{{\]}}>
//  CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize store_stage = 0>
//CHECK-LABEL: hal.executable.export public @batch_matmul_16x1024x1024x80
// CHECK-SAME:   subgroup_size = 32 : index
// CHECK-SAME:   translation_info = #[[$TRANSLATION]]
// CHECK-SAME:   workgroup_size = [64 : index, 2 : index, 1 : index]
//      CHECK: func.func @batch_matmul_16x1024x1024x80()
//      CHECK:   linalg.batch_matmul
// CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

// Small K - not supported by cooperative matrix.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable public @matmul_256x1024x8 {
  hal.executable.variant @vulkan target(<"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixKHR],
      [SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_khr = [
          #spirv.coop_matrix_props_khr<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, acc_sat = false, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64, min_subgroup_size = 32, max_subgroup_size = 64>
       >}>) {
    hal.executable.export public @matmul_256x1024x8 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_256x1024x8() {
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x8xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<8x1024xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x8xf16>> -> tensor<256x8xf16>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x1024xf16>> -> tensor<8x1024xf16>
        %15 = tensor.empty() : tensor<256x1024xf16>
        %16 = linalg.fill ins(%cst : f16) outs(%15 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup"}
            ins(%8, %10 : tensor<256x8xf16>, tensor<8x1024xf16>) outs(%16 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
            : tensor<256x1024xf16> -> !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize pipeline_depth = 1>
// CHECK-LABEL: hal.executable.export public @matmul_256x1024x8
//   CHECK-NOT:   subgroup_size =
//  CHECK-SAME:   translation_info = #[[$TRANSLATION]]
//   CHECK-NOT:   subgroup_size =
