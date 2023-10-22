// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true})))' %s | FileCheck %s

// Odd K that forbids vectorization.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @batch_matmul_1x3x32 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }>) {
    hal.executable.export public @batch_matmul_1x3x32 layout(#pipeline_layout)
    builtin.module {
      func.func @batch_matmul_1x3x32() {
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c3 = arith.constant 3 : index
        %c1 = arith.constant 1 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x3x3xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x3x32xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x3x32xf32>>
        %11 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 3, 3], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x3x3xf32>> -> tensor<1x3x3xf32>
        %14 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 3, 32], strides = [1, 1, 1]
            : !flow.dispatch.tensor<readonly:tensor<1x3x32xf32>> -> tensor<1x3x32xf32>
        %21 = tensor.empty() : tensor<1x3x32xf32>
        %22 = linalg.fill ins(%cst : f32) outs(%21 : tensor<1x3x32xf32>) -> tensor<1x3x32xf32>
        %23 = linalg.batch_matmul {__internal_linalg_transform__ = "workgroup"}
            ins(%11, %14 : tensor<1x3x3xf32>, tensor<1x3x32xf32>) outs(%22 : tensor<1x3x32xf32>) -> tensor<1x3x32xf32>
        flow.dispatch.tensor.store %23, %2, offsets = [0, 0, 0], sizes = [1, 3, 32], strides = [1, 1, 1]
            : tensor<1x3x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x3x32xf32>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 32], [0, 1, 1]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute>
// CHECK-LABEL: hal.executable.export public @batch_matmul_1x3x32
//  CHECK-SAME:   translation_info = #[[$TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [32 : index, 1 : index, 1 : index]
//       CHECK: func.func @batch_matmul_1x3x32()
//       CHECK:   linalg.batch_matmul
//  CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

// 8-bit integers can be vectorized.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_64x16xi8 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 64>>
  }>) {
    hal.executable.export public @matmul_64x16xi8 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_64x16xi8() {
        %c0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<64x32xi8>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<32x16xi8>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<64x16xi32>>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 32], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<64x32xi8>> -> tensor<64x32xi8>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32, 16], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<32x16xi8>> -> tensor<32x16xi8>
        %15 = tensor.empty() : tensor<64x16xi32>
        %16 = linalg.fill ins(%c0_i32 : i32) outs(%15 : tensor<64x16xi32>) -> tensor<64x16xi32>
        %17 = linalg.matmul {__internal_linalg_transform__ = "workgroup"}
            ins(%8, %10 : tensor<64x32xi8>, tensor<32x16xi8>) outs(%16 : tensor<64x16xi32>) -> tensor<64x16xi32>
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [64, 16], strides = [1, 1]
            : tensor<64x16xi32> -> !flow.dispatch.tensor<writeonly:tensor<64x16xi32>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[64, 16], [2, 8], [0, 0, 8]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
// CHECK-LABEL: hal.executable.export public @matmul_64x16xi8
//  CHECK-SAME:   translation_info = #[[$TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [2 : index, 32 : index, 1 : index]
//       CHECK: func.func @matmul_64x16xi8()
//       CHECK:   linalg.matmul
//  CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

// Vectorize non-32 bit types.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_64x16xi64 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, Int64], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 64>>
  }>) {
    hal.executable.export public @matmul_64x16xi64 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_64x16xi64() {
        %c0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<64x32xi64>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<32x16xi64>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<64x16xi64>>
        %8 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 32], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<64x32xi64>> -> tensor<64x32xi64>
        %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [32, 16], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<32x16xi64>> -> tensor<32x16xi64>
        %15 = tensor.empty() : tensor<64x16xi64>
        %16 = linalg.fill ins(%c0_i32 : i32) outs(%15 : tensor<64x16xi64>) -> tensor<64x16xi64>
        %17 = linalg.matmul
            ins(%8, %10 : tensor<64x32xi64>, tensor<32x16xi64>) outs(%16 : tensor<64x16xi64>) -> tensor<64x16xi64>
        flow.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [64, 16], strides = [1, 1]
            : tensor<64x16xi64> -> !flow.dispatch.tensor<writeonly:tensor<64x16xi64>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 16], [1, 4], [0, 0, 4]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
// CHECK-LABEL: hal.executable.export public @matmul_64x16xi64
//  CHECK-SAME:   translation_info = #[[$TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [4 : index, 16 : index, 1 : index]
//       CHECK: func.func @matmul_64x16xi64()
//       CHECK:   linalg.matmul
//  CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

// Odd N that forbids vectorization.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @matmul_400x273 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 64>>
    }>) {
    hal.executable.export public @matmul_400x273 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_400x273() {
        %c0 = arith.constant 0 : index
        %c11775744 = arith.constant 11775744 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c400 = arith.constant 400 : index
        %c273 = arith.constant 273 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c11775744) : !flow.dispatch.tensor<readonly:tensor<273xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<400x576xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<576x273xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<400x273xf32>>
        %9 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [273], strides = [1] : !flow.dispatch.tensor<readonly:tensor<273xf32>> -> tensor<273xf32>
        %11 = tensor.empty() : tensor<400x273xf32>
        %13 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [400, 576], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<400x576xf32>> -> tensor<400x576xf32>
        %15 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [576, 273], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<576x273xf32>> -> tensor<576x273xf32>
        %16 = tensor.empty() : tensor<400x273xf32>
        %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<400x273xf32>) -> tensor<400x273xf32>
        %18 = linalg.matmul ins(%13, %15 : tensor<400x576xf32>, tensor<576x273xf32>) outs(%17 : tensor<400x273xf32>) -> tensor<400x273xf32>
        %19 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%9, %18 : tensor<273xf32>, tensor<400x273xf32>) outs(%11 : tensor<400x273xf32>) {
          ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
            %20 = arith.addf %arg2, %arg3 : f32
            linalg.yield %20 : f32
          } -> tensor<400x273xf32>
        flow.dispatch.tensor.store %19, %3, offsets = [0, 0], sizes = [400, 273], strides = [1, 1]
            : tensor<400x273xf32> -> !flow.dispatch.tensor<writeonly:tensor<400x273xf32>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[2, 32], [1, 1]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute>
// CHECK-LABEL: hal.executable.export public @matmul_400x273
//  CHECK-SAME:   translation_info = #[[$TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [32 : index, 2 : index, 1 : index]
//       CHECK: func.func @matmul_400x273()
//       CHECK:   linalg.matmul
//  CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

// Odd M and non-4-multiplier N

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @matmul_25x546 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 64>>
  }>) {
    hal.executable.export public @matmul_25x546 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_25x546() {
        %c0 = arith.constant 0 : index
        %c15842560 = arith.constant 15842560 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c25 = arith.constant 25 : index
        %c546 = arith.constant 546 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c15842560) : !flow.dispatch.tensor<readonly:tensor<546xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<25x512xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x546xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<25x546xf32>>
        %9 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [546], strides = [1]
            : !flow.dispatch.tensor<readonly:tensor<546xf32>> -> tensor<546xf32>
        %11 = tensor.empty() : tensor<25x546xf32>
        %13 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [25, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<25x512xf32>> -> tensor<25x512xf32>
        %15 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [512, 546], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<512x546xf32>> -> tensor<512x546xf32>
        %16 = tensor.empty() : tensor<25x546xf32>
        %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<25x546xf32>) -> tensor<25x546xf32>
        %18 = linalg.matmul ins(%13, %15 : tensor<25x512xf32>, tensor<512x546xf32>) outs(%17 : tensor<25x546xf32>) -> tensor<25x546xf32>
        %19 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%9, %18 : tensor<546xf32>, tensor<25x546xf32>) outs(%11 : tensor<25x546xf32>) attrs =  {__internal_linalg_transform__ = "workgroup"} {
          ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
            %20 = arith.addf %arg2, %arg3 : f32
            linalg.yield %20 : f32
          } -> tensor<25x546xf32>
        flow.dispatch.tensor.store %19, %3, offsets = [0, 0], sizes = [25, 546], strides = [1, 1]
            : tensor<25x546xf32> -> !flow.dispatch.tensor<writeonly:tensor<25x546xf32>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 2], [1, 1]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute>
// CHECK-LABEL: hal.executable.export public @matmul_25x546
//  CHECK-SAME:   translation_info = #[[$TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [2 : index, 32 : index, 1 : index]
//       CHECK: func.func @matmul_25x546()
//       CHECK:   linalg.matmul
//  CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

// Matmul with consumer pointwise ops

#map0 = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0)[s0] -> (s0, -d0 + 256)>
#map2 = affine_map<(d0)[s0] -> (s0, -d0 + 1024)>
#map3 = affine_map<(d0)[s0] -> (-d0 + 256, s0)>
#map4 = affine_map<(d0)[s0] -> (-d0 + 1024, s0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable private @matmul_pointwise_256x1024 {
  hal.executable.variant public @vulkan_spirv_fb target(#hal.executable.target<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }>) {
    hal.executable.export public @matmul_pointwise_256x1024 layout(#pipeline_layout)
    builtin.module {
      func.func @matmul_pointwise_256x1024() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %c256 = arith.constant 256 : index
        %c1024 = arith.constant 1024 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<256x128xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x1024xf16>>
        %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
        %11 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>> -> tensor<256x1024xf16>
        %12 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<256x1024xf16>> -> tensor<256x1024xf16>
        %13 = tensor.empty() : tensor<256x1024xf16>
        %15 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [256, 128], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<256x128xf16>> -> tensor<256x128xf16>
        %17 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<128x1024xf16>> -> tensor<128x1024xf16>
        %18 = tensor.empty() : tensor<256x1024xf16>
        %19 = linalg.fill ins(%cst : f16) outs(%18 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %20 = linalg.matmul ins(%15, %17 : tensor<256x128xf16>, tensor<128x1024xf16>) outs(%19 : tensor<256x1024xf16>) -> tensor<256x1024xf16>
        %21 = linalg.generic {
            indexing_maps = [#map5, #map5, #map5, #map5], iterator_types = ["parallel", "parallel"]}
            ins(%20, %11, %12 : tensor<256x1024xf16>, tensor<256x1024xf16>, tensor<256x1024xf16>) outs(%13 : tensor<256x1024xf16>) {
          ^bb0(%arg2: f16, %arg3: f16, %arg4: f16, %arg5: f16):  // no predecessors
            %22 = arith.divf %arg2, %arg3 : f16
            %23 = arith.subf %22, %arg4 : f16
            linalg.yield %23 : f16
          } -> tensor<256x1024xf16>
        flow.dispatch.tensor.store %21, %4, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1]
            : tensor<256x1024xf16> -> !flow.dispatch.tensor<writeonly:tensor<256x1024xf16>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 256], [8, 8], [0, 0, 8]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseVectorize>
// CHECK-LABEL: hal.executable.export public @matmul_pointwise_256x1024
//  CHECK-SAME:   translation_info = #[[$TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [32 : index, 2 : index, 1 : index]
//       CHECK: func.func @matmul_pointwise_256x1024()
//       CHECK:   linalg.matmul
//  CHECK-SAME:     lowering_config = #[[$CONFIG]]
