// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true})))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>

hal.executable @i4_dequant_matvec {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniform, GroupNonUniformShuffle], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
        subgroup_size = 64>>
    }> {
    hal.executable.export @i4_dequant_matvec layout(#pipeline_layout)
    builtin.module {
      func.func @i4_dequant_matvec() {
        %cst = arith.constant 0.000000e+00 : f32
        %10 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>>
        %11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>>
        %12 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>>
        %13 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<86x128xf32>>
        %14 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<4096xf32>>
        %15 = flow.dispatch.tensor.load %10, offsets = [0, 0, 0], sizes = [4096, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>> -> tensor<4096x86x128xi4>
        %16 = flow.dispatch.tensor.load %11, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>> -> tensor<4096x86xf32>
        %17 = flow.dispatch.tensor.load %12, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>> -> tensor<4096x86xf32>
        %18 = flow.dispatch.tensor.load %13, offsets = [0, 0], sizes = [86, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<86x128xf32>> -> tensor<86x128xf32>
        %19 = tensor.empty() : tensor<4096xf32>
        %20 = tensor.empty() : tensor<4096x86x128xf32>
        %21 = linalg.fill ins(%cst : f32) outs(%19 : tensor<4096xf32>) -> tensor<4096xf32>
        %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %16, %17 : tensor<4096x86x128xi4>, tensor<4096x86xf32>, tensor<4096x86xf32>) outs(%20 : tensor<4096x86x128xf32>) {
        ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
          %24 = arith.extui %in : i4 to i32
          %25 = arith.uitofp %24 : i32 to f32
          %26 = arith.subf %25, %in_1 : f32
          %27 = arith.mulf %26, %in_0 : f32
          linalg.yield %27 : f32
        } -> tensor<4096x86x128xf32>
        %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"]} ins(%18, %22 : tensor<86x128xf32>, tensor<4096x86x128xf32>) outs(%21 : tensor<4096xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %24 = arith.mulf %in, %in_0 : f32
          %25 = arith.addf %24, %out : f32
          linalg.yield %25 : f32
        } -> tensor<4096xf32>
        flow.dispatch.tensor.store %23, %14, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<4096xf32>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 2, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce>
// CHECK-LABEL: hal.executable.export public @i4_dequant_matvec
//  CHECK-SAME:   translation_info = #[[$TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [64 : index, 1 : index, 1 : index]
//       CHECK: func.func @i4_dequant_matvec()
//       CHECK:   linalg.generic
//  CHECK-SAME:     lowering_config = #[[$CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>

hal.executable @i4_dequant_matvec {
  hal.executable.variant @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniform, GroupNonUniformShuffle], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>>
    }> {
    hal.executable.export @i4_dequant_matvec layout(#pipeline_layout)
    builtin.module {
      func.func @i4_dequant_matvec() {
        %c32_i64 = arith.constant 32 : i64
        %cst = arith.constant 0.000000e+00 : f32
        %c4294967296_i64 = arith.constant 4294967296 : i64
        %22 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x32x128xi4>>
        %23 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x32x1xf32>>
        %24 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x32x1xf32>>
        %25 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<1x1x32x128xf32>>
        %26 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<1x1x4096xf32>>
        %27 = flow.dispatch.tensor.load %22, offsets = [0, 0, 0], sizes = [4096, 32, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32x128xi4>> -> tensor<4096x32x128xi4>
        %28 = flow.dispatch.tensor.load %23, offsets = [0, 0, 0], sizes = [4096, 32, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32x1xf32>> -> tensor<4096x32x1xf32>
        %29 = flow.dispatch.tensor.load %24, offsets = [0, 0, 0], sizes = [4096, 32, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x32x1xf32>> -> tensor<4096x32x1xf32>
        %30 = flow.dispatch.tensor.load %25, offsets = [0, 0, 0, 0], sizes = [1, 1, 32, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x32x128xf32>> -> tensor<1x1x32x128xf32>
        %31 = tensor.empty() : tensor<1x1x4096xf32>
        %32 = tensor.empty() : tensor<4096x32x128xf32>
        %33 = linalg.fill ins(%cst : f32) outs(%31 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
        %34 = linalg.generic {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
              affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
              affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
              affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%27, %28, %29 : tensor<4096x32x128xi4>, tensor<4096x32x1xf32>, tensor<4096x32x1xf32>) outs(%32 : tensor<4096x32x128xf32>) {
        ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
          %36 = arith.extui %in : i4 to i32
          %37 = arith.uitofp %36 : i32 to f32
          %38 = arith.subf %37, %in_1 : f32
          %39 = arith.mulf %38, %in_0 : f32
          linalg.yield %39 : f32
        } -> tensor<4096x32x128xf32>
        %35 = linalg.generic {
            indexing_maps = [
              affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
              affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
              affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>],
            iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
        ins(%30, %34 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%33 : tensor<1x1x4096xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %36 = arith.mulf %in, %in_0 : f32
          %37 = arith.addf %36, %out : f32
          linalg.yield %37 : f32
        } -> tensor<1x1x4096xf32>
        flow.dispatch.tensor.store %35, %26, offsets = [0, 0, 0], sizes = [1, 1, 4096], strides = [1, 1, 1] : tensor<1x1x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x1x4096xf32>>
        return
      }
    }
  }
}

//   CHECK-DAG: #[[$CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1, 1], [0, 0, 0, 32, 128]{{\]}}>
//   CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce>
// CHECK-LABEL: hal.executable.export public @i4_dequant_matvec
//  CHECK-SAME:   translation_info = #[[$TRANSLATION]]
//  CHECK-SAME:   workgroup_size = [1024 : index, 1 : index, 1 : index]
//       CHECK: func.func @i4_dequant_matvec()
//       CHECK:   linalg.generic
//  CHECK-SAME:     lowering_config = #[[$CONFIG]]
