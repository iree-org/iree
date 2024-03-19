// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-select-lowering-strategy-pass)))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer, ReadOnly>,
    #hal.descriptor_set.binding<1, storage_buffer, ReadOnly>,
    #hal.descriptor_set.binding<2, storage_buffer, ReadOnly>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @complex_executable {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformShuffle], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
       subgroup_size = 16>>
    }>) {
    hal.executable.export public @complex_view_as_real ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @complex_view_as_real() {
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1xi32>>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x50xcomplex<f32>>>
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1x32x50x2xf32>>
        %7 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x50x2xf32>>
        %8 = flow.dispatch.tensor.load %4, offsets = [0], sizes = [1], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1xi32>> -> tensor<1xi32>
        %9 = flow.dispatch.tensor.load %6, offsets = [0, 0, 0, 0, 0], sizes = [1, 1, 32, 50, 2], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1x32x50x2xf32>> -> tensor<1x1x32x50x2xf32>
        %10 = tensor.empty() : tensor<32x50x2xf32>
        %extracted = tensor.extract %8[%c0] : tensor<1xi32>
        %11 = arith.extsi %extracted : i32 to i64
        %19 = arith.index_cast %11 : i64 to index
        %20 = flow.dispatch.tensor.load %5, offsets = [%19, 0], sizes = [1, 50], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x50xcomplex<f32>>> -> tensor<50xcomplex<f32>>
        %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<50xcomplex<f32>>) outs(%10 : tensor<32x50x2xf32>) {
        ^bb0(%in: complex<f32>, %out: f32):
          %22 = linalg.index 0 : index
          %23 = linalg.index 1 : index
          %extracted_0 = tensor.extract %9[%c0, %c0, %22, %23, %c0] : tensor<1x1x32x50x2xf32>
          %extracted_1 = tensor.extract %9[%c0, %c0, %22, %23, %c1] : tensor<1x1x32x50x2xf32>
          %24 = complex.create %extracted_0, %extracted_1 : complex<f32>
          %25 = complex.mul %24, %in : complex<f32>
          %26 = complex.re %25 : complex<f32>
          %27 = complex.im %25 : complex<f32>
          %28 = linalg.index 2 : index
          %29 = arith.cmpi eq, %28, %c0 : index
          %30 = arith.select %29, %26, %27 : f32
          linalg.yield %30 : f32
        } -> tensor<32x50x2xf32>
        flow.dispatch.tensor.store %21, %7, offsets = [0, 0, 0], sizes = [32, 50, 2], strides = [1, 1, 1] : tensor<32x50x2xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x50x2xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 2, 2], [1, 1, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVBaseDistribute>
//      CHECK: hal.executable.export public @complex_view_as_real
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [2 : index, 2 : index, 4 : index]
//      CHECK: func.func @complex_view_as_real()
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]
