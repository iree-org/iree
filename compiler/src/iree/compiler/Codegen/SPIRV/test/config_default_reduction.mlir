// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-select-lowering-strategy-pass)))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @subgroup_reduce_f32 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniformShuffle], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
       subgroup_size = 16>>
    }>) {
    hal.executable.export public @subgroup_reduce_f32 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @subgroup_reduce_f32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x512xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x512xf32>> -> tensor<2x512xf32>
        %3 = tensor.empty() : tensor<2xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2xf32>) -> tensor<2xf32>
        %5 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]
        } ins(%2 : tensor<2x512xf32>) outs(%4 : tensor<2xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %6 = arith.addf %arg1, %arg0 : f32
          linalg.yield %6 : f32
        } -> tensor<2xf32>
        flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 512]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce>
//      CHECK: hal.executable.export public @subgroup_reduce_f32
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [128 : index, 1 : index, 1 : index]
//      CHECK: func.func @subgroup_reduce_f32()
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]


// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @subgroup_reduce_f16 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader, Float16, GroupNonUniformShuffle], []>, Unknown:DiscreteGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
       subgroup_size = 64>>
    }>) {
    hal.executable.export public @subgroup_reduce_f16 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @subgroup_reduce_f16() {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x4096x4096xf16>>
        %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x4096x4096xf16>>
        %6 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [16, 4096, 4096], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x4096x4096xf16>> -> tensor<16x4096x4096xf16>
        %7 = tensor.empty() : tensor<16x4096x4096xf16>
        %8 = tensor.empty() : tensor<16x4096xf16>
        %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<16x4096xf16>) -> tensor<16x4096xf16>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6 : tensor<16x4096x4096xf16>) outs(%9 : tensor<16x4096xf16>) {
        ^bb0(%in: f16, %out: f16):
          %12 = arith.addf %in, %out : f16
          linalg.yield %12 : f16
        } -> tensor<16x4096xf16>
        %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %10 : tensor<16x4096x4096xf16>, tensor<16x4096xf16>) outs(%7 : tensor<16x4096x4096xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %12 = arith.divf %in, %in_0 : f16
          linalg.yield %12 : f16
        } -> tensor<16x4096x4096xf16>
        flow.dispatch.tensor.store %11, %5, offsets = [0, 0, 0], sizes = [16, 4096, 4096], strides = [1, 1, 1] : tensor<16x4096x4096xf16> -> !flow.dispatch.tensor<writeonly:tensor<16x4096x4096xf16>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 1], [0, 0, 512]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce>
//      CHECK: hal.executable.export public @subgroup_reduce_f16
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [64 : index, 1 : index, 1 : index]
//      CHECK: func.func @subgroup_reduce_f16()
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer, ReadOnly>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

hal.executable private @subgroup_reduce_dynamic {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {
        spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader, GroupNonUniformShuffle], []>, api=Vulkan, Unknown:DiscreteGPU, #spirv.resource_limits<
      max_compute_shared_memory_size = 65536,
      max_compute_workgroup_invocations = 1024,
      max_compute_workgroup_size = [1024, 1024, 1024],
      subgroup_size = 64>>
    }>) {
    hal.executable.export public @subgroup_reduce_dynamic ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @subgroup_reduce_dynamic() {
        %c32_i64 = arith.constant 32 : i64
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 2.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = arith.extui %0 : i32 to i64
        %3 = arith.extui %1 : i32 to i64
        %4 = arith.shli %3, %c32_i64 : i64
        %5 = arith.ori %2, %4 : i64
        %6 = arith.index_castui %5 : i64 to index
        %7 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
        %8 = flow.dispatch.workload.ordinal %6, 0 : index
        %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x?xf32>>{%8}
        %10 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [8, %8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x?xf32>>{%8} -> tensor<8x?xf32>
        %11 = tensor.empty() : tensor<8xf32>
        %12 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 64]]>} ins(%cst : f32) outs(%11 : tensor<8xf32>) -> tensor<8xf32>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%10 : tensor<8x?xf32>) outs(%12 : tensor<8xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 64]]>} {
        ^bb0(%in: f32, %out: f32):
          %14 = math.powf %in, %cst_0 : f32
          %15 = arith.addf %14, %out : f32
          linalg.yield %15 : f32
        } -> tensor<8xf32>
        flow.dispatch.tensor.store %13, %7, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
        return
      }
    }
  }
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1], [0, 64]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVSubgroupReduce>
//      CHECK: hal.executable.export public @subgroup_reduce_dynamic
// CHECK-SAME:   translation_info = #[[TRANSLATION]]
// CHECK-SAME:   workgroup_size = [64 : index, 1 : index, 1 : index]
//      CHECK: func.func @subgroup_reduce_dynamic()
//      CHECK:   linalg.generic
// CHECK-SAME:     lowering_config = #[[CONFIG]]
