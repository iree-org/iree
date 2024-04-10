// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass, func.func(iree-spirv-lower-executable-target-pass))' %s | FileCheck %s

#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniform, GroupNonUniformShuffle], []>, #spirv.resource_limits<max_compute_shared_memory_size = 32768, max_compute_workgroup_invocations = 512, max_compute_workgroup_size = [512, 512, 512], subgroup_size = 64>>}>
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0)>
module {
  func.func @i4_dequant_matvec_f32() attributes {hal.executable.target = #executable_target_vulkan_spirv_fb} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<86x128xf32>>
    %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<4096xf32>>
    %5 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4096, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>> -> tensor<4096x86x128xi4>
    %6 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>> -> tensor<4096x86xf32>
    %7 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>> -> tensor<4096x86xf32>
    %8 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [86, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<86x128xf32>> -> tensor<86x128xf32>
    %9 = tensor.empty() : tensor<4096xf32>
    %10 = tensor.empty() : tensor<4096x86x128xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%9 : tensor<4096xf32>) -> tensor<4096xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6, %7 : tensor<4096x86x128xi4>, tensor<4096x86xf32>, tensor<4096x86xf32>) outs(%10 : tensor<4096x86x128xf32>) {
    ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
      %14 = arith.extui %in : i4 to i32
      %15 = arith.uitofp %14 : i32 to f32
      %16 = arith.subf %15, %in_1 : f32
      %17 = arith.mulf %16, %in_0 : f32
      linalg.yield %17 : f32
    } -> tensor<4096x86x128xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map, #map3], iterator_types = ["parallel", "reduction", "reduction"]} ins(%8, %12 : tensor<86x128xf32>, tensor<4096x86x128xf32>) outs(%11 : tensor<4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %14 = arith.mulf %in, %in_0 : f32
      %15 = arith.addf %14, %out : f32
      linalg.yield %15 : f32
    } -> tensor<4096xf32>
    flow.dispatch.tensor.store %13, %4, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<4096xf32>>
    return
  }
}

//   CHECK-LABEL: func.func @i4_dequant_matvec_f32()

//         CHECK:   %[[FOR:.+]] = scf.for %arg0 = %c0 to %c86 step %c2 iter_args({{.+}}) -> (vector<1x4xf32>)
//         CHECK:     %[[READ0:.+]] = vector.transfer_read {{.+}} : memref<4096x86x128xi4, #hal.descriptor_type<storage_buffer>>, vector<4xi4>
//         CHECK:     %[[READ1:.+]] = vector.transfer_read {{.+}} : memref<4096x86xf32, #hal.descriptor_type<storage_buffer>>, vector<1xf32>
//         CHECK:     %[[READ2:.+]] = vector.transfer_read {{.+}} : memref<4096x86xf32, #hal.descriptor_type<storage_buffer>>, vector<1xf32>
//         CHECK:     %[[READ3:.+]] = vector.transfer_read {{.+}} : memref<86x128xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//         CHECK:     %[[EXTEND:.+]] = arith.extui %[[READ0]] : vector<4xi4> to vector<4xi32>
//         CHECK:     %[[CVT:.+]] = arith.uitofp %[[EXTEND]] : vector<4xi32> to vector<4xf32>
//         CHECK:     %[[EXTRACT0:.+]] = vector.extract %[[READ1]][0] : f32 from vector<1xf32>
//         CHECK:     %[[SPLAT0:.+]] = vector.splat %[[EXTRACT0]] : vector<4xf32>
//         CHECK:     %[[SUB:.+]] = arith.subf %[[CVT]], %[[SPLAT0]] : vector<4xf32>
//         CHECK:     %[[EXTRACT1:.+]] = vector.extract %[[READ2]][0] : f32 from vector<1xf32>
//         CHECK:     %[[SPLAT1:.+]] = vector.splat %[[EXTRACT1]] : vector<4xf32>
//         CHECK:     %[[MUL0:.+]] = arith.mulf %[[SUB]], %[[SPLAT1]] : vector<4xf32>
//         CHECK:     %[[MUL1:.+]] = arith.mulf %[[READ3]], %[[MUL0]] : vector<4xf32>
//         CHECK:     %[[EXTRACT2:.+]] = vector.extract %arg1[0] : vector<4xf32> from vector<1x4xf32>
//         CHECK:     %[[ADD:.+]] = arith.addf %[[MUL1]], %[[EXTRACT2]] : vector<4xf32>
//         CHECK:     %[[BCAST:.+]] = vector.broadcast %[[ADD]] : vector<4xf32> to vector<1x4xf32>
//         CHECK:     scf.yield %[[BCAST]] : vector<1x4xf32>

//         CHECK:   %[[EXTRACT3:.+]] = vector.extract %[[FOR]][0] : vector<4xf32> from vector<1x4xf32>
//         CHECK:   %[[REDUCE:.+]] = vector.reduction <add>, %[[EXTRACT3]] : vector<4xf32> into f32
//         CHECK:   gpu.subgroup_reduce add %[[REDUCE]] : (f32) -> f32
//         CHECK:   scf.if
//         CHECK:     vector.transfer_write
