// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass)))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>
]>
hal.executable @i4_dequant_matvec_f32 {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniform, GroupNonUniformShuffle], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
        subgroup_size = 64>>
    }>) {
    hal.executable.export @i4_dequant_matvec_f32 layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @i4_dequant_matvec_f32() {
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

//         CHECK:   %[[SCAST:.+]] = vector.shape_cast %[[FOR]] : vector<1x4xf32> to vector<4xf32>
//         CHECK:   vector.reduction <add>, %[[SCAST]] : vector<4xf32> into f32
// CHECK-COUNT-6:   gpu.shuffle  xor
//         CHECK:   scf.if
//         CHECK:     vector.transfer_write
