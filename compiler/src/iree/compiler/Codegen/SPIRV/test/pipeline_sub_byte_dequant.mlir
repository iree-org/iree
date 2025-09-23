// RUN: iree-opt --split-input-file --iree-gpu-test-target=vp_android_baseline_2022@vulkan --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-spirv-configuration-pipeline), iree-codegen-linalg-to-spirv-pipeline)))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable @i4_dequant {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @i4_dequant layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @i4_dequant() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<131072x128xi4>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<131072xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<131072xf32>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<131072x128xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [131072, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<131072x128xi4>> -> tensor<131072x128xi4>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [131072], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<131072xf32>> -> tensor<131072xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [131072], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<131072xf32>> -> tensor<131072xf32>
        %7 = tensor.empty() : tensor<131072x128xf32>
        %8 = linalg.generic {
               indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
               iterator_types = ["parallel", "parallel"]
             } ins(%4, %5, %6 : tensor<131072x128xi4>, tensor<131072xf32>, tensor<131072xf32>) outs(%7 : tensor<131072x128xf32>) {
        ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
          %9 = arith.extui %in : i4 to i32
          %10 = arith.uitofp %9 : i32 to f32
          %11 = arith.subf %10, %in_1 : f32
          %12 = arith.mulf %11, %in_0 : f32
          linalg.yield %12 : f32
        } -> tensor<131072x128xf32>
        iree_tensor_ext.dispatch.tensor.store %8, %3, offsets = [0, 0], sizes = [131072, 128], strides = [1, 1] : tensor<131072x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<131072x128xf32>>
        return
      }
    }
  }
}

//   CHECK-LABEL: spirv.func @i4_dequant()

//         CHECK: %[[BYTE1:.+]] = spirv.VectorShuffle [0 : i32, 1 : i32] {{.*}} : vector<4xi32>, vector<4xi32> -> vector<2xi32>
//         CHECK: %[[MASKED:.+]] = spirv.BitwiseAnd %[[BYTE1]]
//         CHECK: %[[SHIFTED:.+]] = spirv.ShiftRightLogical %[[BYTE1]]
//         CHECK: %[[COPIED:.+]] = spirv.VectorShuffle [0 : i32, 2 : i32, 1 : i32, 3 : i32] %[[MASKED]], %[[SHIFTED]] : vector<2xi32>, vector<2xi32> -> vector<4xi32>
//         CHECK: %[[MASKED2:.+]] = spirv.BitwiseAnd %[[COPIED]]
//         CHECK: spirv.VectorShuffle [2 : i32, 3 : i32] {{.*}} : vector<4xi32>, vector<4xi32> -> vector<2xi32>
//         CHECK: spirv.VectorShuffle [0 : i32, 1 : i32]
//         CHECK: spirv.VectorShuffle [0 : i32, 2 : i32, 1 : i32, 3 : i32]
//         CHECK: spirv.VectorShuffle [2 : i32, 3 : i32]
//         CHECK: spirv.VectorShuffle [0 : i32, 2 : i32, 1 : i32, 3 : i32]
//     CHECK-NOT: spirv.VectorShuffle

// CHECK-COUNT-4: spirv.ConvertUToF {{.+}} : vector<4xi32> to vector<4xf32>
// CHECK-COUNT-4: spirv.FSub
// CHECK-COUNT-4: spirv.FMul
