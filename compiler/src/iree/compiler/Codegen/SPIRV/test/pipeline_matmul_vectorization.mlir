// RUN: iree-opt --split-input-file --iree-gpu-test-target=valhall1 --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-spirv-configuration-pipeline), iree-codegen-linalg-to-spirv-pipeline)))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @fuse_and_vectorize_fill_matmul {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export @fuse_and_vectorize_fill_matmul layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @fuse_and_vectorize_fill_matmul() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c4096 = arith.constant 4096 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
        %8 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>> -> tensor<4096x4096xf32>
        %10 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>> -> tensor<4096x4096xf32>
        %15 = tensor.empty() : tensor<4096x4096xf32>
        %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
        %17 = linalg.matmul ins(%8, %10 : tensor<4096x4096xf32>, tensor<4096x4096xf32>) outs(%16 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
        iree_tensor_ext.dispatch.tensor.store %17, %2, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : tensor<4096x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
        return
      }
    }
  }
}

//    CHECK-LABEL: spirv.func @fuse_and_vectorize_fill_matmul
//      CHECK-NOT:   spirv.Store "StorageBuffer"
//      CHECK-NOT:   spirv.Load "StorageBuffer"
//          CHECK:   spirv.mlir.loop
//  CHECK-COUNT-8:   spirv.Load "StorageBuffer" %{{.*}} : vector<4xf32>
// CHECK-COUNT-16:   spirv.GL.Fma %{{.*}}, %{{.*}} : vector<4xf32>
//  CHECK-COUNT-4:   spirv.Store "StorageBuffer" %{{.*}}, %{{.*}} : vector<4xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @fuse_and_vectorize_matmul_add {
  hal.executable.variant @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export @fuse_and_vectorize_matmul_add layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @fuse_and_vectorize_matmul_add() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x256xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf32>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x256xf32>>
        %10 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x256xf32>> -> tensor<1024x256xf32>
        %13 = tensor.empty() : tensor<1024x256xf32>
        %15 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x512xf32>> -> tensor<1024x512xf32>
        %17 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [512, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf32>> -> tensor<512x256xf32>
        %20 = tensor.empty() : tensor<1024x256xf32>
        %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1024x256xf32>) -> tensor<1024x256xf32>
        %22 = linalg.matmul ins(%15, %17 : tensor<1024x512xf32>, tensor<512x256xf32>) outs(%21 : tensor<1024x256xf32>) -> tensor<1024x256xf32>
        %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %10 : tensor<1024x256xf32>, tensor<1024x256xf32>) outs(%13 : tensor<1024x256xf32>) {
        ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
          %24 = arith.addf %arg2, %arg3 : f32
          linalg.yield %24 : f32
        } -> tensor<1024x256xf32>
        iree_tensor_ext.dispatch.tensor.store %23, %3, offsets = [0, 0], sizes = [1024, 256], strides = [1, 1] : tensor<1024x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x256xf32>>
        return
      }
    }
  }
}

//    CHECK-LABEL: spirv.func @fuse_and_vectorize_matmul_add
//      CHECK-NOT:   spirv.Store "StorageBuffer"
//      CHECK-NOT:   spirv.Load "StorageBuffer"
//          CHECK:   spirv.mlir.loop
//  CHECK-COUNT-8:     spirv.Load "StorageBuffer" %{{.*}} : vector<4xf32>
// CHECK-COUNT-16:     spirv.GL.Fma %{{.*}}, %{{.*}} : vector<4xf32>
//          CHECK:   spirv.mlir.merge
//  CHECK-COUNT-4:   spirv.Load "StorageBuffer" %{{.*}} : vector<4xf32>
//      CHECK-NOT:   spirv.Load "StorageBuffer"
//      CHECK-NOT:   spirv.Store "StorageBuffer"
//  CHECK-COUNT-4:   spirv.FAdd %{{.*}}, %{{.*}} : vector<4xf32>
//  CHECK-COUNT-4:   spirv.Store "StorageBuffer" %{{.*}}, %{{.*}} : vector<4xf32>
