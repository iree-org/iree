// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-linalg-to-spirv-pipeline)))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#map = affine_map<(d0, d1) -> (d0, d1)>

hal.executable @matmul_f32_128x256x64 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Shader], []>, NVIDIA:DiscreteGPU, #spirv.resource_limits<
      max_compute_shared_memory_size = 49152,
      max_compute_workgroup_invocations = 1024,
      max_compute_workgroup_size = [65535, 65535, 65535],
      subgroup_size = 32>>}> {
    hal.executable.export public @matmul_f32_128x256x64 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_f32_128x256x64() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x512xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<512x256xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x256xf32>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x512xf32>> -> tensor<128x512xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x256xf32>> -> tensor<512x256xf32>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
        %7 = tensor.empty() : tensor<128x256xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %9 = linalg.matmul ins(%4, %5 : tensor<128x512xf32>, tensor<512x256xf32>) outs(%8 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
                ins(%9, %6 : tensor<128x256xf32>, tensor<128x256xf32>) outs(%7 : tensor<128x256xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
          %11 = arith.divf %arg0, %arg1 : f32
          linalg.yield %11 : f32
        } -> tensor<128x256xf32>
        flow.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : tensor<128x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x256xf32>>
        return
      }
    }
  }
}

// Default promotion requires 1024 x vector<4xf32>, however padding to avoid bank conflicts
// produces an array of size 1056. Similarly 256 gets padded to 288 x vector<4xf32>.
// CHECK-DAG: spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<1056 x vector<4xf32>>)>, Workgroup>
// CHECK-DAG: spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<288 x vector<4xf32>>)>, Workgroup>

// CHECK-LABEL: spirv.func @matmul_f32_128x256x64

//   CHECK-COUNT-5: spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>

//           CHECK: spirv.mlir.loop
//           CHECK:   spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
//   CHECK-COUNT-5:   spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//           CHECK:   spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

//  CHECK-COUNT-64:   spirv.Load "Workgroup" %{{.+}} : vector<4xf32>
// CHECK-COUNT-128:   spirv.GL.Fma %{{.+}}, %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-5:   spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//           CHECK:   spirv.mlir.merge

//           CHECK: spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
//   CHECK-COUNT-5: spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//           CHECK: spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

//  CHECK-COUNT-64: spirv.Load "Workgroup" %{{.+}} : vector<4xf32>
// CHECK-COUNT-128: spirv.GL.Fma %{{.+}}, %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-4: spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//   CHECK-COUNT-4: spirv.FDiv %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-4: spirv.Store "StorageBuffer" %{{.+}}, %{{.+}} : vector<4xf32>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#map = affine_map<(d0, d1) -> (d0, d1)>

hal.executable @matmul_f16_128x256x64 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Shader, Float16], []>, NVIDIA:DiscreteGPU, #spirv.resource_limits<
      max_compute_shared_memory_size = 49152,
      max_compute_workgroup_invocations = 1024,
      max_compute_workgroup_size = [65535, 65535, 65535],
      subgroup_size = 32>>}> {
    hal.executable.export public @matmul_f16_128x256x64 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_f16_128x256x64() {
        %cst = arith.constant 0.0 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x512xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<512x256xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<128x256xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x256xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x512xf16>> -> tensor<128x512xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x256xf16>> -> tensor<512x256xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf16>> -> tensor<128x256xf16>
        %7 = tensor.empty() : tensor<128x256xf16>
        %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<128x256xf16>) -> tensor<128x256xf16>
        %9 = linalg.matmul ins(%4, %5 : tensor<128x512xf16>, tensor<512x256xf16>) outs(%8 : tensor<128x256xf16>) -> tensor<128x256xf16>
        %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
                ins(%9, %6 : tensor<128x256xf16>, tensor<128x256xf16>) outs(%7 : tensor<128x256xf16>) {
        ^bb0(%arg0: f16, %arg1: f16, %arg2: f16):
          %11 = arith.divf %arg0, %arg1 : f16
          linalg.yield %11 : f16
        } -> tensor<128x256xf16>
        flow.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : tensor<128x256xf16> -> !flow.dispatch.tensor<writeonly:tensor<128x256xf16>>
        return
      }
    }
  }
}

// Ditto on the above.
//    1024 x vector<4xf32> -> 1056 x vector<4xf32>
//    256 x vector<4xf32> -> 320 x vector<4xf32>
// CHECK-DAG: spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<1056 x vector<4xf32>>)>, Workgroup>
// CHECK-DAG: spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<320 x vector<4xf32>>)>, Workgroup>

// CHECK-LABEL: spirv.func @matmul_f16_128x256x64

//   CHECK-COUNT-5: spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>

//           CHECK: spirv.mlir.loop
//           CHECK:   spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
//   CHECK-COUNT-5:   spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//           CHECK:   spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

//  CHECK-COUNT-64:   spirv.Load "Workgroup" %{{.+}} : vector<4xf32>
// CHECK-COUNT-512:   spirv.GL.Fma %{{.+}}, %{{.+}}, %{{.+}} : vector<4xf16>
//   CHECK-COUNT-5:   spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//           CHECK:   spirv.mlir.merge

//           CHECK: spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
//   CHECK-COUNT-5: spirv.Store "Workgroup" %{{.+}}, %{{.+}} : vector<4xf32>
//           CHECK: spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

//  CHECK-COUNT-64: spirv.Load "Workgroup" %{{.+}} : vector<4xf32>
// CHECK-COUNT-512: spirv.GL.Fma %{{.+}}, %{{.+}}, %{{.+}} : vector<4xf16>
//   CHECK-COUNT-8: spirv.Load "StorageBuffer" %{{.+}} : vector<4xf32>
//  CHECK-COUNT-16: spirv.FDiv %{{.+}}, %{{.+}} : vector<4xf16>
//   CHECK-COUNT-8: spirv.Store "StorageBuffer" %{{.+}}, %{{.+}} : vector<4xf32>

// -----

// Check scalar load/store for promotion to shared memory.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>

#user_config = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[16, 128, 16]]>,
  translation_info = <SPIRVMatmulPromoteVectorize>,
  workgroup_size = [16, 8, 1]>

hal.executable @matmul_f16_32x1280x1280 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Shader, Float16, StorageBuffer16BitAccess], []>, NVIDIA:DiscreteGPU, #spirv.resource_limits<
      max_compute_shared_memory_size = 49152,
      max_compute_workgroup_invocations = 1024,
      max_compute_workgroup_size = [65535, 65535, 65535],
      subgroup_size = 32>>}> {
    hal.executable.export public @matmul_f16_32x1280x1280 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_f16_32x1280x1280() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<32x1280xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1280x1280xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x1280xf16>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x1280xf16>> -> tensor<32x1280xf16>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x1280xf16>> -> tensor<1280x1280xf16>
        %5 = tensor.empty() : tensor<32x1280xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<32x1280xf16>) -> tensor<32x1280xf16>
        %7 = linalg.matmul {compilation_info = #user_config}
             ins(%3, %4 : tensor<32x1280xf16>, tensor<1280x1280xf16>) outs(%6 : tensor<32x1280xf16>) -> tensor<32x1280xf16>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [32, 1280], strides = [1, 1] : tensor<32x1280xf16> -> !flow.dispatch.tensor<writeonly:tensor<32x1280xf16>>
        return
      }
    }
  }
}

//   CHECK-DAG: spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<2176 x f16>)>, Workgroup>
//   CHECK-DAG: spirv.GlobalVariable @{{.+}} : !spirv.ptr<!spirv.struct<(!spirv.array<384 x f16>)>, Workgroup>

// CHECK-LABEL: spirv.func @matmul_f16_32x1280x1280

//           CHECK: spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
//   CHECK-COUNT-2: spirv.mlir.loop
//           CHECK:   spirv.Store "Workgroup" %{{.+}}, %{{.+}} : f16
//   CHECK-COUNT-2: spirv.mlir.merge
//   CHECK-COUNT-2: spirv.mlir.loop
//           CHECK:   spirv.Store "Workgroup" %{{.+}}, %{{.+}} : f16
//   CHECK-COUNT-2: spirv.mlir.merge
//           CHECK: spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>

// CHECK-COUNT-160: spirv.Load "Workgroup" %{{.+}} : f16
