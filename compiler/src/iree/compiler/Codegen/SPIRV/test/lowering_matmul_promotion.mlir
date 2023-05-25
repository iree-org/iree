// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass)))' %s | FileCheck %s

// Verify pipelining + multi-buffering.

#compilation = #iree_codegen.compilation_info<
    lowering_config  = <tile_sizes = [[64, 64, 16]]>,
    translation_info = <SPIRVMatmulPromoteVectorize pipeline_depth = 2>,
    workgroup_size = [16, 8, 1]>
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
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Shader], []>, AMD:DiscreteGPU, #spirv.resource_limits<
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
        %9 = linalg.matmul {compilation_info = #compilation}
           ins(%4, %5 : tensor<128x512xf32>, tensor<512x256xf32>) outs(%8 : tensor<128x256xf32>) -> tensor<128x256xf32>
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

//       CHECK-DAG: #[[MAP:.+]] = affine_map<(d0) -> ((d0 floordiv 16) mod 2)>
//       CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize pipeline_depth = 2>
//           CHECK: hal.executable.export public @matmul_f32_128x256x64
//      CHECK-SAME:   translation_info = #[[TRANSLATION]]
//      CHECK-SAME:   workgroup_size = [16 : index, 8 : index, 1 : index]
//           CHECK: func.func @matmul_f32_128x256x64()
//           CHECK:   %[[CST0:.+]] = arith.constant 0.000000e+00 : f32
//           CHECK:   memref.alloc() : memref<2x64x20xf32, #gpu.address_space<workgroup>>
//           CHECK:   memref.alloc() : memref<2x16x68xf32, #gpu.address_space<workgroup>>
//           CHECK:   scf.for
//           CHECK:     gpu.barrier
//           CHECK:     affine.apply #[[MAP]]
//   CHECK-COUNT-2:     vector.transfer_write %{{.+}}, %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<2x64x20xf32, #gpu.address_space<workgroup>>
//   CHECK-COUNT-2:     vector.transfer_write %{{.+}}, %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<2x16x68xf32, #gpu.address_space<workgroup>>
//           CHECK:     gpu.barrier
//  CHECK-COUNT-32:     vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<2x64x20xf32, #gpu.address_space<workgroup>>, vector<4xf32>
//  CHECK-COUNT-16:     vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<2x16x68xf32, #gpu.address_space<workgroup>>, vector<4xf32>
// CHECK-COUNT-128:     vector.fma %{{.+}}, %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-2:     vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<128x512xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//   CHECK-COUNT-2:     vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<512x256xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:     scf.yield
//           CHECK:   gpu.barrier
//   CHECK-COUNT-2:   vector.transfer_write %{{.+}}, %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<2x64x20xf32, #gpu.address_space<workgroup>>
//   CHECK-COUNT-2:   vector.transfer_write %{{.+}}, %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<2x16x68xf32, #gpu.address_space<workgroup>>
//           CHECK:   gpu.barrier
//  CHECK-COUNT-32:   vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<2x64x20xf32, #gpu.address_space<workgroup>>, vector<4xf32>
//  CHECK-COUNT-16:   vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<2x16x68xf32, #gpu.address_space<workgroup>>, vector<4xf32>
// CHECK-COUNT-128:   vector.fma %{{.+}}, %{{.+}}, %{{.+}} : vector<4xf32>
//           CHECK:   gpu.barrier
//   CHECK-COUNT-2:   vector.transfer_write %{{.+}}, %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<2x64x20xf32, #gpu.address_space<workgroup>>
//   CHECK-COUNT-2:   vector.transfer_write %{{.+}}, %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<2x16x68xf32, #gpu.address_space<workgroup>>
//           CHECK:   gpu.barrier
//  CHECK-COUNT-32:   vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<2x64x20xf32, #gpu.address_space<workgroup>>, vector<4xf32>
//  CHECK-COUNT-16:   vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<2x16x68xf32, #gpu.address_space<workgroup>>, vector<4xf32>
// CHECK-COUNT-128:   vector.fma %{{.+}}, %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-8:   vector.transfer_write %{{.+}}, %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<128x256xf32, #hal.descriptor_type<storage_buffer>>
//  CHECK-COUNT-16:   vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<128x256xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//   CHECK-COUNT-8:   arith.divf %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-8:   vector.transfer_write %{{.+}}, %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<128x256xf32, #hal.descriptor_type<storage_buffer>>

// -----

// Store in stage 0 of pipeline.

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
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Shader], []>, AMD:DiscreteGPU, #spirv.resource_limits<
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

//       CHECK-DAG: #[[MAP:.+]] = affine_map<(d0) -> ((d0 floordiv 16) mod 3)>
//       CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize pipeline_depth = 2 store_stage = 0>
//           CHECK: hal.executable.export public @matmul_f32_128x256x64
//      CHECK-SAME:   translation_info = #[[TRANSLATION]]
//      CHECK-SAME:   workgroup_size = [16 : index, 8 : index, 1 : index]
//           CHECK: func.func @matmul_f32_128x256x64()
//           CHECK:   %[[CST0:.+]] = arith.constant 0.000000e+00 : f32
//           CHECK:   memref.alloc() : memref<3x64x20xf32, #gpu.address_space<workgroup>>
//           CHECK:   memref.alloc() : memref<3x16x68xf32, #gpu.address_space<workgroup>>
// TODO: transfer_writes should be forwarded to the following transfer_reads
//   CHECK-COUNT-8:   vector.transfer_write %{{.+}}, %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<128x256xf32, #hal.descriptor_type<storage_buffer>>
//   CHECK-COUNT-8:   vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<128x256xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:   vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<128x512xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:   vector.transfer_write %{{.+}}, %{{.+}} {__pipelining_first_stage__, in_bounds = [true]} : vector<4xf32>, memref<3x64x20xf32, #gpu.address_space<workgroup>>
//           CHECK:   vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<128x512xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:   vector.transfer_write %{{.+}}, %{{.+}} {__pipelining_first_stage__, in_bounds = [true]} : vector<4xf32>, memref<3x64x20xf32, #gpu.address_space<workgroup>>
//           CHECK:   vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<512x256xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:   vector.transfer_write %{{.+}}, %{{.+}} {__pipelining_first_stage__, in_bounds = [true]} : vector<4xf32>, memref<3x16x68xf32, #gpu.address_space<workgroup>>
//           CHECK:   vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<512x256xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:   vector.transfer_write %{{.+}}, %{{.+}} {__pipelining_first_stage__, in_bounds = [true]} : vector<4xf32>, memref<3x16x68xf32, #gpu.address_space<workgroup>>
//           CHECK:   vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<128x512xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:   vector.transfer_write %{{.+}}, %{{.+}} {__pipelining_first_stage__, in_bounds = [true]} : vector<4xf32>, memref<3x64x20xf32, #gpu.address_space<workgroup>>
//           CHECK:   vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<128x512xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:   vector.transfer_write %{{.+}}, %{{.+}} {__pipelining_first_stage__, in_bounds = [true]} : vector<4xf32>, memref<3x64x20xf32, #gpu.address_space<workgroup>>
//           CHECK:   vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<512x256xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:   vector.transfer_write %{{.+}}, %{{.+}} {__pipelining_first_stage__, in_bounds = [true]} : vector<4xf32>, memref<3x16x68xf32, #gpu.address_space<workgroup>>
//           CHECK:   vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<512x256xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:   vector.transfer_write %{{.+}}, %{{.+}} {__pipelining_first_stage__, in_bounds = [true]} : vector<4xf32>, memref<3x16x68xf32, #gpu.address_space<workgroup>>
//           CHECK:   gpu.barrier {__pipelining_first_stage__}
//           CHECK:   scf.for
//  CHECK-COUNT-32:     vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<3x64x20xf32, #gpu.address_space<workgroup>>, vector<4xf32>
//  CHECK-COUNT-16:     vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<3x16x68xf32, #gpu.address_space<workgroup>>, vector<4xf32>
// CHECK-COUNT-128:     vector.fma %{{.+}}, %{{.+}}, %{{.+}} : vector<4xf32>
//       CHECK-DAG:     %[[APPLY:.+]] = affine.apply #[[MAP]]
//       CHECK-DAG:     vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<128x512xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:     vector.transfer_write %{{.+}}, %{{.+}}[%[[APPLY]], {{.+}}] {__pipelining_first_stage__, in_bounds = [true]} : vector<4xf32>, memref<3x64x20xf32, #gpu.address_space<workgroup>>
//           CHECK:     vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<128x512xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:     vector.transfer_write %{{.+}}, %{{.+}}[%[[APPLY]], {{.+}}] {__pipelining_first_stage__, in_bounds = [true]} : vector<4xf32>, memref<3x64x20xf32, #gpu.address_space<workgroup>>
//           CHECK:     vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<512x256xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:     vector.transfer_write %{{.+}}, %{{.+}}[%[[APPLY]], {{.+}}] {__pipelining_first_stage__, in_bounds = [true]} : vector<4xf32>, memref<3x16x68xf32, #gpu.address_space<workgroup>>
//           CHECK:     vector.transfer_read %{{.+}}, %[[CST0]] {__pipelining_first_stage__, in_bounds = [true]} : memref<512x256xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//           CHECK:     vector.transfer_write %{{.+}}, %{{.+}}[%[[APPLY]], {{.+}}] {__pipelining_first_stage__, in_bounds = [true]} : vector<4xf32>, memref<3x16x68xf32, #gpu.address_space<workgroup>>
//           CHECK:     gpu.barrier {__pipelining_first_stage__}
//           CHECK:     scf.yield
//  CHECK-COUNT-32:   vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<3x64x20xf32, #gpu.address_space<workgroup>>, vector<4xf32>
//  CHECK-COUNT-16:   vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<3x16x68xf32, #gpu.address_space<workgroup>>, vector<4xf32>
// CHECK-COUNT-128:   vector.fma %{{.+}}, %{{.+}}, %{{.+}} : vector<4xf32>
//  CHECK-COUNT-32:   vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<3x64x20xf32, #gpu.address_space<workgroup>>, vector<4xf32>
//  CHECK-COUNT-16:   vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<3x16x68xf32, #gpu.address_space<workgroup>>, vector<4xf32>
// CHECK-COUNT-128:   vector.fma %{{.+}}, %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-8:   vector.transfer_write %{{.+}}, %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<128x256xf32, #hal.descriptor_type<storage_buffer>>
//  CHECK-COUNT-16:   vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<128x256xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//   CHECK-COUNT-8:   arith.divf %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-8:   vector.transfer_write %{{.+}}, %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<128x256xf32, #hal.descriptor_type<storage_buffer>>
