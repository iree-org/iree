// RUN: iree-opt --split-input-file --iree-gpu-test-target=volta@vulkan --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-spirv-configuration-pipeline, func.func(iree-spirv-lower-executable-target-pass)))))' %s | FileCheck %s

// TODO (MaheshRavishankar): This test should be modified to run just on the inner module/func.func. Blocked
// today since `TileAndDistributeToWorkgroups` runs the `FoldAffineMinOverWorkgroupIds` pattern that
// doesnt work without the entry point.

// Verify pipelining + multi-buffering.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#compilation = #iree_codegen.compilation_info<
    lowering_config  = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 16]]>,
    translation_info = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1], {pipeline_depth = 2, store_stage = 1}>>
#map = affine_map<(d0, d1) -> (d0, d1)>
hal.executable @matmul_f32_128x256x64 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @matmul_f32_128x256x64 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_f32_128x256x64() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x512xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x256xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x512xf32>> -> tensor<128x512xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf32>> -> tensor<512x256xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
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
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : tensor<128x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x256xf32>>
        return
      }
    }
  }
}

//       CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> ((d0 floordiv 16) mod 2)>
//       CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1]
//     CHECK-LABEL: func.func @matmul_f32_128x256x64()
//      CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//           CHECK:   %[[CST0:.+]] = arith.constant 0.000000e+00 : f32
//           CHECK:   memref.alloc() : memref<2x64x20xf32, #gpu.address_space<workgroup>>
//           CHECK:   memref.alloc() : memref<2x16x68xf32, #gpu.address_space<workgroup>>
//           CHECK:   scf.for
//           CHECK:     gpu.barrier
//           CHECK:     affine.apply #[[$MAP]]
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
//   CHECK-COUNT-8:   vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<128x256xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//   CHECK-COUNT-8:   arith.divf %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-8:   vector.transfer_write %{{.+}}, %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<128x256xf32, #hal.descriptor_type<storage_buffer>>

// -----

// Store in stage 0 of pipeline.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#compilation = #iree_codegen.compilation_info<
    lowering_config  = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 16]]>,
    translation_info = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1], {pipeline_depth = 2, store_stage = 0}>>
#map = affine_map<(d0, d1) -> (d0, d1)>
hal.executable @matmul_f32_128x256x64 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @matmul_f32_128x256x64 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_f32_128x256x64() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x512xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf32>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x256xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x512xf32>> -> tensor<128x512xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x256xf32>> -> tensor<512x256xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
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
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : tensor<128x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x256xf32>>
        return
      }
    }
  }
}

//       CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> ((d0 floordiv 16) mod 3)>
//       CHECK-DAG: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [16, 8, 1]
//     CHECK-LABEL: func.func @matmul_f32_128x256x64()
//      CHECK-SAME:     translation_info = #[[$TRANSLATION]]
//           CHECK:   %[[CST0:.+]] = arith.constant 0.000000e+00 : f32
//           CHECK:   memref.alloc() : memref<3x64x20xf32, #gpu.address_space<workgroup>>
//           CHECK:   memref.alloc() : memref<3x16x68xf32, #gpu.address_space<workgroup>>
// TODO: transfer_writes should be forwarded to the following transfer_reads
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
//       CHECK-DAG:     %[[APPLY:.+]] = affine.apply #[[$MAP]]
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
//   CHECK-COUNT-8:   vector.transfer_read %{{.+}}, %[[CST0]] {in_bounds = [true]} : memref<128x256xf32, #hal.descriptor_type<storage_buffer>>, vector<4xf32>
//   CHECK-COUNT-8:   arith.divf %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-8:   vector.transfer_write %{{.+}}, %{{.+}} {in_bounds = [true]} : vector<4xf32>, memref<128x256xf32, #hal.descriptor_type<storage_buffer>>

// -----

// Check that fused transposed consumer elementwise op does not cause extra workgroup memory allocations.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#compilation = #iree_codegen.compilation_info<
    lowering_config  = #iree_codegen.lowering_config<tile_sizes = [[64, 256, 32]]>,
    translation_info = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [32, 8, 1], {pipeline_depth = 1, store_stage = 1}>>
hal.executable @matmul_f16_4096x512x512 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @matmul_f16_4096x512x512 ordinal(0) layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_f16_4096x512x512() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x512xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x512xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512xf16>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x4096xf16>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x512xf16>> -> tensor<4096x512xf16>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x512xf16>> -> tensor<512x512xf16>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [512], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512xf16>> -> tensor<512xf16>
        %7 = tensor.empty() : tensor<512x4096xf16>
        %8 = tensor.empty() : tensor<4096x512xf16>
        %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<4096x512xf16>) -> tensor<4096x512xf16>
        %10 = linalg.matmul {compilation_info = #compilation}
          ins(%4, %5 : tensor<4096x512xf16>, tensor<512x512xf16>) outs(%9 : tensor<4096x512xf16>) -> tensor<4096x512xf16>
        %11 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%10, %6 : tensor<4096x512xf16>, tensor<512xf16>) outs(%7 : tensor<512x4096xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %12 = arith.addf %in, %in_0 : f16
          linalg.yield %12 : f16
        } -> tensor<512x4096xf16>
        iree_tensor_ext.dispatch.tensor.store %11, %3, offsets = [0, 0], sizes = [512, 4096], strides = [1, 1] : tensor<512x4096xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x4096xf16>>
        return
      }
    }
  }
}

//     CHECK-LABEL: func.func @matmul_f16_4096x512x512()
//       CHECK-NOT:   memref.alloc()
//           CHECK:   %{{.+}} = memref.alloc() : memref<32x264xf16, #gpu.address_space<workgroup>>
//           CHECK:   %{{.+}} = memref.alloc() : memref<64x40xf16, #gpu.address_space<workgroup>>
//       CHECK-NOT:   memref.alloc()
//           CHECK:   scf.for %{{.+}} = %c0 to %c480 step %c32
// CHECK-COUNT-512:     vector.fma
//           CHECK:   scf.yield
