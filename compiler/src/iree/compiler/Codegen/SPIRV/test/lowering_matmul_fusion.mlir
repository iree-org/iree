// RUN: iree-opt --split-input-file --iree-gpu-test-target=cdna2@vulkan --pass-pipeline='builtin.module(iree-codegen-spirv-configuration-pipeline, func.func(iree-spirv-lower-executable-target-pass))' %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 1, 32]]>
#map = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#translation = #iree_codegen.translation_info<pipeline = SPIRVMatmulPromoteVectorize workgroup_size = [32, 8, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
func.func @matmul_i4_quant_weight() {
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<86x128x2048xi4>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<86x2048xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<86x2048xi4>>
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x86x128xf32>>
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x2048xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %5 = affine.apply #map()[%workgroup_id_y]
  %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [%5, 0, 0], sizes = [%c32, 86, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x86x128xf32>> -> tensor<?x86x128xf32>
  %7 = affine.apply #map1()[%workgroup_id_x]
  %8 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, %7], sizes = [86, 128, %c128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<86x128x2048xi4>> -> tensor<86x128x?xi4>
  %9 = affine.apply #map1()[%workgroup_id_x]
  %10 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, %9], sizes = [86, %c128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<86x2048xf32>> -> tensor<86x?xf32>
  %11 = affine.apply #map1()[%workgroup_id_x]
  %12 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, %11], sizes = [86, %c128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<86x2048xi4>> -> tensor<86x?xi4>
  %13 = tensor.empty() : tensor<86x128x128xf32>
  %cast = tensor.cast %8 : tensor<86x128x?xi4> to tensor<86x128x128xi4>
  %cast_0 = tensor.cast %10 : tensor<86x?xf32> to tensor<86x128xf32>
  %cast_1 = tensor.cast %12 : tensor<86x?xi4> to tensor<86x128xi4>
  %14 = linalg.generic {indexing_maps = [#map2, #map3, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cast, %cast_0, %cast_1 : tensor<86x128x128xi4>, tensor<86x128xf32>, tensor<86x128xi4>) outs(%13 : tensor<86x128x128xf32>) {
  ^bb0(%in: i4, %in_4: f32, %in_5: i4, %out: f32):
    %20 = arith.extsi %in : i4 to i32
    %21 = arith.extsi %in_5 : i4 to i32
    %22 = arith.subi %20, %21 : i32
    %23 = arith.sitofp %22 : i32 to f32
    %24 = arith.mulf %23, %in_4 : f32
    linalg.yield %24 : f32
  } -> tensor<86x128x128xf32>
  %15 = tensor.empty() : tensor<32x128xf32>
  %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x128xf32>) -> tensor<32x128xf32>
  %cast_2 = tensor.cast %6 : tensor<?x86x128xf32> to tensor<32x86x128xf32>
  %17 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%cast_2, %14 : tensor<32x86x128xf32>, tensor<86x128x128xf32>) outs(%16 : tensor<32x128xf32>) attrs =  {compilation_info = #compilation} {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
    %20 = arith.mulf %in, %in_4 : f32
    %21 = arith.addf %out, %20 : f32
    linalg.yield %21 : f32
  } -> tensor<32x128xf32>
  %cast_3 = tensor.cast %17 : tensor<32x128xf32> to tensor<?x?xf32>
  %18 = affine.apply #map()[%workgroup_id_y]
  %19 = affine.apply #map1()[%workgroup_id_x]
  iree_tensor_ext.dispatch.tensor.store %cast_3, %4, offsets = [%18, %19], sizes = [%c32, %c128], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x2048xf32>>
  return
}

//     CHECK-LABEL: func.func @matmul_i4_quant_weight()
//           CHECK:   %[[A_ALLOC:.+]] = memref.alloc() : memref<32x1x36xf32, #gpu.address_space<workgroup>>
//           CHECK:   %[[B_ALLOC:.+]] = memref.alloc() : memref<1x32x132xf32, #gpu.address_space<workgroup>>
//           CHECK:   %[[WEIGHT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//           CHECK:   %[[SCALE_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//           CHECK:   %[[ZP_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//           CHECK:   scf.for %arg0 = %c0 to %c86 step %c1 iter_args({{.+}}) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
//           CHECK:     %[[SCALE0:.+]] = vector.transfer_read %[[SCALE_BINDING]]
//           CHECK:     %[[SCALE1:.+]] = vector.transfer_read %[[SCALE_BINDING]]
//           CHECK:     %[[ZP:.+]] = vector.transfer_read %[[ZP_BINDING]]
//           CHECK:     %[[SLICE0:.+]] = vector.extract_strided_slice %[[ZP]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xi4> to vector<4xi4>
//           CHECK:     %[[ZP_EXT0:.+]] = arith.extsi %[[SLICE0]] : vector<4xi4> to vector<4xi32>
//           CHECK:     %[[SLICE1:.+]] = vector.extract_strided_slice %[[ZP]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xi4> to vector<4xi4>
//           CHECK:     %[[ZP_EXT1:.+]] = arith.extsi %[[SLICE1]] : vector<4xi4> to vector<4xi32>

//           CHECK:     scf.for %arg5 = %c0 to %c96 step %c32 iter_args({{.+}}) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)

//           CHECK:       vector.transfer_read %[[WEIGHT_BINDING]]
//   CHECK-COUNT-2:       arith.extsi %{{.+}} : vector<4xi4> to vector<4xi32>
//           CHECK:       arith.subi %{{.+}}, %[[ZP_EXT0]] : vector<4xi32>
//           CHECK:       arith.subi %{{.+}}, %[[ZP_EXT1]] : vector<4xi32>
//   CHECK-COUNT-2:       arith.sitofp %{{.+}} : vector<4xi32> to vector<4xf32>
//           CHECK:       arith.mulf %{{.+}}, %[[SCALE0]] : vector<4xf32>
//           CHECK:       arith.mulf %{{.+}}, %[[SCALE1]] : vector<4xf32>
//     CHECK-COUNT:       vector.transfer_write %{{.+}}, %[[B_ALLOC]]

//           CHECK:       vector.transfer_read %[[WEIGHT_BINDING]]
//   CHECK-COUNT-2:       arith.extsi %{{.+}} : vector<4xi4> to vector<4xi32>
//           CHECK:       arith.subi %{{.+}}, %[[ZP_EXT0]] : vector<4xi32>
//           CHECK:       arith.subi %{{.+}}, %[[ZP_EXT1]] : vector<4xi32>
//   CHECK-COUNT-2:       arith.sitofp %{{.+}} : vector<4xi32> to vector<4xf32>
//           CHECK:       arith.mulf %{{.+}}, %[[SCALE0]] : vector<4xf32>
//           CHECK:       arith.mulf %{{.+}}, %[[SCALE1]] : vector<4xf32>
//     CHECK-COUNT:       vector.transfer_write %{{.+}}, %[[B_ALLOC]]

//           CHECK:       vector.transfer_write %{{.+}}, %[[A_ALLOC]]
//           CHECK:       gpu.barrier

//  CHECK-COUNT-32:       vector.transfer_read %[[A_ALLOC]]
//  CHECK-COUNT-32:       vector.transfer_read %[[B_ALLOC]]
// CHECK-COUNT-128:       vector.fma %{{.+}}, %{{.+}}, %{{.+}} : vector<4xf32>
//   CHECK-COUNT-2:     scf.yield
