// RUN: iree-opt --iree-codegen-enable-workgroup-specialization --pass-pipeline="builtin.module(func.func(iree-codegen-workgroup-specialization),canonicalize,cse)" --split-input-file %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [16, 4, 0], [0, 0, 64]]>
#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * -64 + 123, 64)>
#map2 = affine_map<()[s0] -> (s0 * -64 + 789, 64)>
func.func @matmul_tensors() {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<123x456xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<456x789xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<123x789xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply #map()[%workgroup_id_y]
  %4 = affine.min #map1()[%workgroup_id_y]
  %5 = affine.apply #map()[%workgroup_id_x]
  %6 = affine.min #map2()[%workgroup_id_x]
  %7 = flow.dispatch.tensor.load %0, offsets = [%3, 0], sizes = [%4, 456], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<123x456xf32>> -> tensor<?x456xf32>
  %8 = flow.dispatch.tensor.load %1, offsets = [0, %5], sizes = [456, %6], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<456x789xf32>> -> tensor<456x?xf32>
  %9 = tensor.empty(%4, %6) : tensor<?x?xf32>
  %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %11 = linalg.matmul {lowering_config = #config} ins(%7, %8 : tensor<?x456xf32>, tensor<456x?xf32>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
  flow.dispatch.tensor.store %11, %2, offsets = [%3, %5], sizes = [%4, %6], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<123x789xf32>>
  return
}


// CHECK: func.func @matmul_tensors()
// CHECK:       %[[C64:.+]] = arith.constant 64 : index
// CHECK:       %[[CMP0:.+]] = arith.cmpi eq, %{{.+}}, %[[C64]] : index
// CHECK:       %[[CMP1:.+]] = arith.cmpi eq, %{{.+}}, %[[C64]] : index
// CHECK:       %[[COND:.+]] = arith.andi %[[CMP0]], %[[CMP1]] : i1
// CHECK:       scf.if %[[COND]] {
// CHECK:         linalg.matmul
// CHECK-SAME:                  ins(%{{.+}}, %{{.+}} : tensor<64x456xf32>, tensor<456x64xf32>) outs(%{{.+}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK:       } else {
// CHECK:         linalg.matmul
// CHECK-SAME:                  ins(%{{.+}}, %{{.+}} : tensor<?x456xf32>, tensor<456x?xf32>) outs(%{{.+}} : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [16, 4, 0], [0, 0, 64]]>
#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * -64 + 123, 64)>
#map2 = affine_map<()[s0] -> (s0 * -64 + 789, 64)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
func.func @add_tensors() {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<123x789xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<123x789xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<123x789xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply #map()[%workgroup_id_y]
  %4 = affine.min #map1()[%workgroup_id_y]
  %5 = affine.apply #map()[%workgroup_id_x]
  %6 = affine.min #map2()[%workgroup_id_x]
  %7 = flow.dispatch.tensor.load %0, offsets = [%3, %5], sizes = [%4, %6], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<123x789xf32>> -> tensor<?x?xf32>
  %8 = flow.dispatch.tensor.load %1, offsets = [%3, %5], sizes = [%4, %6], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<123x789xf32>> -> tensor<?x?xf32>
  %9 = tensor.empty(%4, %6) : tensor<?x?xf32>
  %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %11 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%10 : tensor<?x?xf32>) attrs =  {lowering_config = #config} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %12 = arith.addf %in, %in_0 : f32
    linalg.yield %12 : f32
  } -> tensor<?x?xf32>
  flow.dispatch.tensor.store %11, %2, offsets = [%3, %5], sizes = [%4, %6], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<123x789xf32>>
  return
}

// CHECK: func.func @add_tensors()
// CHECK:       %[[C64:.+]] = arith.constant 64 : index
// CHECK:       %[[CMP0:.+]] = arith.cmpi eq, %{{.+}}, %[[C64]] : index
// CHECK:       %[[CMP1:.+]] = arith.cmpi eq, %{{.+}}, %[[C64]] : index
// CHECK:       %[[COND:.+]] = arith.andi %[[CMP0]], %[[CMP1]] : i1
// CHECK:       scf.if %[[COND]] {
// CHECK:         linalg.generic
// CHECK-SAME:                  ins(%{{.+}}, %{{.+}} : tensor<64x64xf32>, tensor<64x64xf32>) outs(%{{.+}} : tensor<64x64xf32>)
// CHECK:       } else {
// CHECK:         linalg.generic
// CHECK-SAME:                  ins(%{{.+}}, %{{.+}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.+}} : tensor<?x?xf32>)

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[2, 256, 4]]>
#map = affine_map<()[s0] -> (s0 * 2)>
#map1 = affine_map<()[s0] -> (s0 * 256)>
#map2 = affine_map<()[s0] -> (s0 * -256 + 30522, 256)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
func.func @unaligned_partial_loop() {
  %c512 = arith.constant 512 : index
  %c786944 = arith.constant 786944 : index
  %c265458176 = arith.constant 265458176 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c512) : !flow.dispatch.tensor<readonly:tensor<128x768xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c786944) : !flow.dispatch.tensor<readonly:tensor<768x30522xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c265458176) : !flow.dispatch.tensor<readonly:tensor<30522xf32>>
  %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x30522xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %4 = affine.apply #map()[%workgroup_id_y]
  %5 = affine.apply #map1()[%workgroup_id_x]
  %6 = affine.min #map2()[%workgroup_id_x]
  %7 = flow.dispatch.tensor.load %0, offsets = [%4, 0], sizes = [2, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x768xf32>> -> tensor<2x768xf32>
  %8 = flow.dispatch.tensor.load %1, offsets = [0, %5], sizes = [768, %6], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<768x30522xf32>> -> tensor<768x?xf32>
  %9 = tensor.empty(%6) : tensor<2x?xf32>
  %10 = linalg.fill {lowering_config = #config} ins(%cst : f32) outs(%9 : tensor<2x?xf32>) -> tensor<2x?xf32>
  %11 = linalg.matmul {lowering_config = #config} ins(%7, %8 : tensor<2x768xf32>, tensor<768x?xf32>) outs(%10 : tensor<2x?xf32>) -> tensor<2x?xf32>
  %12 = flow.dispatch.tensor.load %2, offsets = [%5], sizes = [%6], strides = [1] : !flow.dispatch.tensor<readonly:tensor<30522xf32>> -> tensor<?xf32>
  %13 = tensor.empty(%6) : tensor<2x?xf32>
  %14 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%11, %12 : tensor<2x?xf32>, tensor<?xf32>) outs(%13 : tensor<2x?xf32>) attrs =  {lowering_config = #config} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %15 = arith.addf %in, %in_0 : f32
    linalg.yield %15 : f32
  } -> tensor<2x?xf32>
  flow.dispatch.tensor.store %14, %3, offsets = [%4, %5], sizes = [2, %6], strides = [1, 1] : tensor<2x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x30522xf32>>
  return
}

// CHECK: func.func @unaligned_partial_loop()
// CHECK:       %[[C256:.+]] = arith.constant 256 : index
// CHECK:       %[[COND:.+]] = arith.cmpi eq, %{{.+}}, %[[C256]] : index
// CHECK:       scf.if %[[COND]] {
// CHECK:         linalg.matmul
// CHECK-SAME:                  ins(%{{.+}}, %{{.+}} : tensor<2x768xf32>, tensor<768x256xf32>) outs(%{{.+}} : tensor<2x256xf32>)
// CHECK:       } else {
// CHECK:         linalg.matmul
// CHECK-SAME:                  ins(%{{.+}}, %{{.+}} : tensor<2x768xf32>, tensor<768x?xf32>) outs(%{{.+}} : tensor<2x?xf32>)

