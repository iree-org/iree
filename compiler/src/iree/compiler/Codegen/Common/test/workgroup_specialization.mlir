// RUN: iree-opt --iree-codegen-enable-workgroup-specialization --iree-codegen-workgroup-specialization --canonicalize --cse --split-input-file %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [16, 4, 0], [0, 0, 64]]>
#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0) -> (-d0 + 123, 64)>
#map2 = affine_map<(d0) -> (-d0 + 789, 64)>

func.func @matmul_tensors() {
  %c123 = arith.constant 123 : index
  %c789 = arith.constant 789 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<123x456xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<456x789xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<123x789xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %3 = affine.apply #map0()[%workgroup_id_y]
  %4 = affine.apply #map0()[%workgroup_count_y]
  scf.for %arg0 = %3 to %c123 step %4 {
    %5 = affine.min #map1(%arg0)
    %6 = affine.apply #map0()[%workgroup_id_x]
    %7 = affine.apply #map0()[%workgroup_count_x]
    scf.for %arg1 = %6 to %c789 step %7 {
      %8 = affine.min #map2(%arg1)
      %9 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%5, 456], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<123x456xf32>> -> tensor<?x456xf32>
      %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [456, %8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<456x789xf32>> -> tensor<456x?xf32>
      %11 = tensor.empty(%5, %8) : tensor<?x?xf32>
      %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %13 = linalg.matmul {lowering_config = #config} ins(%9, %10 : tensor<?x456xf32>, tensor<456x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
      flow.dispatch.tensor.store %13, %2, offsets = [%arg0, %arg1], sizes = [%5, %8], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<123x789xf32>>
    }
  }
  return
}

// CHECK: func.func @matmul_tensors()
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       %[[CMP0:.+]] = arith.cmpi eq, %{{.+}}, %c64 : index
// CHECK:       %[[CMP1:.+]] = arith.cmpi eq, %{{.+}}, %c64 : index
// CHECK:       %[[COND:.+]] = arith.andi %[[CMP0]], %[[CMP1]] : i1
// CHECK:       scf.if %[[COND]] {
// CHECK:         linalg.matmul
// CHECK-SAME:                  ins(%{{.+}}, %{{.+}} : tensor<64x456xf32>, tensor<456x64xf32>) outs(%{{.+}} : tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK:       } else {
// CHECK:         linalg.matmul
// CHECK-SAME:                  ins(%{{.+}}, %{{.+}} : tensor<?x456xf32>, tensor<456x?xf32>) outs(%{{.+}} : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [16, 4, 0], [0, 0, 64]]>
#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<(d0) -> (-d0 + 123, 64)>
#map2 = affine_map<(d0) -> (-d0 + 789, 64)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

func.func @add_tensors() {
  %c123 = arith.constant 123 : index
  %c789 = arith.constant 789 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<123x789xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<123x789xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<123x789xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %3 = affine.apply #map0()[%workgroup_id_y]
  %4 = affine.apply #map0()[%workgroup_count_y]
  scf.for %arg0 = %3 to %c123 step %4 {
    %5 = affine.min #map1(%arg0)
    %6 = affine.apply #map0()[%workgroup_id_x]
    %7 = affine.apply #map0()[%workgroup_count_x]
    scf.for %arg1 = %6 to %c789 step %7 {
      %8 = affine.min #map2(%arg1)
      %9 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg1], sizes = [%5, %8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<123x789xf32>> -> tensor<?x?xf32>
      %10 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg1], sizes = [%5, %8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<123x789xf32>> -> tensor<?x?xf32>
      %11 = tensor.empty(%5, %8) : tensor<?x?xf32>
      %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %13 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%9, %10 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) attrs =  {lowering_config = #config} {
      ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
        %14 = arith.addf %arg2, %arg3 : f32
        linalg.yield %14 : f32
      } -> tensor<?x?xf32>
      flow.dispatch.tensor.store %13, %2, offsets = [%arg0, %arg1], sizes = [%5, %8], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<123x789xf32>>
    }
  }
  return
}

// CHECK: func.func @add_tensors()
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       %[[CMP0:.+]] = arith.cmpi eq, %{{.+}}, %c64 : index
// CHECK:       %[[CMP1:.+]] = arith.cmpi eq, %{{.+}}, %c64 : index
// CHECK:       %[[COND:.+]] = arith.andi %[[CMP0]], %[[CMP1]] : i1
// CHECK:       scf.if %[[COND]] {
// CHECK:         linalg.generic
// CHECK-SAME:                  ins(%{{.+}}, %{{.+}} : tensor<64x64xf32>, tensor<64x64xf32>) outs(%{{.+}} : tensor<64x64xf32>)
// CHECK:       } else {
// CHECK:         linalg.generic
// CHECK-SAME:                  ins(%{{.+}}, %{{.+}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.+}} : tensor<?x?xf32>)

// -----

func.func @unaligned_partial_loop() {
  %c2 = arith.constant 2 : index
  %c128 = arith.constant 128 : index
  %c30522 = arith.constant 30522 : index
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
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%workgroup_id_y]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%workgroup_count_y]
  scf.for %arg0 = %4 to %c128 step %5 {
    %6 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_id_x]
    %7 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_count_x]
    scf.for %arg1 = %6 to %c30522 step %7 {
      %8 = affine.min affine_map<(d0) -> (-d0 + 30522, 256)>(%arg1)
      %9 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%c2, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x768xf32>> -> tensor<?x768xf32>
      %10 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [768, %8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<768x30522xf32>> -> tensor<768x?xf32>
      %11 = tensor.empty(%8) : tensor<2x?xf32>
      %12 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[2, 256, 4]]>} ins(%cst : f32) outs(%11 : tensor<2x?xf32>) -> tensor<2x?xf32>
      %13 = tensor.cast %9 : tensor<?x768xf32> to tensor<2x768xf32>
      %14 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[2, 256, 4]]>} ins(%13, %10 : tensor<2x768xf32>, tensor<768x?xf32>) outs(%12 : tensor<2x?xf32>) -> tensor<2x?xf32>
      %15 = flow.dispatch.tensor.load %2, offsets = [%arg1], sizes = [%8], strides = [1] : !flow.dispatch.tensor<readonly:tensor<30522xf32>> -> tensor<?xf32>
      %16 = tensor.empty(%8) : tensor<2x?xf32>
      %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %15 : tensor<2x?xf32>, tensor<?xf32>) outs(%16 : tensor<2x?xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[2, 256, 4]]>} {
      ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
        %19 = arith.addf %arg2, %arg3 : f32
        linalg.yield %19 : f32
      } -> tensor<2x?xf32>
      %18 = tensor.cast %17 : tensor<2x?xf32> to tensor<?x?xf32>
      flow.dispatch.tensor.store %18, %3, offsets = [%arg0, %arg1], sizes = [%c2, %8], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x30522xf32>>
    }
  }
  return
}

// CHECK: func.func @unaligned_partial_loop()
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       %[[COND:.+]] = arith.cmpi eq, %{{.+}}, %c256 : index
// CHECK:       scf.if %[[COND]] {
// CHECK:         linalg.matmul
// CHECK-SAME:                  ins(%{{.+}}, %{{.+}} : tensor<2x768xf32>, tensor<768x256xf32>) outs(%{{.+}} : tensor<2x256xf32>)
// CHECK:       } else {
// CHECK:         linalg.matmul
// CHECK-SAME:                  ins(%{{.+}}, %{{.+}} : tensor<2x768xf32>, tensor<768x?xf32>) outs(%{{.+}} : tensor<2x?xf32>)

