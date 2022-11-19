// RUN: iree-opt --iree-codegen-tile-and-distribute-winograd-input %s | FileCheck %s

func.func @_winograd_input_transform_dispatch_0() {
  %c1 = arith.constant 1 : index
  %c1280 = arith.constant 1280 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<1x10x10x1280xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<8x8x1x2x2x1280xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %2 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
  %3 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
  scf.for %arg0 = %2 to %c1 step %3 {
    %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
    %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
    scf.for %arg1 = %4 to %c1280 step %5 {
      %6 = flow.dispatch.tensor.load %1, offsets = [0, 0, %arg0, 0, 0, %arg1], sizes = [8, 8, 1, 2, 2, 64], strides = [1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<8x8x1x2x2x1280xf32>> -> tensor<8x8x1x2x2x64xf32>
      %7 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0, 0, %arg1], sizes = [1, 10, 10, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x10x10x1280xf32>> -> tensor<1x10x10x64xf32>
      %8 = iree_linalg_ext.winograd.input_transform {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64]]>} output_tile_size(6) kernel_size(3) tensor_format("nhwc") ins(%7 : tensor<1x10x10x64xf32>) outs(%6 : tensor<8x8x1x2x2x64xf32>) -> tensor<8x8x1x2x2x64xf32>
      flow.dispatch.tensor.store %8, %1, offsets = [0, 0, %arg0, 0, 0, %arg1], sizes = [8, 8, 1, 2, 2, 64], strides = [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x8x1x2x2x1280xf32>>
    }
  }
  return
}

// CHECK-DAG: #map = affine_map<()[s0] -> (s0 * 64)>
// CHECK-DAG: #map1 = affine_map<(d0) -> (d0 * 6)>
// CHECK-DAG: #map2 = affine_map<(d0) -> (-d0 + 10, 8)>
// CHECK:     func.func @_winograd_input_transform_dispatch_0() {
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[C1280:.+]] = arith.constant 1280 : index
// CHECK:       %[[C1:.+]] = arith.constant 1 : index
// CHECK:       %[[CST:.+]] = arith.constant dense<
// CHECK:       %[[CST_0:.+]] = arith.constant dense<
// CHECK:       %[[CST_1:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[C2:.+]] = arith.constant 2 : index
// CHECK:       %[[C64:.+]] = arith.constant 64 : index
// CHECK:       %[[D0:.+]] = tensor.empty() : tensor<8x8xf32>
// CHECK:       %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) alignment(64)
// CHECK-SAME:               : !flow.dispatch.tensor<readonly:tensor<1x10x10x1280xf32>>
// CHECK:       %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%[[C0]]) alignment(64)
// CHECK-SAME:               : !flow.dispatch.tensor<writeonly:tensor<8x8x1x2x2x1280xf32>>
// CHECK:       %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK:       %[[WORKGROUP_COUNT_X:.+]] = hal.interface.workgroup.count[0] : index
// CHECK:       %[[WORKGROUP_ID_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK:       %[[WORKGROUP_COUNT_Y:.+]] = hal.interface.workgroup.count[1] : index
// CHECK:       %[[D3:.+]] = affine.apply #map()[%[[WORKGROUP_ID_Y]]]
// CHECK:       %[[D4:.+]] = affine.apply #map()[%[[WORKGROUP_COUNT_Y]]]
// CHECK:       scf.for %[[ARG0:[a-zA-Z0-9_]+]] = %[[D3]] to %[[C1]] step %[[D4]] {
// CHECK:         %[[D5:.+]] = affine.apply #map()[%[[WORKGROUP_ID_X]]]
// CHECK:         %[[D6:.+]] = affine.apply #map()[%[[WORKGROUP_COUNT_X]]]
// CHECK:         scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[D5]] to %[[C1280]] step %[[D6]] {
// CHECK:           %[[D7:.+]] = flow.dispatch.tensor.load %[[D2]], offsets = [0, 0, %[[ARG0]], 0, 0, %[[ARG1]]],
// CHECK-SAME:          sizes = [8, 8, 1, 2, 2, 64], strides = [1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<8x8x1x2x2x1280xf32>>
// CHECK-SAME:          -> tensor<8x8x1x2x2x64xf32>
// CHECK:           %[[D8:.+]] = flow.dispatch.tensor.load %[[D1]], offsets = [%[[ARG0]], 0, 0, %[[ARG1]]], sizes = [1, 10, 10, 64],
// CHECK-SAME:          strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x10x10x1280xf32>> -> tensor<1x10x10x64xf32>
// CHECK:           %[[D9:.+]] = scf.for %[[ARG2:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// CHECK-SAME:          iter_args(%[[ARG3:[a-zA-Z0-9_]+]] = %[[D7]]) -> (tensor<8x8x1x2x2x64xf32>) {
// CHECK:             %[[D10:.+]] = scf.for %[[ARG4:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:            iter_args(%[[ARG5:[a-zA-Z0-9_]+]] = %[[ARG3]]) -> (tensor<8x8x1x2x2x64xf32>) {
// CHECK:               %[[D11:.+]] = affine.apply #map1(%[[ARG4]])
// CHECK:               %[[D12:.+]] = affine.min #map2(%[[D11]])
// CHECK:               %[[D13:.+]] = scf.for %[[ARG6:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:              iter_args(%[[ARG7:[a-zA-Z0-9_]+]] = %[[ARG5]]) -> (tensor<8x8x1x2x2x64xf32>) {
// CHECK:                 %[[D14:.+]] = affine.apply #map1(%[[ARG6]])
// CHECK:                 %[[D15:.+]] = affine.min #map2(%[[D14]])
// CHECK:                 %[[D16:.+]] = scf.for %[[ARG8:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C64]] step %[[C1]]
// CHECK-SAME:                iter_args(%[[ARG9:[a-zA-Z0-9_]+]] = %[[ARG7]]) -> (tensor<8x8x1x2x2x64xf32>) {
// CHECK:                   %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[D8]][%[[ARG2]], %[[D11]], %[[D14]], %[[ARG8]]]
// CHECK-SAME:                  [1, %[[D12]], %[[D15]], 1] [1, 1, 1, 1] : tensor<1x10x10x64xf32> to tensor<?x?xf32>
// CHECK:                   %[[D17:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[D0]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                   %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[EXTRACTED_SLICE]] into %[[D17]][0, 0] [%[[D12]], %[[D15]]] [1, 1]
// CHECK-SAME:                  : tensor<?x?xf32> into tensor<8x8xf32>
// CHECK:                   %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG9]][0, 0, %[[ARG2]], %[[ARG4]], %[[ARG6]], %[[ARG8]]]
// CHECK-SAME:                  [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x64xf32> to tensor<8x8xf32>
// CHECK:                   %[[D18:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[EXTRACTED_SLICE_2]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                   %[[D19:.+]] = linalg.matmul {winograd.matmul = "I x B"} ins(%[[INSERTED_SLICE]], %[[CST_0]] : tensor<8x8xf32>, tensor<8x8xf32>)
// CHECK-SAME:                  outs(%[[D18]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                   %[[D20:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[EXTRACTED_SLICE_2]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                   %[[D21:.+]] = linalg.matmul {winograd.matmul = "B' x I x B"} ins(%[[CST]], %[[D19]] : tensor<8x8xf32>, tensor<8x8xf32>)
// CHECK-SAME:                  outs(%[[D20]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                   %[[INSERTED_SLICE_3:.+]] = tensor.insert_slice %[[D21]] into %[[ARG9]][0, 0, %[[ARG2]], %[[ARG4]], %[[ARG6]], %[[ARG8]]]
// CHECK-SAME:                  [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8xf32> into tensor<8x8x1x2x2x64xf32>
// CHECK:                   scf.yield %[[INSERTED_SLICE_3]] : tensor<8x8x1x2x2x64xf32>
// CHECK:                 } {iree.spirv.distribute_dim = 0 : index}
// CHECK:                 scf.yield %[[D16]] : tensor<8x8x1x2x2x64xf32>
// CHECK:               } {iree.spirv.distribute_dim = 1 : index}
// CHECK:               scf.yield %[[D13]] : tensor<8x8x1x2x2x64xf32>
// CHECK:             } {iree.spirv.distribute_dim = 2 : index}
// CHECK:             scf.yield %[[D10]] : tensor<8x8x1x2x2x64xf32>
// CHECK:           }
// CHECK:           flow.dispatch.tensor.store %[[D9]], %[[D2]], offsets = [0, 0, %[[ARG0]], 0, 0, %[[ARG1]]], sizes = [8, 8, 1, 2, 2, 64],
// CHECK-SAME:         strides = [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x8x1x2x2x1280xf32>>
// CHECK:         }
// CHECK:       }
// CHECK:       return
// CHECK:     }
