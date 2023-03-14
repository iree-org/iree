// RUN: iree-opt --split-input-file  -iree-llvmgpu-tensor-pad %s --fold-memref-alias-ops -canonicalize -cse | FileCheck %s

func.func @transpose_no_align_dispatch_0_generic_48x32() {
  %c48 = arith.constant 48 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<32x48xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<48x32xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %2 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
  %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_y]
  scf.for %arg0 = %2 to %c48 step %3 {
    %4 = affine.min affine_map<(d0) -> (-d0 + 48, 32)>(%arg0)
    %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
    %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
    scf.for %arg1 = %5 to %c32 step %6 {
      %7 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg1], sizes = [%4, 32], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<48x32xf32>> -> tensor<?x32xf32>
      %8 = flow.dispatch.tensor.load %0, offsets = [%arg1, %arg0], sizes = [32, %4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x48xf32>> -> tensor<32x?xf32>
      %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<32x?xf32>) outs(%7 : tensor<?x32xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32]]>} {
      ^bb0(%arg2: f32, %arg3: f32):
        linalg.yield %arg2 : f32
      } -> tensor<?x32xf32>
      flow.dispatch.tensor.store %9, %1, offsets = [%arg0, %arg1], sizes = [%4, 32], strides = [1, 1] : tensor<?x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<48x32xf32>>
    }
  }
  return
}

// CHECK-LABEL:  func.func @transpose_no_align_dispatch_0_generic_48x32
//       CHECK:  %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK:  %[[C48:.*]] = arith.constant 48 : index
//       CHECK:  %[[C32:.*]] = arith.constant 32 : index
//       CHECK:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[D0:.*]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[C0]]) : !flow.dispatch.tensor<readonly:tensor<32x48xf32>>
//       CHECK:  %[[D1:.*]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[C0]]) : !flow.dispatch.tensor<writeonly:tensor<48x32xf32>>
//       CHECK:  %[[WORKGROUP_ID_X:.*]] = hal.interface.workgroup.id[0] : index
//       CHECK:  %[[WORKGROUP_COUNT_X:.*]] = hal.interface.workgroup.count[0] : index
//       CHECK:  %[[WORKGROUP_ID_Y:.*]] = hal.interface.workgroup.id[1] : index
//       CHECK:  %[[WORKGROUP_COUNT_Y:.*]] = hal.interface.workgroup.count[1] : index
//       CHECK:  %[[D2:.*]] = affine.apply #{{.*}}(){{\[}}%[[WORKGROUP_ID_Y]]]
//       CHECK:  %[[D3:.*]] = affine.apply #{{.*}}(){{\[}}%[[WORKGROUP_COUNT_Y]]]
//       CHECK:  scf.for %[[ARG0:.*]] = %[[D2]] to %[[C48]] step %[[D3]] {
//       CHECK:    %[[D4:.*]] = affine.min #{{.*}}(%[[ARG0]])
//       CHECK:    %[[D5:.*]] = affine.apply #{{.*}}(){{\[}}%[[WORKGROUP_ID_X]]]
//       CHECK:    %[[D6:.*]] = affine.apply #{{.*}}(){{\[}}%[[WORKGROUP_COUNT_X]]]
//       CHECK:    scf.for %[[ARG1:.*]] = %[[D5]] to %[[C32]] step %[[D6]] {
//       CHECK:      %[[D7:.*]] = flow.dispatch.tensor.load %[[D1]], offsets = {{\[}}%[[ARG0]], %[[ARG1]]], sizes = {{\[}}%[[D4]], 32], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<48x32xf32>> -> tensor<?x32xf32>
//       CHECK:      %[[D8:.*]] = flow.dispatch.tensor.load %[[D0]], offsets = {{\[}}%[[ARG1]], %[[ARG0]]], sizes = [32, %[[D4]]], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x48xf32>> -> tensor<32x?xf32>
//       CHECK:      %[[D9:.*]] = affine.apply #{{.*}}(%[[D4]])
//       CHECK:      %[[D10:.*]] = tensor.pad %[[D8]] low{{\[}}%[[C0]], %[[C0]]] high{{\[}}%[[C0]], %[[D9]]] {
//       CHECK:      ^bb0(%[[ARG2:.*]]: index, %[[ARG3:.*]]: index):
//       CHECK:        tensor.yield %[[CST]] : f32
//       CHECK:      } : tensor<32x?xf32> to tensor<32x32xf32>
//       CHECK:      %[[D11:.*]] = tensor.pad %[[D7]] low{{\[}}%[[C0]], %[[C0]]] high{{\[}}%[[D9]], %[[C0]]] {
//       CHECK:      ^bb0(%[[ARG2:.*]]: index, %[[ARG3:.*]]: index):
//       CHECK:        tensor.yield %[[CST]] : f32
//       CHECK:      } : tensor<?x32xf32> to tensor<32x32xf32>
//       CHECK:      %[[D12:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel"]} ins(%[[D10:.*]] : tensor<32x32xf32>) outs(%[[D11:.*]] : tensor<32x32xf32>) attrs =  {lowering_config = #config} {
//       CHECK:      ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32):
//       CHECK:        linalg.yield %[[ARG2]] : f32
//       CHECK:      } -> tensor<32x32xf32>
//       CHECK:      %[[D13:.*]] = tensor.extract_slice %[[D12:.*]][0, 0] {{\[}}%[[D4]], 32] [1, 1] : tensor<32x32xf32> to tensor<?x32xf32>
//       CHECK:      flow.dispatch.tensor.store %[[D13]], %[[D1]], offsets = {{\[}}%[[ARG0]], %[[ARG1]]], sizes = {{\[}}%[[D4]], 32], strides = [1, 1] : tensor<?x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<48x32xf32>>

// -----

#map = affine_map<()[s0] -> (s0 * 16)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
#map2 = affine_map<(d0) -> (d0 ceildiv 2)>
#map3 = affine_map<(d0) -> (d0 floordiv 2)>
func.func @unpack_dynamic() {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = arith.index_castui %0 : i32 to index
  %5 = arith.index_castui %1 : i32 to index
  %6 = arith.index_castui %2 : i32 to index
  %7 = arith.index_castui %3 : i32 to index
  %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c64) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?x2x2xi32>>{%4, %5}
  %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %10 = affine.apply #map()[%workgroup_id_y]
  %11 = affine.apply #map()[%workgroup_count_y]
  scf.for %arg0 = %10 to %6 step %11 {
    %12 = affine.min #map1(%arg0)[%6]
    %13 = affine.apply #map()[%workgroup_id_x]
    %14 = affine.apply #map()[%workgroup_count_x]
    scf.for %arg1 = %13 to %7 step %14 {
      %15 = affine.min #map1(%arg1)[%7]
      %16 = flow.dispatch.tensor.load %9, offsets = [%arg0, %arg1], sizes = [%12, %15], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7} -> tensor<?x?xi32>
      %17 = affine.apply #map2(%12)
      %18 = affine.apply #map2(%15)
      %19 = affine.apply #map3(%arg0)
      %20 = affine.apply #map3(%arg1)
      %21 = flow.dispatch.tensor.load %8, offsets = [%19, %20, 0, 0], sizes = [%17, %18, 2, 2], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x2x2xi32>>{%4, %5} -> tensor<?x?x2x2xi32>
      %c16 = arith.constant 16 : index
      %c0_i32 = arith.constant 0 : i32
      %22 = arith.subi %c16, %12 : index
      %23 = arith.subi %c16, %15 : index
      %unpack = tensor.unpack %21 inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %16 : tensor<?x?x2x2xi32> -> tensor<?x?xi32>
      flow.dispatch.tensor.store %unpack, %9, offsets = [%arg0, %arg1], sizes = [%12, %15], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xi32>>{%6, %7}
    }
  }
  return
}
// CHECK-LABEL: func.func @unpack_dynamic
// CHECK:         %[[DEST_BUF:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK:           %[[LOAD:.+]] = flow.dispatch.tensor.load %[[DEST_BUF]]
// CHECK:           %[[PAD:.+]] = tensor.pad %[[LOAD]]
// CHECK:           %[[UNPACK:.+]] = tensor.unpack {{.+}} into %[[PAD]]
// CHECK:           %[[SLICE:.+]] = tensor.extract_slice %[[UNPACK]]
// CHECK:           flow.dispatch.tensor.store %[[SLICE]], %[[DEST_BUF]]
