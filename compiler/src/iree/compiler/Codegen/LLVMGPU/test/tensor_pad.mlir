// RUN: iree-opt --split-input-file  -iree-llvmgpu-tensor-pad %s --fold-memref-alias-ops -canonicalize -cse | FileCheck %s

func.func @transpose_no_align_dispatch_0_generic_48x32() {
  %c48 = arith.constant 48 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<32x48xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<48x32xf32>>
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
//       CHECK:  %[[D0:.*]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) alignment(64) : !flow.dispatch.tensor<readonly:tensor<32x48xf32>>
//       CHECK:  %[[D1:.*]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%[[C0]]) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<48x32xf32>>
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

func.func @transpose_no_align_3d_dispatch_0_generic_10x48x2048() {
  %c10 = arith.constant 10 : index
  %c48 = arith.constant 48 : index
  %c2048 = arith.constant 2048 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:10x2048x48xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:10x48x2048xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %workgroup_count_z = hal.interface.workgroup.count[2] : index
  scf.for %arg0 = %workgroup_id_z to %c10 step %workgroup_count_z {
    %2 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
    %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_y]
    scf.for %arg1 = %2 to %c48 step %3 {
      %4 = affine.min affine_map<(d0) -> (-d0 + 48, 32)>(%arg1)
      %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
      %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
      scf.for %arg2 = %5 to %c2048 step %6 {
        %7 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg1, %arg2], sizes = [1, %4, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<writeonly:10x48x2048xf32> -> tensor<1x?x32xf32>
        %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg2, %arg1], sizes = [1, 32, %4], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:10x2048x48xf32> -> tensor<1x32x?xf32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<1x32x?xf32>) outs(%7 : tensor<1x?x32xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32]]>} {
        ^bb0(%arg3: f32, %arg4: f32):
          linalg.yield %arg3 : f32
        } -> tensor<1x?x32xf32>
        flow.dispatch.tensor.store %9, %1, offsets = [%arg0, %arg1, %arg2], sizes = [1, %4, 32], strides = [1, 1, 1] : tensor<1x?x32xf32> -> !flow.dispatch.tensor<writeonly:10x48x2048xf32>
      }
    }
  }
  return
}

// CHECK-LABEL:  func.func @transpose_no_align_3d_dispatch_0_generic_10x48x2048
//       CHECK:  %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK:  %[[C10:.*]] = arith.constant 10 : index
//       CHECK:  %[[C48:.*]] = arith.constant 48 : index
//       CHECK:  %[[C2048:.*]] = arith.constant 2048 : index
//       CHECK:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[D0:.*]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) alignment(64) : !flow.dispatch.tensor<readonly:10x2048x48xf32>
//       CHECK:  %[[D1:.*]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%[[C0]]) alignment(64) : !flow.dispatch.tensor<writeonly:10x48x2048xf32>
//       CHECK:  %[[WORKGROUP_ID_X:.*]] = hal.interface.workgroup.id[0] : index
//       CHECK:  %[[WORKGROUP_COUNT_X:.*]] = hal.interface.workgroup.count[0] : index
//       CHECK:  %[[WORKGROUP_ID_Y:.*]] = hal.interface.workgroup.id[1] : index
//       CHECK:  %[[WORKGROUP_COUNT_Y:.*]] = hal.interface.workgroup.count[1] : index
//       CHECK:  %[[WORKGROUP_ID_Z:.*]] = hal.interface.workgroup.id[2] : index
//       CHECK:  %[[WORKGROUP_COUNT_Z:.*]] = hal.interface.workgroup.count[2] : index
//       CHECK:  scf.for %[[ARG0:.*]] = %[[WORKGROUP_ID_Z]] to %[[C10]] step %[[WORKGROUP_COUNT_Z]] {
//       CHECK:    %[[D2:.*]] = affine.apply #{{.*}}(){{\[}}%[[WORKGROUP_ID_Y]]]
//       CHECK:    %[[D3:.*]] = affine.apply #{{.*}}(){{\[}}%[[WORKGROUP_COUNT_Y]]]
//       CHECK:    scf.for %[[ARG1:.*]] = %[[D2]] to %[[C48]] step %[[D3]] {
//       CHECK:      %[[D4:.*]] = affine.min #{{.*}}(%[[ARG1]])
//       CHECK:      %[[D5:.*]] = affine.apply #{{.*}}(){{\[}}%[[WORKGROUP_ID_X]]]
//       CHECK:      %[[D6:.*]] = affine.apply #{{.*}}(){{\[}}%[[WORKGROUP_COUNT_X]]]
//       CHECK:      scf.for %[[ARG2:.*]] = %[[D5]] to %[[C2048]] step %[[D6]] {
//       CHECK:        %[[D7:.*]] = flow.dispatch.tensor.load %[[D1]], offsets = {{\[}}%[[ARG0]], %[[ARG1]], %[[ARG2]]], sizes = [1, %[[D4]], 32], strides = [1, 1, 1] : !flow.dispatch.tensor<writeonly:10x48x2048xf32> -> tensor<1x?x32xf32>
//       CHECK:        %[[D8:.*]] = flow.dispatch.tensor.load %[[D0]], offsets = {{\[}}%[[ARG0]], %[[ARG2]], %[[ARG1]]], sizes = [1, 32, %[[D4]]], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:10x2048x48xf32> -> tensor<1x32x?xf32>
//       CHECK:        %[[D9:.*]] = affine.apply #{{.*}}(%[[D4]])
//       CHECK:        %[[D10:.*]] = tensor.pad %[[D8]] low{{\[}}%[[C0]], %[[C0]], %[[C0]]] high{{\[}}%[[C0]], %[[C0]], %[[D9]]] {
//       CHECK:        ^bb0(%[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index):
//       CHECK:          tensor.yield %[[CST]] : f32
//       CHECK:        } : tensor<1x32x?xf32> to tensor<1x32x32xf32>
//       CHECK:        %[[D11:.*]] = tensor.pad %[[D7]] low{{\[}}%[[C0]], %[[C0]], %[[C0]]] high{{\[}}%[[C0]], %[[D9]], %[[C0]]] {
//       CHECK:        ^bb0(%[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index):
//       CHECK:          tensor.yield %[[CST]] : f32
//       CHECK:        } : tensor<1x?x32xf32> to tensor<1x32x32xf32>
//       CHECK:        %[[D12:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[D10:.*]] : tensor<1x32x32xf32>) outs(%[[D11:.*]] : tensor<1x32x32xf32>) attrs =  {lowering_config = #config} {
//       CHECK:        ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
//       CHECK:          linalg.yield %[[ARG3]] : f32
//       CHECK:        } -> tensor<1x32x32xf32>
//       CHECK:        %[[D13:.*]] = tensor.extract_slice %[[D12:.*]][0, 0, 0] [1, %[[D4]], 32] [1, 1, 1] : tensor<1x32x32xf32> to tensor<1x?x32xf32>
//       CHECK:        flow.dispatch.tensor.store %[[D13]], %[[D1]], offsets = {{\[}}%[[ARG0]], %[[ARG1]], %[[ARG2]]], sizes = [1, %[[D4]], 32], strides = [1, 1, 1] : tensor<1x?x32xf32> -> !flow.dispatch.tensor<writeonly:10x48x2048xf32>


// -----


func.func @transpose_align_dispatch_0_generic_64x32() {
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:32x64xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:64x32xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %2 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
  %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_y]
  scf.for %arg0 = %2 to %c64 step %3 {
    %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
    %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
    scf.for %arg1 = %4 to %c32 step %5 {
      %6 = flow.dispatch.tensor.load %1, offsets = [%arg0, %arg1], sizes = [32, 32], strides = [1, 1] : !flow.dispatch.tensor<writeonly:64x32xf32> -> tensor<32x32xf32>
      %7 = flow.dispatch.tensor.load %0, offsets = [%arg1, %arg0], sizes = [32, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:32x64xf32> -> tensor<32x32xf32>
      %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<32x32xf32>) outs(%6 : tensor<32x32xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 32]]>} {
      ^bb0(%arg2: f32, %arg3: f32):
        linalg.yield %arg2 : f32
      } -> tensor<32x32xf32>
      flow.dispatch.tensor.store %8, %1, offsets = [%arg0, %arg1], sizes = [32, 32], strides = [1, 1] : tensor<32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x32xf32>
    }
  }
  return
}

// CHECK-LABEL:  func.func @transpose_align_dispatch_0_generic_64x32
//   CHECK-NOT:  tensor.pad

// -----
