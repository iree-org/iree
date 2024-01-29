// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-spirv-annotate-winograd-loops))" %s | FileCheck %s

func.func @_wino_input_dispatch_0() {
  %c0 = arith.constant 0 : index
  %c1280 = arith.constant 1280 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant dense<0.1> : tensor<8x8xf32>
  %cst_0 = arith.constant dense<0.1> : tensor<8x8xf32>
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x10x10x1280xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x8x2x2x2x1280xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  scf.for %arg0 = %workgroup_id_y to %c2 step %workgroup_count_y {
    %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
    %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
    scf.for %arg1 = %3 to %c1280 step %4 {
      %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, %arg0, 0, 0, %arg1], sizes = [8, 8, 1, 2, 2, 32], strides = [1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<8x8x2x2x2x1280xf32>> -> tensor<8x8x1x2x2x32xf32>
      %6 = flow.dispatch.tensor.load %1, offsets = [%arg0, 0, 0, %arg1], sizes = [1, 10, 10, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x10x10x1280xf32>> -> tensor<1x10x10x32xf32>
      %7 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %5) -> (tensor<8x8x1x2x2x32xf32>) {
        %8 = affine.apply affine_map<(d0) -> (d0 * 6)>(%arg2)
        %9 = affine.min affine_map<(d0) -> (d0 * -6 + 10, 8)>(%arg2)
        %10 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> (tensor<8x8x1x2x2x32xf32>) {
          %11 = affine.apply affine_map<(d0) -> (d0 * 6)>(%arg4)
          %12 = affine.min affine_map<(d0) -> (d0 * -6 + 10, 8)>(%arg4)
          %13 = scf.for %arg6 = %c0 to %c32 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x8x1x2x2x32xf32>) {
            %extracted_slice = tensor.extract_slice %6[0, %8, %11, %arg6] [1, %9, %12, 1] [1, 1, 1, 1] : tensor<1x10x10x32xf32> to tensor<?x?xf32>
            %14 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
            %inserted_slice = tensor.insert_slice %extracted_slice into %14[0, 0] [%9, %12] [1, 1] : tensor<?x?xf32> into tensor<8x8xf32>
            %extracted_slice_2 = tensor.extract_slice %arg7[0, 0, 0, %arg2, %arg4, %arg6] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8xf32>
            %15 = linalg.fill ins(%cst_1 : f32) outs(%extracted_slice_2 : tensor<8x8xf32>) -> tensor<8x8xf32>
            %16 = linalg.matmul ins(%inserted_slice, %cst_0 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%15 : tensor<8x8xf32>) -> tensor<8x8xf32>
            %17 = linalg.matmul ins(%cst, %16 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%15 : tensor<8x8xf32>) -> tensor<8x8xf32>
            %inserted_slice_3 = tensor.insert_slice %17 into %arg7[0, 0, 0, %arg2, %arg4, %arg6] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8xf32> into tensor<8x8x1x2x2x32xf32>
            scf.yield %inserted_slice_3 : tensor<8x8x1x2x2x32xf32>
          }
          scf.yield %13 : tensor<8x8x1x2x2x32xf32>
        }
        scf.yield %10 : tensor<8x8x1x2x2x32xf32>
      }
      flow.dispatch.tensor.store %7, %2, offsets = [0, 0, %arg0, 0, 0, %arg1], sizes = [8, 8, 1, 2, 2, 32], strides = [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x8x2x2x2x1280xf32>>
    }
  }
  return
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (d0 * 6)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * -6 + 10, 8)>
// CHECK:      func.func @_wino_input_dispatch_0() {
// CHECK:        %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[C1280:.+]] = arith.constant 1280 : index
// CHECK:        %[[C2:.+]] = arith.constant 2 : index
// CHECK:        %[[CST:.+]] = arith.constant dense<1.000000e-01> : tensor<8x8xf32>
// CHECK:        %[[CST_0:.+]] = arith.constant dense<1.000000e-01> : tensor<8x8xf32>
// CHECK:        %[[CST_1:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[C32:.+]] = arith.constant 32 : index
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x8xf32>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[C0]])
// CHECK-SAME:     : !flow.dispatch.tensor<readonly:tensor<2x10x10x1280xf32>>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[C0]])
// CHECK-SAME:     : !flow.dispatch.tensor<writeonly:tensor<8x8x2x2x2x1280xf32>>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK:        %[[WORKGROUP_COUNT_X:.+]] = hal.interface.workgroup.count[0] : index
// CHECK:        %[[WORKGROUP_ID_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK:        %[[WORKGROUP_COUNT_Y:.+]] = hal.interface.workgroup.count[1] : index
// CHECK:        scf.for %[[ARG0:[a-zA-Z0-9_]+]] = %[[WORKGROUP_ID_Y]] to %[[C2]] step %[[WORKGROUP_COUNT_Y]] {
// CHECK-DAG:        %[[D3:.+]] = affine.apply #[[MAP]]()[%[[WORKGROUP_ID_X]]]
// CHECK-DAG:        %[[D4:.+]] = affine.apply #[[MAP]]()[%[[WORKGROUP_COUNT_X]]]
// CHECK:          scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[D3]] to %[[C1280]] step %[[D4]] {
// CHECK:            %[[D5:.+]] = flow.dispatch.tensor.load %[[D2]], offsets = [0, 0, %[[ARG0]], 0, 0, %[[ARG1]]], sizes
// CHECK-SAME:         = [8, 8, 1, 2, 2, 32], strides = [1, 1, 1, 1, 1, 1] :
// CHECK-SAME:         !flow.dispatch.tensor<writeonly:tensor<8x8x2x2x2x1280xf32>> -> tensor<8x8x1x2x2x32xf32>
// CHECK:            %[[D6:.+]] = flow.dispatch.tensor.load %[[D1]], offsets = [%[[ARG0]], 0, 0, %[[ARG1]]], sizes = [1,
// CHECK-SAME:         10, 10, 32], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x10x10x1280xf32>> ->
// CHECK-SAME:         tensor<1x10x10x32xf32>
// CHECK:            %[[D7:.+]] = scf.for %[[ARG2:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:         iter_args(%[[ARG3:[a-zA-Z0-9_]+]] = %[[D5]]) -> (tensor<8x8x1x2x2x32xf32>) {
// CHECK-DAG:            %[[D8:.+]] = affine.apply #[[MAP1]](%[[ARG2]])
// CHECK-DAG:            %[[D9:.+]] = affine.min #[[MAP2]](%[[ARG2]])
// CHECK:              %[[D10:.+]] = scf.for %[[ARG4:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:           iter_args(%[[ARG5:[a-zA-Z0-9_]+]] = %[[ARG3]]) -> (tensor<8x8x1x2x2x32xf32>) {
// CHECK-DAG:              %[[D11:.+]] = affine.apply #[[MAP1]](%[[ARG4]])
// CHECK-DAG:              %[[D12:.+]] = affine.min #[[MAP2]](%[[ARG4]])
// CHECK:                %[[D13:.+]] = scf.for %[[ARG6:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-SAME:             iter_args(%[[ARG7:[a-zA-Z0-9_]+]] = %[[ARG5]]) -> (tensor<8x8x1x2x2x32xf32>) {
// CHECK:                  %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[D6]][0, %[[D8]], %[[D11]], %[[ARG6]]] [1,
// CHECK-SAME:               %[[D9]], %[[D12]], 1] [1, 1, 1, 1] : tensor<1x10x10x32xf32> to tensor<?x?xf32>
// CHECK:                  %[[D14:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[D0]] : tensor<8x8xf32>) ->
// CHECK-SAME:               tensor<8x8xf32>
// CHECK:                  %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[EXTRACTED_SLICE]] into %[[D14]][0, 0]
// CHECK-SAME:               [%[[D9]], %[[D12]]] [1, 1] : tensor<?x?xf32> into tensor<8x8xf32>
// CHECK:                  %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[ARG7]][0, 0, 0, %[[ARG2]], %[[ARG4]],
// CHECK-SAME:               %[[ARG6]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to
// CHECK-SAME:               tensor<8x8xf32>
// CHECK:                  %[[D15:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[EXTRACTED_SLICE_2]] :
// CHECK-SAME:               tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                  %[[D16:.+]] = linalg.matmul ins(%[[INSERTED_SLICE]], %[[CST_0]] : tensor<8x8xf32>,
// CHECK-SAME:               tensor<8x8xf32>) outs(%[[D15]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                  %[[D17:.+]] = linalg.matmul ins(%[[CST]], %[[D16]] : tensor<8x8xf32>, tensor<8x8xf32>)
// CHECK-SAME:               outs(%[[D15]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                  %[[INSERTED_SLICE_3:.+]] = tensor.insert_slice %[[D17]] into %[[ARG7]][0, 0, 0, %[[ARG2]],
// CHECK-SAME:               %[[ARG4]], %[[ARG6]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8xf32> into
// CHECK-SAME:               tensor<8x8x1x2x2x32xf32>
// CHECK:                  scf.yield %[[INSERTED_SLICE_3]] : tensor<8x8x1x2x2x32xf32>
// CHECK:                } {iree.spirv.distribute_dim = 0 : index}
// CHECK:                scf.yield %[[D13]] : tensor<8x8x1x2x2x32xf32>
// CHECK:              } {iree.spirv.distribute_dim = 1 : index}
// CHECK:              scf.yield %[[D10]] : tensor<8x8x1x2x2x32xf32>
// CHECK:            } {iree.spirv.distribute_dim = 2 : index}
// CHECK:            flow.dispatch.tensor.store %[[D7]], %[[D2]], offsets = [0, 0, %[[ARG0]], 0, 0, %[[ARG1]]], sizes =
// CHECK-SAME:         [8, 8, 1, 2, 2, 32], strides = [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> ->
// CHECK-SAME:         !flow.dispatch.tensor<writeonly:tensor<8x8x2x2x2x1280xf32>>
// CHECK:          }
// CHECK:        }
// CHECK:        return
// CHECK:      }
