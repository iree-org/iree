// RUN: iree-opt --iree-llvmcpu-tensor-pad --split-input-file %s | FileCheck %s

func.func @pad_for_fusion() {
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = hal.interface.constant.load[3] : i32
  %4 = arith.index_castui %0 : i32 to index
  %5 = arith.index_castui %1 : i32 to index
  %6 = arith.index_castui %2 : i32 to index
  %7 = arith.index_castui %3 : i32 to index
  %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %4}
  %9 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%5, %7}
  %10 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %11 = affine.apply affine_map<()[s0] -> (s0 * 192)>()[%workgroup_id_y]
  %12 = affine.apply affine_map<()[s0] -> (s0 * 192)>()[%workgroup_count_y]
  scf.for %arg0 = %11 to %6 step %12 {
    %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 192)>(%arg0)[%6]
    %14 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
    %15 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
    scf.for %arg1 = %14 to %7 step %15 {
      %16 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 128)>(%arg1)[%7]
      %17 = flow.dispatch.tensor.load %10, offsets = [%arg0, %arg1], sizes = [%13, %16], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
      %18 = flow.dispatch.tensor.load %8, offsets = [%arg0, 0], sizes = [%13, %4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %4} -> tensor<?x?xf32>
      %19 = flow.dispatch.tensor.load %9, offsets = [0, %arg1], sizes = [%4, %16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%5, %7} -> tensor<?x?xf32>
      %20 = scf.for %arg2 = %c0 to %13 step %c8 iter_args(%arg3 = %17) -> (tensor<?x?xf32>) {
        %21 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 8)>(%arg2)[%13]
        %22 = scf.for %arg4 = %c0 to %16 step %c32 iter_args(%arg5 = %arg3) -> (tensor<?x?xf32>) {
          %23 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 32)>(%arg4)[%16]
          %extracted_slice = tensor.extract_slice %18[%arg2, 0] [%21, %4] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %extracted_slice_0 = tensor.extract_slice %19[0, %arg4] [%4, %23] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %extracted_slice_1 = tensor.extract_slice %arg5[%arg2, %arg4] [%21, %23] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %24 = linalg.fill ins(%cst : f32) outs(%extracted_slice_1 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %25 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[192, 128, 0], [8, 32, 0], [0, 0, 16]]>} ins(%extracted_slice, %extracted_slice_0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%24 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%25 : tensor<?x?xf32>) {
          ^bb0(%out: f32):
            %27 = math.exp %out : f32
            linalg.yield %27 : f32
          } -> tensor<?x?xf32>
          %inserted_slice = tensor.insert_slice %26 into %arg5[%arg2, %arg4] [%21, %23] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
          scf.yield %inserted_slice : tensor<?x?xf32>
        }
        scf.yield %22 : tensor<?x?xf32>
      }
      flow.dispatch.tensor.store %20, %10, offsets = [%arg0, %arg1], sizes = [%13, %16], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
    }
  }
  return
}
// CHECK-LABEL: func.func @pad_for_fusion
// CHECK:         %[[PAD0:.+]] = tensor.pad
// CHECK:         %[[FILL:.+]] = linalg.fill {{.+}} outs(%[[PAD0]] : tensor<8x32xf32>
// CHECK:         %[[PAD1:.+]] = tensor.pad
// CHECK:         %[[PAD2:.+]] = tensor.pad
// CHECK:         %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:      ins(%[[PAD1]], %[[PAD2]] : tensor<8x?xf32>, tensor<?x32xf32>
// CHECK-SAME:      outs(%[[FILL]]
// CHECK:         %{{.+}} = linalg.generic
// CHECK-SAME:      outs(%[[MATMUL]]
