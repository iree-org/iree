// RUN: iree-opt --iree-llvmcpu-peel -split-input-file %s | FileCheck %s

func.func @peel_static_matmul() {
  %c16 = arith.constant 16 : index
  %c49 = arith.constant 49 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<128x49xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<49x512xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 65)>()[%workgroup_id_y]
  %4 = affine.apply affine_map<()[s0] -> (s0 * 65)>()[%workgroup_count_y]
  %5 = affine.apply affine_map<()[s0] -> (s0 * 65)>()[%workgroup_id_x]
  %6 = affine.apply affine_map<()[s0] -> (s0 * 65)>()[%workgroup_count_x]
  scf.for %arg0 = %3 to %c128 step %4 {
    %7 = affine.min affine_map<(d0) -> (-d0 + 128, 65)>(%arg0)
    %8 = flow.dispatch.tensor.load %0, offsets = [%arg0, 0], sizes = [%7, 49], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x49xf32>> -> tensor<?x49xf32>
    scf.for %arg1 = %5 to %c512 step %6 {
      %9 = affine.min affine_map<(d0) -> (-d0 + 512, 65)>(%arg1)
      %10 = flow.dispatch.tensor.load %2, offsets = [%arg0, %arg1], sizes = [%7, %9], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>> -> tensor<?x?xf32>
      %11 = flow.dispatch.tensor.load %1, offsets = [0, %arg1], sizes = [49, %9], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<49x512xf32>> -> tensor<49x?xf32>
      %12 = scf.for %arg2 = %c0 to %7 step %c8 iter_args(%arg3 = %10) -> (tensor<?x?xf32>) {
        %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 8)>(%arg2)[%7]
        %extracted_slice = tensor.extract_slice %8[%arg2, 0] [%13, 49] [1, 1] : tensor<?x49xf32> to tensor<?x49xf32>
        %14 = scf.for %arg4 = %c0 to %9 step %c32 iter_args(%arg5 = %arg3) -> (tensor<?x?xf32>) {
          %15 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 32)>(%arg4)[%9]
          %extracted_slice_0 = tensor.extract_slice %11[0, %arg4] [49, %15] [1, 1] : tensor<49x?xf32> to tensor<49x?xf32>
          %extracted_slice_1 = tensor.extract_slice %arg5[%arg2, %arg4] [%13, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %extracted_slice_2 = tensor.extract_slice %extracted_slice_1[0, 0] [%13, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %16 = linalg.fill ins(%cst : f32) outs(%extracted_slice_2 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %extracted_slice_3 = tensor.extract_slice %16[0, 0] [%13, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %17 = scf.for %arg6 = %c0 to %c49 step %c16 iter_args(%arg7 = %extracted_slice_3) -> (tensor<?x?xf32>) {
            %18 = affine.min affine_map<(d0) -> (-d0 + 49, 16)>(%arg6)
            %extracted_slice_5 = tensor.extract_slice %extracted_slice[0, %arg6] [%13, %18] [1, 1] : tensor<?x49xf32> to tensor<?x?xf32>
            %extracted_slice_6 = tensor.extract_slice %extracted_slice_0[%arg6, 0] [%18, %15] [1, 1] : tensor<49x?xf32> to tensor<?x?xf32>
            %19 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[65, 65, 0], [8, 32, 0], [0, 0, 16]]>} ins(%extracted_slice_5, %extracted_slice_6 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg7 : tensor<?x?xf32>) -> tensor<?x?xf32>
            scf.yield %19 : tensor<?x?xf32>
          }
          %inserted_slice = tensor.insert_slice %17 into %16[0, 0] [%13, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
          %inserted_slice_4 = tensor.insert_slice %inserted_slice into %arg5[%arg2, %arg4] [%13, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
          scf.yield %inserted_slice_4 : tensor<?x?xf32>
        }
        scf.yield %14 : tensor<?x?xf32>
      }
      flow.dispatch.tensor.store %12, %2, offsets = [%arg0, %arg1], sizes = [%7, %9], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
    }
  }
  return
}
// CHECK-LABEL: func.func @peel_static_matmul
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               scf.for
// CHECK:                 linalg.fill {{.*}} -> tensor<8x32xf32>
// CHECK:                 %[[T0:.+]] = scf.for
// CHECK:                   linalg.matmul {{.*}} tensor<8x32xf32>
// CHECK:                 linalg.matmul {{.*}} outs(%[[T0]] : tensor<8x32xf32>) -> tensor<8x32xf32>
// CHECK:               scf.for
// CHECK:                 linalg.fill {{.*}} -> tensor<8x?xf32>
// CHECK:                 %[[T1:.+]] = scf.for
// CHECK:                   linalg.matmul {{.*}} tensor<8x?xf32>
// CHECK:               scf.for
// CHECK:                 linalg.fill {{.*}} -> tensor<?x?xf32>
// CHECK:                 %[[T2:.+]] = scf.for
// CHECK:                   linalg.matmul {{.*}} tensor<?x?xf32>

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[4, 64], [1, 16]]>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
#map1 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @peel_pack(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x16x1xf32>) -> tensor<?x?x16x1xf32> {
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?x?x16x1xf32>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?x16x1xf32>
    %0 = scf.for %arg2 = %c0 to %dim step %c1 iter_args(%arg3 = %arg1) -> (tensor<?x?x16x1xf32>) {
      %1 = scf.for %arg4 = %c0 to %dim_0 step %c16 iter_args(%arg5 = %arg3) -> (tensor<?x?x16x1xf32>) {
        %2 = affine.min #map(%arg4)[%dim_0]
        %3 = affine.apply #map1(%arg2)
        %extracted_slice = tensor.extract_slice %arg0[%3, %arg4] [16, %2] [1, 1] : tensor<?x?xf32> to tensor<16x?xf32>
        %extracted_slice_1 = tensor.extract_slice %arg5[%arg2, %arg4, 0, 0] [1, %2, 16, 1] [1, 1, 1, 1] : tensor<?x?x16x1xf32> to tensor<1x?x16x1xf32>
        %pack = tensor.pack %extracted_slice inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %extracted_slice_1 {lowering_config = #config} : tensor<16x?xf32> -> tensor<1x?x16x1xf32>
        %inserted_slice = tensor.insert_slice %pack into %arg5[%arg2, %arg4, 0, 0] [1, %2, 16, 1] [1, 1, 1, 1] : tensor<1x?x16x1xf32> into tensor<?x?x16x1xf32>
        scf.yield %inserted_slice : tensor<?x?x16x1xf32>
      }
      scf.yield %1 : tensor<?x?x16x1xf32>
    }
    return %0 : tensor<?x?x16x1xf32>
  }
}
// CHECK-LABEL: func.func @peel_pack
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             tensor.pack {{.*}} : tensor<16x16xf32> -> tensor<1x16x16x1xf32>
// CHECK:           scf.for
// CHECK:             tensor.pack {{.*}} : tensor<16x?xf32> -> tensor<1x?x16x1xf32>
