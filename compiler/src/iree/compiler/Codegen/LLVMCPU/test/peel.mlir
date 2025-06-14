// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-peel))" -split-input-file %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[65, 65, 0], [8, 32, 0], [0, 0, 16]]>
func.func @peel_static_matmul(%arg0: tensor<128x49xf32>, %arg1: tensor<49x512xf32>, %arg2: tensor<128x512xf32>) -> tensor<128x512xf32> {
  %c16 = arith.constant 16 : index
  %c49 = arith.constant 49 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %0 = affine.apply affine_map<()[s0] -> (s0 * 65)>()[%workgroup_id_y]
  %1 = affine.apply affine_map<()[s0] -> (s0 * 65)>()[%workgroup_count_y]
  %2 = affine.apply affine_map<()[s0] -> (s0 * 65)>()[%workgroup_id_x]
  %3 = affine.apply affine_map<()[s0] -> (s0 * 65)>()[%workgroup_count_x]
  %4 = scf.for %arg3 = %0 to %c128 step %1 iter_args(%arg4 = %arg2) -> (tensor<128x512xf32>) {
    %5 = affine.min affine_map<(d0) -> (-d0 + 128, 65)>(%arg3)
    %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [%5, 49] [1, 1] : tensor<128x49xf32> to tensor<?x49xf32>
    %6 = scf.for %arg5 = %2 to %c512 step %3 iter_args(%arg6 = %arg4) -> (tensor<128x512xf32>) {
      %7 = affine.min affine_map<(d0) -> (-d0 + 512, 65)>(%arg5)
      %extracted_slice_0 = tensor.extract_slice %arg6[%arg3, %arg5] [%5, %7] [1, 1] : tensor<128x512xf32> to tensor<?x?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg5] [49, %7] [1, 1] : tensor<49x512xf32> to tensor<49x?xf32>
      %8 = scf.for %arg7 = %c0 to %5 step %c8 iter_args(%arg8 = %extracted_slice_0) -> (tensor<?x?xf32>) {
        %9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 8)>(%arg7)[%5]
        %extracted_slice_2 = tensor.extract_slice %extracted_slice[%arg7, 0] [%9, 49] [1, 1] : tensor<?x49xf32> to tensor<?x49xf32>
        %10 = scf.for %arg9 = %c0 to %7 step %c32 iter_args(%arg10 = %arg8) -> (tensor<?x?xf32>) {
          %11 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 32)>(%arg9)[%7]
          %extracted_slice_3 = tensor.extract_slice %extracted_slice_1[0, %arg9] [49, %11] [1, 1] : tensor<49x?xf32> to tensor<49x?xf32>
          %extracted_slice_4 = tensor.extract_slice %arg10[%arg7, %arg9] [%9, %11] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %extracted_slice_5 = tensor.extract_slice %extracted_slice_4[0, 0] [%9, %11] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %12 = linalg.fill ins(%cst : f32) outs(%extracted_slice_5 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %extracted_slice_6 = tensor.extract_slice %12[0, 0] [%9, %11] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %13 = scf.for %arg11 = %c0 to %c49 step %c16 iter_args(%arg12 = %extracted_slice_6) -> (tensor<?x?xf32>) {
            %14 = affine.min affine_map<(d0) -> (-d0 + 49, 16)>(%arg11)
            %extracted_slice_9 = tensor.extract_slice %extracted_slice_2[0, %arg11] [%9, %14] [1, 1] : tensor<?x49xf32> to tensor<?x?xf32>
            %extracted_slice_10 = tensor.extract_slice %extracted_slice_3[%arg11, 0] [%14, %11] [1, 1] : tensor<49x?xf32> to tensor<?x?xf32>
            %15 = linalg.matmul {lowering_config = #config} ins(%extracted_slice_9, %extracted_slice_10 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg12 : tensor<?x?xf32>) -> tensor<?x?xf32>
            scf.yield %15 : tensor<?x?xf32>
          }
          %inserted_slice_7 = tensor.insert_slice %13 into %12[0, 0] [%9, %11] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
          %inserted_slice_8 = tensor.insert_slice %inserted_slice_7 into %arg10[%arg7, %arg9] [%9, %11] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
          scf.yield %inserted_slice_8 : tensor<?x?xf32>
        }
        scf.yield %10 : tensor<?x?xf32>
      }
      %inserted_slice = tensor.insert_slice %8 into %arg6[%arg3, %arg5] [%5, %7] [1, 1] : tensor<?x?xf32> into tensor<128x512xf32>
      scf.yield %inserted_slice : tensor<128x512xf32>
    }
    scf.yield %6 : tensor<128x512xf32>
  }
  return %4 : tensor<128x512xf32>
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
        %pack = linalg.pack %extracted_slice inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %extracted_slice_1 {lowering_config = #config} : tensor<16x?xf32> -> tensor<1x?x16x1xf32>
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
// CHECK:             linalg.pack {{.*}} : tensor<16x16xf32> -> tensor<1x16x16x1xf32>
// CHECK:           scf.for
// CHECK:             linalg.pack {{.*}} : tensor<16x?xf32> -> tensor<1x?x16x1xf32>
