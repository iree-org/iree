// RUN: iree-opt -iree-codegen-rewrite-linalg-destructive-updates -split-input-file %s | FileCheck %s

func.func @matmul() {
  %cst = arith.constant 0.000000e+00 : f32
  %c786432 = arith.constant 786432 : index
  %c0 = arith.constant 0 : index
  %c384 = arith.constant 384 : index
  %c128 = arith.constant 128 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:128x1536xf32>
  %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c786432) alignment(64) : !flow.dispatch.tensor<readonly:1536x384xf32>
  %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:128x384xf32>
  %3 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : !flow.dispatch.tensor<writeonly:128x384xf32> -> tensor<128x384xf32>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 1536], strides = [1, 1] : !flow.dispatch.tensor<readonly:128x1536xf32> -> tensor<128x1536xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1536, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:1536x384xf32> -> tensor<1536x384xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %6 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
  %7 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
  %8 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
  %9 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
  %10 = scf.for %arg0 = %6 to %c128 step %7 iter_args(%arg1 = %3) -> (tensor<128x384xf32>) {
    %11 = tensor.extract_slice %4[%arg0, 0] [64, 1536] [1, 1] : tensor<128x1536xf32> to tensor<64x1536xf32>
    %12 = scf.for %arg2 = %8 to %c384 step %9 iter_args(%arg3 = %arg1) -> (tensor<128x384xf32>) {
      %13 = tensor.extract_slice %5[0, %arg2] [1536, 64] [1, 1] : tensor<1536x384xf32> to tensor<1536x64xf32>
      %14 = tensor.extract_slice %arg3[%arg0, %arg2] [64, 64] [1, 1] : tensor<128x384xf32> to tensor<64x64xf32>
      %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %16 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 0], [8, 32, 0], [0, 0, 16]]>} ins(%11, %13 : tensor<64x1536xf32>, tensor<1536x64xf32>) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %17 = tensor.insert_slice %16 into %arg3[%arg0, %arg2] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x384xf32>
      scf.yield %17 : tensor<128x384xf32>
    }
    scf.yield %12 : tensor<128x384xf32>
  }
  flow.dispatch.tensor.store %10, %2, offsets = [0, 0], sizes = [128, 384], strides = [1, 1] : tensor<128x384xf32> -> !flow.dispatch.tensor<writeonly:128x384xf32>
  return
}
// CHECK-LABEL: func @matmul
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             %[[MATMUL:.+]] = linalg.matmul
// CHECK:             flow.dispatch.tensor.store %[[MATMUL]]

// -----

func.func @check_offset_strides() {
  %lhs_offset_y = hal.interface.constant.load[0] : index
  %lhs_offset_x = hal.interface.constant.load[1] : index
  %lhs_stride_y = hal.interface.constant.load[2] : index
  %lhs_stride_x = hal.interface.constant.load[3] : index
  %rhs_offset_y = hal.interface.constant.load[4] : index
  %rhs_offset_x = hal.interface.constant.load[5] : index
  %rhs_stride_y = hal.interface.constant.load[6] : index
  %rhs_stride_x = hal.interface.constant.load[7] : index
  %lhs_binding_size_y = hal.interface.constant.load[8] : index
  %lhs_binding_size_x = hal.interface.constant.load[9] : index
  %rhs_binding_size_y = hal.interface.constant.load[10] : index
  %rhs_binding_size_x = hal.interface.constant.load[11] : index
  %lhs_size_y = hal.interface.constant.load[12] : index
  %lhs_size_x = hal.interface.constant.load[13] : index
  %rhs_size_y = hal.interface.constant.load[14] : index
  %rhs_size_x = hal.interface.constant.load[15] : index
  %output_offset_y = hal.interface.constant.load[16] : index
  %output_offset_x = hal.interface.constant.load[17] : index
  %output_stride_y = hal.interface.constant.load[18] : index
  %output_stride_x = hal.interface.constant.load[19] : index
  %output_binding_size_y = hal.interface.constant.load[20] : index
  %output_binding_size_x = hal.interface.constant.load[21] : index
  %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
      : !flow.dispatch.tensor<readonly:?x?xf32>{%lhs_binding_size_y, %lhs_binding_size_x}
  %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
      : !flow.dispatch.tensor<readonly:?x?xf32>{%rhs_binding_size_y, %rhs_binding_size_x}
  %output_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
      : !flow.dispatch.tensor<writeonly:?x?xf32>{%output_binding_size_y, %output_binding_size_x}
  %lhs = flow.dispatch.tensor.load %lhs_binding,
      offsets = [%lhs_offset_y, %lhs_offset_x],
      sizes = [%lhs_size_y, %lhs_size_x],
      strides = [%lhs_stride_y, %lhs_stride_x]
      : !flow.dispatch.tensor<readonly:?x?xf32>{%lhs_binding_size_y, %lhs_binding_size_x} -> tensor<?x?xf32>
  %rhs = flow.dispatch.tensor.load %rhs_binding,
      offsets = [%rhs_offset_y, %rhs_offset_x],
      sizes = [%rhs_size_y, %rhs_size_x],
      strides = [%rhs_stride_y, %rhs_stride_x]
      : !flow.dispatch.tensor<readonly:?x?xf32>{%rhs_binding_size_y, %rhs_binding_size_x} -> tensor<?x?xf32>
  %cst = arith.constant 0.0 : f32
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %lb_y = affine.apply affine_map<()[s0] -> (s0*64)>()[%workgroup_id_y]
  %step_y = affine.apply affine_map<()[s0] -> (s0*64)>()[%workgroup_count_y]
  %lb_x = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
  %step_x = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
  %init = linalg.init_tensor [%lhs_size_y, %rhs_size_x] : tensor<?x?xf32>
  %0 =  scf.for %iv0 = %lb_y to %lhs_size_y step %step_y iter_args(%arg0 = %init) -> tensor<?x?xf32> {
    %1 = scf.for %iv1 = %lb_x to %rhs_size_x step %step_x iter_args(%arg1 = %arg0) -> tensor<?x?xf32> {
      %tilesize_y = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%iv0)[%lhs_size_y]
      %tilesize_x = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%iv1)[%rhs_size_x]
      %lhs_tile = tensor.extract_slice %lhs[%iv0, 0] [%tilesize_y, %lhs_size_x] [1, 1]
          : tensor<?x?xf32> to tensor<?x?xf32>
      %rhs_tile = tensor.extract_slice %rhs[0, %iv1] [%rhs_size_y, %tilesize_x] [1, 1]
          : tensor<?x?xf32> to tensor<?x?xf32>
      %out_tile = tensor.extract_slice %arg1[%iv0, %iv1] [%tilesize_y, %tilesize_x] [1, 1]
          : tensor<?x?xf32> to tensor<?x?xf32>
      %fill = linalg.fill ins(%cst : f32) outs(%out_tile : tensor<?x?xf32>) -> tensor<?x?xf32>
      %gemm = linalg.matmul ins(%lhs_tile, %rhs_tile : tensor<?x?xf32>, tensor<?x?xf32>)
          outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
      %yield = tensor.insert_slice %gemm into %arg1[%iv0, %iv1] [%tilesize_y, %tilesize_x] [1, 1]
          : tensor<?x?xf32> into tensor<?x?xf32>
      scf.yield %yield : tensor<?x?xf32>
    }
    scf.yield %1 : tensor<?x?xf32>
  }
  flow.dispatch.tensor.store %0, %output_binding,
      offsets = [%output_offset_y, %output_offset_x],
      sizes = [%lhs_size_y, %rhs_size_x],
      strides = [%output_stride_y, %output_stride_x]
      : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%output_binding_size_y, %output_binding_size_x}
  return
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 64)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>
//      CHECK: func @check_offset_strides()
//  CHECK-DAG:   %[[LHS_OFFSET_Y:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[LHS_OFFSET_X:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[LHS_STRIDE_Y:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[LHS_STRIDE_X:.+]] = hal.interface.constant.load[3]
//  CHECK-DAG:   %[[RHS_OFFSET_Y:.+]] = hal.interface.constant.load[4]
//  CHECK-DAG:   %[[RHS_OFFSET_X:.+]] = hal.interface.constant.load[5]
//  CHECK-DAG:   %[[RHS_STRIDE_Y:.+]] = hal.interface.constant.load[6]
//  CHECK-DAG:   %[[RHS_STRIDE_X:.+]] = hal.interface.constant.load[7]
//  CHECK-DAG:   %[[LHS_SIZE_Y:.+]] = hal.interface.constant.load[12]
//  CHECK-DAG:   %[[LHS_SIZE_X:.+]] = hal.interface.constant.load[13]
//  CHECK-DAG:   %[[RHS_SIZE_Y:.+]] = hal.interface.constant.load[14]
//  CHECK-DAG:   %[[RHS_SIZE_X:.+]] = hal.interface.constant.load[15]
//  CHECK-DAG:   %[[OUTPUT_OFFSET_Y:.+]] = hal.interface.constant.load[16]
//  CHECK-DAG:   %[[OUTPUT_OFFSET_X:.+]] = hal.interface.constant.load[17]
//  CHECK-DAG:   %[[OUTPUT_STRIDE_Y:.+]] = hal.interface.constant.load[18]
//  CHECK-DAG:   %[[OUTPUT_STRIDE_X:.+]] = hal.interface.constant.load[19]
//  CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//  CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
//  CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
//      CHECK:   scf.for %[[IV0:.+]] =
//      CHECK:     scf.for %[[IV1:.+]] =
//  CHECK-DAG:       %[[TILESIZE_Y:.+]] = affine.min #[[MAP1]](%[[IV0]])[%[[LHS_SIZE_Y]]]
//  CHECK-DAG:       %[[TILESIZE_X:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[RHS_SIZE_X]]]
//  CHECK-DAG:       %[[OFFSET_Y:.+]] = affine.apply #[[MAP2]]()[%[[IV0]], %[[LHS_STRIDE_Y]], %[[LHS_OFFSET_Y]]]
//      CHECK:       %[[LHS_TILE:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:           offsets = [%[[OFFSET_Y]], %[[LHS_OFFSET_X]]]
// CHECK-SAME:           sizes = [%[[TILESIZE_Y]], %[[LHS_SIZE_X]]]
// CHECK-SAME:           strides = [%[[LHS_STRIDE_Y]], %[[LHS_STRIDE_X]]]
//      CHECK:       %[[OFFSET_X:.+]] = affine.apply #[[MAP2]]()[%[[IV1]], %[[RHS_STRIDE_X]], %[[RHS_OFFSET_X]]]
//      CHECK:       %[[RHS_TILE:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:           offsets = [%[[RHS_OFFSET_Y]], %[[OFFSET_X]]]
// CHECK-SAME:           sizes = [%[[RHS_SIZE_Y]], %[[TILESIZE_X]]]
// CHECK-SAME:           strides = [%[[RHS_STRIDE_Y]], %[[RHS_STRIDE_X]]]
//      CHECK:       linalg.fill
//      CHECK:       %[[GEMM:.+]] = linalg.matmul
//  CHECK-DAG:       %[[OUT_OFFSET_Y:.+]] = affine.apply #[[MAP2]]()[%[[IV0]], %[[OUTPUT_STRIDE_Y]], %[[OUTPUT_OFFSET_Y]]]
//  CHECK-DAG:       %[[OUT_OFFSET_X:.+]] = affine.apply #[[MAP2]]()[%[[IV1]], %[[OUTPUT_STRIDE_X]], %[[OUTPUT_OFFSET_X]]]
//      CHECK:       flow.dispatch.tensor.store %[[GEMM]], %[[RESULT_BINDING]]
// CHECK-SAME:           offsets = [%[[OUT_OFFSET_Y]], %[[OUT_OFFSET_X]]]
// CHECK-SAME:           sizes = [%[[TILESIZE_Y]], %[[TILESIZE_X]]]
// CHECK-SAME:           strides = [%[[OUTPUT_STRIDE_Y]], %[[OUTPUT_STRIDE_X]]]
