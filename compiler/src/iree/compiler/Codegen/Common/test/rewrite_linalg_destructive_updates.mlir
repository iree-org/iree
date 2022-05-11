// RUN: iree-opt --iree-codegen-rewrite-linalg-destructive-updates --split-input-file %s | FileCheck %s

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

// -----

func @argmax() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = hal.interface.constant.load[0] : i32
  %t0 = hal.interface.constant.load[1] : i32
  %1 = hal.interface.constant.load[2] : i32
  %2 = arith.index_cast %0 : i32 to index
  %t1 = arith.index_cast %t0 : i32 to index
  %3 = arith.index_cast %1 : i32 to index
  %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:?x?x?xf32>{%2, %t1, %3}
  %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:?x?xi32>{%2, %t1}
  %6 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [%2, %t1, %3], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:?x?x?xf32>{%2, %t1, %3} -> tensor<?x?x?xf32>
  %7 = linalg.init_tensor [%2, %t1] : tensor<?x?xi32>
  %8 = linalg.init_tensor [%2, %t1] : tensor<?x?xf32>
  %t2 = tensor.dim %6, %c0 : tensor<?x?x?xf32>
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %t3 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
  %t4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
  %t5:2 = scf.for %t6 = %t3 to %t2 step %t4 iter_args(%t7 = %8, %t8 = %7) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
    %9 = tensor.dim %6, %c1 : tensor<?x?x?xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %10 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
    %11 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
    %12:2 = scf.for %arg0 = %10 to %9 step %11 iter_args(%arg1 = %t7, %arg2 = %t8) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
      %t9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%t6)[%t2]
      %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg0)[%9]
      %14 = tensor.dim %6, %c2 : tensor<?x?x?xf32>
      %15 = tensor.extract_slice %6[%t6, %arg0, 0] [%t9, %13, %14] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
      %16 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg0)[%9]
      %17 = tensor.extract_slice %arg1[%t6, %arg0] [%t9, %16] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %18 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg0)[%9]
      %19 = tensor.extract_slice %arg2[%t6, %arg0] [%t9, %18] [1, 1] : tensor<?x?xi32> to tensor<?x?xi32>
      %20:2 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%15 : tensor<?x?x?xf32>) outs(%17, %19 : tensor<?x?xf32>, tensor<?x?xi32>) {
        ^bb0(%arg3: f32, %arg4: f32, %arg5: i32):
          %23 = linalg.index 1 : index
          %24 = arith.index_cast %23 : index to i32
          %25 = arith.cmpf olt, %arg4, %arg3 : f32
          %26 = arith.select %25, %arg4, %arg3 : f32
          %27 = arith.select %25, %arg5, %24 : i32
          linalg.yield %26, %27 : f32, i32
        } -> (tensor<?x?xf32>, tensor<?x?xi32>)
      %21 = tensor.insert_slice %20#0 into %arg1[%t6, %arg0] [%t9, %16] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      %22 = tensor.insert_slice %20#1 into %arg2[%t6, %arg0] [%t9, %18] [1, 1] : tensor<?x?xi32> into tensor<?x?xi32>
      scf.yield %21, %22 : tensor<?x?xf32>, tensor<?x?xi32>
    }
    scf.yield %12#0, %12#1 : tensor<?x?xf32>, tensor<?x?xi32>
  }
  flow.dispatch.tensor.store %t5#1, %5, offsets = [0, 0], sizes = [%2, %t1], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:?x?xi32>{%2, %t1}
  return
}
// CHECK-LABEL: func @argmax()
//       CHECK: scf.for
//   CHECK-NOT:     iter_args
//       CHECK:   scf.for
//   CHECK-NOT:     iter_args
//       CHECK:     %[[INIT1:.+]] = linalg.init_tensor
//       CHECK:     %[[INIT2:.+]] = linalg.init_tensor
//       CHECK:     %[[GENERIC:.+]]:2 = linalg.generic
//  CHECK-SAME:         outs(%[[INIT1]], %[[INIT2]] :
//       CHECK:     flow.dispatch.tensor.store %[[GENERIC]]#1
//   CHECK-NOT:     flow.dispatch.tensor.store

// -----

func @reduce() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = hal.interface.constant.load[0] : i32
  %t0 = hal.interface.constant.load[1] : i32
  %1 = hal.interface.constant.load[2] : i32
  %2 = arith.index_cast %0 : i32 to index
  %t1 = arith.index_cast %t0 : i32 to index
  %3 = arith.index_cast %1 : i32 to index
  %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:?x?x?xf32>{%2, %t1, %3}
  %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:?x?xi32>{%2, %t1}
  %o1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:?x?xf32>{%2, %t1}
  %6 = flow.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [%2, %t1, %3], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:?x?x?xf32>{%2, %t1, %3} -> tensor<?x?x?xf32>
  %7 = linalg.init_tensor [%2, %t1] : tensor<?x?xi32>
  %8 = linalg.init_tensor [%2, %t1] : tensor<?x?xf32>
  %t2 = tensor.dim %6, %c0 : tensor<?x?x?xf32>
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %t3 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
  %t4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
  %t5:2 = scf.for %t6 = %t3 to %t2 step %t4 iter_args(%t7 = %8, %t8 = %7) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
    %9 = tensor.dim %6, %c1 : tensor<?x?x?xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %10 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
    %11 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
    %12:2 = scf.for %arg0 = %10 to %9 step %11 iter_args(%arg1 = %t7, %arg2 = %t8) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
      %t9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%t6)[%t2]
      %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg0)[%9]
      %14 = tensor.dim %6, %c2 : tensor<?x?x?xf32>
      %15 = tensor.extract_slice %6[%t6, %arg0, 0] [%t9, %13, %14] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
      %16 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg0)[%9]
      %17 = tensor.extract_slice %arg1[%t6, %arg0] [%t9, %16] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %18 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg0)[%9]
      %19 = tensor.extract_slice %arg2[%t6, %arg0] [%t9, %18] [1, 1] : tensor<?x?xi32> to tensor<?x?xi32>
      %20:2 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>,
                           affine_map<(d0, d1, d2) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%15 : tensor<?x?x?xf32>) outs(%17, %19 : tensor<?x?xf32>, tensor<?x?xi32>) {
        ^bb0(%arg3: f32, %arg4: f32, %arg5: i32):
          %23 = linalg.index 1 : index
          %24 = arith.index_cast %23 : index to i32
          %25 = arith.cmpf olt, %arg4, %arg3 : f32
          %26 = arith.select %25, %arg4, %arg3 : f32
          %27 = arith.select %25, %arg5, %24 : i32
          linalg.yield %26, %27 : f32, i32
        } -> (tensor<?x?xf32>, tensor<?x?xi32>)
      %21 = tensor.insert_slice %20#0 into %arg1[%t6, %arg0] [%t9, %16] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      %22 = tensor.insert_slice %20#1 into %arg2[%t6, %arg0] [%t9, %18] [1, 1] : tensor<?x?xi32> into tensor<?x?xi32>
      scf.yield %21, %22 : tensor<?x?xf32>, tensor<?x?xi32>
    }
    scf.yield %12#0, %12#1 : tensor<?x?xf32>, tensor<?x?xi32>
  }
  flow.dispatch.tensor.store %t5#1, %5, offsets = [0, 0], sizes = [%2, %t1], strides = [1, 1] : tensor<?x?xi32> -> !flow.dispatch.tensor<writeonly:?x?xi32>{%2, %t1}
  flow.dispatch.tensor.store %t5#0, %o1, offsets = [0, 0], sizes = [%2, %t1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:?x?xf32>{%2, %t1}
  return
}
// CHECK-LABEL: func @reduce()
//   CHECK-DAG:   %[[OUT1:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[OUT2:.+]] = hal.interface.binding.subspan set(0) binding(2)
//       CHECK:   scf.for
//   CHECK-NOT:       iter_args
//       CHECK:     scf.for
//   CHECK-NOT:       iter_args
//       CHECK:       %[[INIT1:.+]] = linalg.init_tensor
//       CHECK:       %[[INIT2:.+]] = linalg.init_tensor
//       CHECK:       %[[GENERIC:.+]]:2 = linalg.generic
//  CHECK-SAME:           outs(%[[INIT1]], %[[INIT2]] :
//   CHECK-DAG:       flow.dispatch.tensor.store %[[GENERIC]]#1, %[[OUT1]]
//   CHECK-DAG:       flow.dispatch.tensor.store %[[GENERIC]]#0, %[[OUT2]]

// -----

func @scatter() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %t0 = hal.interface.constant.load[2] : i32
  %2 = arith.index_cast %0 : i32 to index
  %3 = arith.index_cast %1 : i32 to index
  %t1 = arith.index_cast %t0 : i32 to index
  %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:?x?xf32>{%2, %3}
  %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:?x1xi32>{%2}
  %t2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readwrite:?x?xf32>{%t1, %3}
  %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x?xf32>{%2, %3} -> tensor<?x?xf32>
  %7 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%2, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:?x1xi32>{%2} -> tensor<?x1xi32>
  %8 = flow.dispatch.tensor.load %t2, offsets = [0, 0], sizes = [%2, 1], strides = [1, 1] : !flow.dispatch.tensor<readwrite:?x?xf32>{%t1, %3} -> tensor<?x?xf32>
  %t3 = tensor.dim %6, %c0 : tensor<?x?xf32>
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_count_y = hal.interface.workgroup.count[1] : index
  %t4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
  %t5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
  %t8 = scf.for %t6 = %t4 to %t3 step %t5 iter_args(%t7 = %8) -> tensor<?x?xf32> {
    %9 = tensor.dim %6, %c1 : tensor<?x?xf32>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %10 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]
    %11 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_x]
    %12 = scf.for %arg0 = %10 to %9 step %11 iter_args(%arg1 = %t7) -> tensor<?x?xf32> {
      %t9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%t6)[%t3]
      %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 64)>(%arg0)[%9]
      %15 = tensor.extract_slice %6[%t6, %arg0] [%t9, %13] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %17 = tensor.extract_slice %7[%t6, %arg0] [%t9, 1] [1, 1] : tensor<?x1xi32> to tensor<?x1xi32>
      %20 = iree_linalg_ext.scatter unique_indices(true)
          ins(%15, %17 : tensor<?x?xf32>, tensor<?x1xi32>) outs(%arg1 : tensor<?x?xf32>) {
        ^bb0(%arg3: f32, %arg4: f32):
          iree_linalg_ext.yield %arg3 : f32
        } -> tensor<?x?xf32>
      scf.yield %20 : tensor<?x?xf32>
    }
    scf.yield %12 : tensor<?x?xf32>
  }
  flow.dispatch.tensor.store %t8, %t2, offsets = [0, 0], sizes = [%t1, %3], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:?x?xf32>{%t1, %3}
  return
}
// CHECK-LABEL: func @scatter()
//       CHECK:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(2)
//       CHECK:   %[[OUT_TENSOR:.+]] = flow.dispatch.tensor.load %[[OUT]]
//       CHECK:   scf.for
//   CHECK-NOT:       iter_args
//       CHECK:     scf.for
//   CHECK-NOT:       iter_args
//       CHECK:       %[[SCATTER:.+]] = iree_linalg_ext.scatter
//  CHECK-SAME:           outs(%[[OUT_TENSOR]] :
//   CHECK-DAG:       flow.dispatch.tensor.store %[[SCATTER]], %[[OUT]]
