// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-gpu-fuse-and-hoist-parallel-loops))' --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 4)>
#map2 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map3 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
#map4 = affine_map<(d0) -> (d0 * 16)>
func.func @forall_fuse_then_hoist(%3: tensor<128x128xf16>, %4: tensor<128x128xf16>, %5: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %6 = tensor.empty() : tensor<128x4xf16>
  %7 = tensor.empty() : tensor<4x128xf16>
  %8 = scf.for %arg0 = %c0 to %c128 step %c4 iter_args(%arg1 = %5) -> (tensor<128x128xf32>) {
    %9 = scf.forall (%arg2, %arg3) in (64, 1) shared_outs(%arg4 = %6) -> (tensor<128x4xf16>) {
      %12 = affine.apply #map(%arg2)
      %13 = affine.apply #map1(%arg3)
      %14 = affine.apply #map(%arg2)
      %15 = affine.apply #map2(%arg3)[%arg0]
      %extracted_slice = tensor.extract_slice %3[%14, %15] [2, 4] [1, 1] : tensor<128x128xf16> to tensor<2x4xf16>
      %extracted_slice_0 = tensor.extract_slice %arg4[%12, %13] [2, 4] [1, 1] : tensor<128x4xf16> to tensor<2x4xf16>
      %16 = linalg.copy ins(%extracted_slice : tensor<2x4xf16>) outs(%extracted_slice_0 : tensor<2x4xf16>) -> tensor<2x4xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %16 into %arg4[%12, %13] [2, 4] [1, 1] : tensor<2x4xf16> into tensor<128x4xf16>
      }
    } {mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]}
    %10 = scf.forall (%arg2, %arg3) in (2, 32) shared_outs(%arg4 = %7) -> (tensor<4x128xf16>) {
      %12 = affine.apply #map(%arg2)
      %13 = affine.apply #map1(%arg3)
      %14 = affine.apply #map3(%arg2)[%arg0]
      %15 = affine.apply #map1(%arg3)
      %extracted_slice = tensor.extract_slice %4[%14, %15] [2, 4] [1, 1] : tensor<128x128xf16> to tensor<2x4xf16>
      %extracted_slice_0 = tensor.extract_slice %arg4[%12, %13] [2, 4] [1, 1] : tensor<4x128xf16> to tensor<2x4xf16>
      %16 = linalg.copy ins(%extracted_slice : tensor<2x4xf16>) outs(%extracted_slice_0 : tensor<2x4xf16>) -> tensor<2x4xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %16 into %arg4[%12, %13] [2, 4] [1, 1] : tensor<2x4xf16> into tensor<4x128xf16>
      }
    } {mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]}
    %11 = scf.forall (%arg2, %arg3) in (8, 8) shared_outs(%arg4 = %arg1) -> (tensor<128x128xf32>) {
      %12 = affine.apply #map4(%arg2)
      %13 = affine.apply #map4(%arg3)
      %extracted_slice = tensor.extract_slice %9[%12, 0] [16, 4] [1, 1] : tensor<128x4xf16> to tensor<16x4xf16>
      %extracted_slice_0 = tensor.extract_slice %10[0, %13] [4, 16] [1, 1] : tensor<4x128xf16> to tensor<4x16xf16>
      %extracted_slice_1 = tensor.extract_slice %arg4[%12, %13] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %14 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<16x4xf16>, tensor<4x16xf16>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %14 into %arg4[%12, %13] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]}
    scf.yield %11 : tensor<128x128xf32>
  }
  return %8 : tensor<128x128xf32>
}

// CHECK-LABEL: func @forall_fuse_then_hoist
//       CHECK:   %[[OUTER_PARALLEL:.+]] = scf.forall
//       CHECK:     %[[LOOP:.+]] = scf.for
//       CHECK:     scf.yield {{.*}} : tensor<16x16xf32>
//       CHECK:   scf.forall.in_parallel
//  CHECK-NEXT:     tensor.parallel_insert_slice %[[LOOP]]
//       CHECK:   return %[[OUTER_PARALLEL]]

// -----

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 4)>
#map2 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map3 = affine_map<(d0) -> (d0 * 16)>
func.func @forall_fuse_then_hoist_mixed_mappings(%3: tensor<128x128xf16>, %5: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.0> : tensor<4x128xf16>
  %6 = tensor.empty() : tensor<128x4xf16>
  %7 = tensor.empty() : tensor<4x128xf16>
  %8 = scf.for %arg0 = %c0 to %c128 step %c4 iter_args(%arg1 = %5) -> (tensor<128x128xf32>) {
    %9 = scf.forall (%arg2, %arg3, %arg4) in (1, 64, 1) shared_outs(%arg5 = %6) -> (tensor<128x4xf16>) {
      %12 = affine.apply #map(%arg3)
      %13 = affine.apply #map1(%arg4)
      %14 = affine.apply #map(%arg3)
      %15 = affine.apply #map2(%arg4)[%arg0]
      %extracted_slice = tensor.extract_slice %3[%14, %15] [2, 4] [1, 1] : tensor<128x128xf16> to tensor<2x4xf16>
      %extracted_slice_0 = tensor.extract_slice %arg5[%12, %13] [2, 4] [1, 1] : tensor<128x4xf16> to tensor<2x4xf16>
      %16 = linalg.copy ins(%extracted_slice : tensor<2x4xf16>) outs(%extracted_slice_0 : tensor<2x4xf16>) -> tensor<2x4xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %16 into %arg5[%12, %13] [2, 4] [1, 1] : tensor<2x4xf16> into tensor<128x4xf16>
      }
    } {mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_2>]}
    %11 = scf.forall (%arg2, %arg3) in (8, 8) shared_outs(%arg4 = %arg1) -> (tensor<128x128xf32>) {
      %12 = affine.apply #map3(%arg2)
      %13 = affine.apply #map3(%arg3)
      %extracted_slice = tensor.extract_slice %9[%12, 0] [16, 4] [1, 1] : tensor<128x4xf16> to tensor<16x4xf16>
      %extracted_slice_0 = tensor.extract_slice %cst[0, %13] [4, 16] [1, 1] : tensor<4x128xf16> to tensor<4x16xf16>
      %extracted_slice_1 = tensor.extract_slice %arg4[%12, %13] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %14 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<16x4xf16>, tensor<4x16xf16>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %14 into %arg4[%12, %13] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]}
    scf.yield %11 : tensor<128x128xf32>
  }
  return %8 : tensor<128x128xf32>
}

// CHECK-LABEL: func @forall_fuse_then_hoist_mixed_mappings
//       CHECK:   %[[OUTER_PARALLEL:.+]] = scf.forall
//       CHECK:     %[[LOOP:.+]] = scf.for
//       CHECK:     scf.yield {{.*}} : tensor<16x16xf32>
//       CHECK:   scf.forall.in_parallel
//  CHECK-NEXT:     tensor.parallel_insert_slice %[[LOOP]]
//   CHECK-NOT:   scf.forall
//       CHECK:   return %[[OUTER_PARALLEL]]

// -----

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 4)>
#map2 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map3 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
#map4 = affine_map<(d0) -> (d0 * 16)>
func.func @forall_fuse_then_hoist_with_fill(%3: tensor<128x128xf16>, %4: tensor<128x128xf16>) -> tensor<128x128xf32> {
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<128x128xf32>
  %cst = arith.constant 0.0 : f32
  %5 = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %6 = tensor.empty() : tensor<128x4xf16>
  %7 = tensor.empty() : tensor<4x128xf16>
  %8 = scf.for %arg0 = %c0 to %c128 step %c4 iter_args(%arg1 = %5) -> (tensor<128x128xf32>) {
    %9 = scf.forall (%arg2, %arg3) in (64, 1) shared_outs(%arg4 = %6) -> (tensor<128x4xf16>) {
      %12 = affine.apply #map(%arg2)
      %13 = affine.apply #map1(%arg3)
      %14 = affine.apply #map(%arg2)
      %15 = affine.apply #map2(%arg3)[%arg0]
      %extracted_slice = tensor.extract_slice %3[%14, %15] [2, 4] [1, 1] : tensor<128x128xf16> to tensor<2x4xf16>
      %extracted_slice_0 = tensor.extract_slice %arg4[%12, %13] [2, 4] [1, 1] : tensor<128x4xf16> to tensor<2x4xf16>
      %16 = linalg.copy ins(%extracted_slice : tensor<2x4xf16>) outs(%extracted_slice_0 : tensor<2x4xf16>) -> tensor<2x4xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %16 into %arg4[%12, %13] [2, 4] [1, 1] : tensor<2x4xf16> into tensor<128x4xf16>
      }
    } {mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]}
    %10 = scf.forall (%arg2, %arg3) in (2, 32) shared_outs(%arg4 = %7) -> (tensor<4x128xf16>) {
      %12 = affine.apply #map(%arg2)
      %13 = affine.apply #map1(%arg3)
      %14 = affine.apply #map3(%arg2)[%arg0]
      %15 = affine.apply #map1(%arg3)
      %extracted_slice = tensor.extract_slice %4[%14, %15] [2, 4] [1, 1] : tensor<128x128xf16> to tensor<2x4xf16>
      %extracted_slice_0 = tensor.extract_slice %arg4[%12, %13] [2, 4] [1, 1] : tensor<4x128xf16> to tensor<2x4xf16>
      %16 = linalg.copy ins(%extracted_slice : tensor<2x4xf16>) outs(%extracted_slice_0 : tensor<2x4xf16>) -> tensor<2x4xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %16 into %arg4[%12, %13] [2, 4] [1, 1] : tensor<2x4xf16> into tensor<4x128xf16>
      }
    } {mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]}
    %11 = scf.forall (%arg2, %arg3) in (8, 8) shared_outs(%arg4 = %arg1) -> (tensor<128x128xf32>) {
      %12 = affine.apply #map4(%arg2)
      %13 = affine.apply #map4(%arg3)
      %extracted_slice = tensor.extract_slice %9[%12, 0] [16, 4] [1, 1] : tensor<128x4xf16> to tensor<16x4xf16>
      %extracted_slice_0 = tensor.extract_slice %10[0, %13] [4, 16] [1, 1] : tensor<4x128xf16> to tensor<4x16xf16>
      %extracted_slice_1 = tensor.extract_slice %arg4[%12, %13] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %14 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<16x4xf16>, tensor<4x16xf16>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %14 into %arg4[%12, %13] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]}
    scf.yield %11 : tensor<128x128xf32>
  }
  return %8 : tensor<128x128xf32>
}

// CHECK-LABEL: func @forall_fuse_then_hoist_with_fill
//       CHECK:   %[[OUTER_PARALLEL:.+]] = scf.forall
//       CHECK:     %[[FILL:.+]] = linalg.fill
//       CHECK:     %[[LOOP:.+]] = scf.for {{.*}} iter_args(%{{.*}} = %[[FILL]])
//       CHECK:     scf.yield {{.*}} : tensor<16x16xf32>
//       CHECK:   scf.forall.in_parallel
//  CHECK-NEXT:     tensor.parallel_insert_slice %[[LOOP]]
//       CHECK:   return %[[OUTER_PARALLEL]]

// -----

func.func @multi_hoist_and_fuse_trailing_stuff(%2: tensor<128x128xf16>) -> tensor<128x128xf16> {
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<128x128xf16>
  %8 = scf.for %arg0 = %c0 to %c128 step %c4 iter_args(%arg1 = %empty) -> (tensor<128x128xf16>) {
    %9 = scf.forall (%arg2, %arg3) in (2, 2) shared_outs(%arg4 = %arg1) -> (tensor<128x128xf16>) {
      %extracted_slice = tensor.extract_slice %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<128x128xf16> to tensor<64x64xf16>
      %10 = scf.forall (%arg5, %arg6) in (32, 16) shared_outs(%arg7 = %extracted_slice) -> (tensor<64x64xf16>) {
        %extracted_slice_1 = tensor.extract_slice %2[%arg5, %arg6] [2, 4] [1, 1] : tensor<128x128xf16> to tensor<2x4xf16>
        %extracted_slice_2 = tensor.extract_slice %arg7[%arg5, %arg6] [2, 4] [1, 1] : tensor<64x64xf16> to tensor<2x4xf16>
        %16 = linalg.copy ins(%extracted_slice_1 : tensor<2x4xf16>) outs(%extracted_slice_2 : tensor<2x4xf16>) -> tensor<2x4xf16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %16 into %arg7[%arg5, %arg6] [2, 4] [1, 1] : tensor<2x4xf16> into tensor<64x64xf16>
        }
      } {mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10 into %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<64x64xf16> into tensor<128x128xf16>
      }
    } {mapping = [#gpu.warp<linear_dim_0>, #gpu.warp<linear_dim_1>]}
    scf.yield %9 : tensor<128x128xf16>
  }
  %transpose = linalg.transpose ins(%8: tensor<128x128xf16>) outs(%empty: tensor<128x128xf16>) permutation = [1, 0]
  %ceil = linalg.ceil ins(%transpose: tensor<128x128xf16>) outs(%empty: tensor<128x128xf16>) -> tensor<128x128xf16>
  return %ceil : tensor<128x128xf16>
}

// CHECK-LABEL: func @multi_hoist_and_fuse_trailing_stuff
//       CHECK:   scf.forall
//       CHECK:     scf.forall
//       CHECK:       %[[LOOP:.+]] = scf.for {{.*}} -> (tensor<2x4xf16>)
//       CHECK:         linalg.copy
//       CHECK:       %[[T:.+]] = linalg.transpose ins(%[[LOOP]] : tensor<2x4xf16>)
//       CHECK:       linalg.ceil ins(%[[T]] : tensor<4x2xf16>) {{.*}} -> tensor<4x2xf16>
//       CHECK:     scf.forall.in_parallel
//       CHECK:   scf.forall.in_parallel
//       CHECK:   return

// -----

func.func @multi_hoist_and_fuse_trailing_with_producer_fusion(%2: tensor<128x128xf16>, %3: tensor<128x128xf16>) -> tensor<128x128xf16> {
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<128x128xf16>
  %8 = scf.for %arg0 = %c0 to %c128 step %c4 iter_args(%arg1 = %empty) -> (tensor<128x128xf16>) {
    %9 = scf.forall (%arg2, %arg3) in (2, 2) shared_outs(%arg4 = %arg1) -> (tensor<128x128xf16>) {
      %extracted_slice = tensor.extract_slice %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<128x128xf16> to tensor<64x64xf16>
      %10 = scf.forall (%arg5, %arg6) in (32, 16) shared_outs(%arg7 = %extracted_slice) -> (tensor<64x64xf16>) {
        %extracted_slice_1 = tensor.extract_slice %2[%arg5, %arg6] [2, 4] [1, 1] : tensor<128x128xf16> to tensor<2x4xf16>
        %extracted_slice_2 = tensor.extract_slice %arg7[%arg5, %arg6] [2, 4] [1, 1] : tensor<64x64xf16> to tensor<2x4xf16>
        %16 = linalg.copy ins(%extracted_slice_1 : tensor<2x4xf16>) outs(%extracted_slice_2 : tensor<2x4xf16>) -> tensor<2x4xf16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %16 into %arg7[%arg5, %arg6] [2, 4] [1, 1] : tensor<2x4xf16> into tensor<64x64xf16>
        }
      } {mapping = [#gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_1>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10 into %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<64x64xf16> into tensor<128x128xf16>
      }
    } {mapping = [#gpu.warp<linear_dim_0>, #gpu.warp<linear_dim_1>]}
    scf.yield %9 : tensor<128x128xf16>
  }
  %transpose_input = linalg.transpose ins(%3: tensor<128x128xf16>) outs(%empty: tensor<128x128xf16>) permutation = [1, 0]
  %add = linalg.add
    ins(%8, %transpose_input : tensor<128x128xf16>, tensor<128x128xf16>)
    outs(%empty: tensor<128x128xf16>) -> tensor<128x128xf16>
  return %add : tensor<128x128xf16>
}

// CHECK-LABEL: func @multi_hoist_and_fuse_trailing_with_producer_fusion
//  CHECK-SAME:   %[[I0:[A-Za-z0-9]+]]: tensor<128x128xf16>
//  CHECK-SAME:   %[[I1:[A-Za-z0-9]+]]: tensor<128x128xf16>
//       CHECK:   scf.forall
//       CHECK:     scf.forall
//       CHECK:       %[[LOOP:.+]] = scf.for {{.*}} -> (tensor<2x4xf16>)
//       CHECK:         linalg.copy
//       CHECK:       %[[T:.+]] = linalg.transpose ins(%{{.*}} : tensor<4x2xf16>)
//       CHECK:       linalg.add ins(%[[LOOP]], %[[T]] : tensor<2x4xf16>, tensor<2x4xf16>) {{.*}} -> tensor<2x4xf16>
//       CHECK:     scf.forall.in_parallel
//       CHECK:   scf.forall.in_parallel
//       CHECK:   return
