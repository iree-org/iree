// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-codegen-gpu-fuse-and-hoist-parallel-loops))' --split-input-file | FileCheck %s

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 4)>
#map2 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map3 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
#map4 = affine_map<(d0) -> (d0 * 16)>
func.func @forall_fuse_then_hoist(%3: tensor<128x128xf16>, %4: tensor<128x128xf16>, %5: tensor<128x128xf32>) -> tensor<128x128xf32>
    attributes {translation_info = #translation_info} {
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
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
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
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
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
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
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

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 4)>
#map2 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map3 = affine_map<(d0) -> (d0 * 16)>
func.func @forall_fuse_then_hoist_mixed_mappings(%3: tensor<128x128xf16>, %5: tensor<128x128xf32>) -> tensor<128x128xf32>
    attributes {translation_info = #translation_info} {
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
    } {mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
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
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
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

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 4)>
#map2 = affine_map<(d0)[s0] -> (d0 * 4 + s0)>
#map3 = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
#map4 = affine_map<(d0) -> (d0 * 16)>
func.func @forall_fuse_then_hoist_with_fill(%3: tensor<128x128xf16>, %4: tensor<128x128xf16>) -> tensor<128x128xf32>
    attributes {translation_info = #translation_info} {
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
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
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
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
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
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
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
      } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10 into %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<64x64xf16> into tensor<128x128xf16>
      }
    } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
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
      } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10 into %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<64x64xf16> into tensor<128x128xf16>
      }
    } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
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

// -----

func.func @multi_hoist_with_other_ops_in_loop(%2: tensor<128x128xf16>, %3: tensor<128x128xf16>) -> tensor<128x128xf16> {
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<128x128xf16>
  %8 = scf.for %arg0 = %c0 to %c128 step %c4 iter_args(%arg1 = %empty) -> (tensor<128x128xf16>) {
    %abs = math.absi %arg0 : index
    %id = arith.ori %abs, %arg0 : index
    %9 = scf.forall (%arg2, %arg3) in (2, 2) shared_outs(%arg4 = %arg1) -> (tensor<128x128xf16>) {
      %extracted_slice = tensor.extract_slice %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<128x128xf16> to tensor<64x64xf16>
      %10 = scf.forall (%arg5, %arg6) in (32, 16) shared_outs(%arg7 = %extracted_slice) -> (tensor<64x64xf16>) {
        %add = arith.addi %arg5, %id : index
        %extracted_slice_1 = tensor.extract_slice %2[%add, %arg6] [2, 4] [1, 1] : tensor<128x128xf16> to tensor<2x4xf16>
        %extracted_slice_2 = tensor.extract_slice %arg7[%arg5, %arg6] [2, 4] [1, 1] : tensor<64x64xf16> to tensor<2x4xf16>
        %16 = linalg.copy ins(%extracted_slice_1 : tensor<2x4xf16>) outs(%extracted_slice_2 : tensor<2x4xf16>) -> tensor<2x4xf16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %16 into %arg7[%arg5, %arg6] [2, 4] [1, 1] : tensor<2x4xf16> into tensor<64x64xf16>
        }
      } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10 into %arg4[%arg2, %arg3] [64, 64] [1, 1] : tensor<64x64xf16> into tensor<128x128xf16>
      }
    } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
    scf.yield %9 : tensor<128x128xf16>
  }
  return %8 : tensor<128x128xf16>
}

// CHECK-LABEL: func @multi_hoist_with_other_ops_in_loop
//  CHECK-SAME:   %[[I0:[A-Za-z0-9]+]]: tensor<128x128xf16>
//  CHECK-SAME:   %[[I1:[A-Za-z0-9]+]]: tensor<128x128xf16>
//       CHECK:   scf.forall
//       CHECK:     scf.forall
//       CHECK:       %[[LOOP:.+]] = scf.for {{.*}} -> (tensor<2x4xf16>)
//       CHECK:         math.absi
//       CHECK:         arith.ori
//       CHECK:         arith.addi
//       CHECK:         linalg.copy
//       CHECK:     scf.forall.in_parallel
//       CHECK:   scf.forall.in_parallel
//       CHECK:   return

// -----

func.func @hoist_with_single_trip_loops(%2: tensor<128x128xf16>, %3: tensor<128x128xf16>) -> tensor<128x128xf16> {
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<128x128xf16>
  %8 = scf.for %arg0 = %c0 to %c128 step %c4 iter_args(%arg1 = %empty) -> (tensor<128x128xf16>) {
    %9 = scf.forall (%arg2, %arg3) in (1, 1) shared_outs(%arg4 = %arg1) -> (tensor<128x128xf16>) {
      %extracted_slice = tensor.extract_slice %arg4[%arg2, %arg3] [128, 128] [1, 1] : tensor<128x128xf16> to tensor<128x128xf16>
      %10 = scf.forall (%arg5, %arg6) in (1, 1) shared_outs(%arg7 = %extracted_slice) -> (tensor<128x128xf16>) {
        %16 = linalg.copy ins(%arg7 : tensor<128x128xf16>) outs(%2 : tensor<128x128xf16>) -> tensor<128x128xf16>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %16 into %arg7[%arg5, %arg6] [128, 128] [1, 1] : tensor<128x128xf16> into tensor<128x128xf16>
        }
      } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10 into %arg4[%arg2, %arg3] [128, 128] [1, 1] : tensor<128x128xf16> into tensor<128x128xf16>
      }
    } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
    scf.yield %9 : tensor<128x128xf16>
  }
  return %8 : tensor<128x128xf16>
}

// CHECK-LABEL: func @hoist_with_single_trip_loops
//  CHECK-SAME:   %[[I0:[A-Za-z0-9]+]]: tensor<128x128xf16>
//  CHECK-SAME:   %[[I1:[A-Za-z0-9]+]]: tensor<128x128xf16>
//       CHECK:   scf.forall
//       CHECK:     scf.forall
//       CHECK:       %[[LOOP:.+]] = scf.for {{.*}} -> (tensor<128x128xf16>)
//       CHECK:         linalg.copy
//       CHECK:     scf.forall.in_parallel
//       CHECK:   scf.forall.in_parallel
//       CHECK:   return

// -----

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 16)>
func.func @no_fuse_forall_without_workgroup_size(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = tensor.empty() : tensor<128x128xf32>
  %2 = scf.forall (%arg5, %arg6) in (64, 1) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
    %4 = affine.apply #map(%arg5)
    %extracted_slice = tensor.extract_slice %arg0[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
    %extracted_slice_0 = tensor.extract_slice %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
    %5 = linalg.copy ins(%extracted_slice : tensor<2x128xf32>) outs(%extracted_slice_0 : tensor<2x128xf32>) -> tensor<2x128xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<2x128xf32> into tensor<128x128xf32>
    }
  } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  %3 = scf.forall (%arg5, %arg6) in (8, 8) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
    %6 = affine.apply #map1(%arg5)
    %7 = affine.apply #map1(%arg6)
    %extracted_slice_0 = tensor.extract_slice %2[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
    %extracted_slice_1 = tensor.extract_slice %arg7[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
    %8 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %arg7[%6, %7] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
    }
  } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  return %3 : tensor<128x128xf32>
}

//   CHECK-LABEL: func @no_fuse_forall_without_workgroup_size
// CHECK-COUNT-2:   scf.forall {{.*}} -> (tensor<128x128xf32>)

// -----

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64>
#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 16)>
func.func @no_fuse_forall_workgroup_size_mismatch(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32>
  attributes {translation_info = #translation_info} {
  %0 = tensor.empty() : tensor<128x128xf32>
  %2 = scf.forall (%arg5, %arg6) in (128, 1) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
    %4 = affine.apply #map(%arg5)
    %extracted_slice = tensor.extract_slice %arg0[%4, %arg6] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
    %extracted_slice_0 = tensor.extract_slice %arg7[%4, %arg6] [1, 128] [1, 1] : tensor<128x128xf32> to tensor<1x128xf32>
    %5 = linalg.copy ins(%extracted_slice : tensor<1x128xf32>) outs(%extracted_slice_0 : tensor<1x128xf32>) -> tensor<1x128xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg7[%4, %arg6] [1, 128] [1, 1] : tensor<1x128xf32> into tensor<128x128xf32>
    }
  } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  // We have 128 threads but only use 64 here, so loops cannot be fused.
  %3 = scf.forall (%arg5, %arg6) in (8, 8) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
    %6 = affine.apply #map1(%arg5)
    %7 = affine.apply #map1(%arg6)
    %extracted_slice_0 = tensor.extract_slice %2[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
    %extracted_slice_1 = tensor.extract_slice %arg7[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
    %8 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %arg7[%6, %7] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
    }
  } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  return %3 : tensor<128x128xf32>
}

//   CHECK-LABEL: func @no_fuse_forall_workgroup_size_mismatch
// CHECK-COUNT-2:   scf.forall {{.*}} -> (tensor<128x128xf32>)

// -----

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>
#map1 = affine_map<(d0) -> (d0 * 16)>
func.func @fuse_direct_forall_use(%arg0: tensor<128x128xf32>, %arg1: tensor<16x16xf32>) -> tensor<128x128xf32>
  attributes {translation_info = #translation_info} {
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = tensor.empty() : tensor<16x16xf32>
  %2 = scf.forall (%arg5, %arg6) in (4, 4) shared_outs(%arg7 = %1) -> (tensor<16x16xf32>) {
    %extracted_slice = tensor.extract_slice %arg1[%arg5, %arg6] [4, 4] [1, 1] : tensor<16x16xf32> to tensor<4x4xf32>
    %extracted_slice_0 = tensor.extract_slice %arg7[%arg5, %arg6] [4, 4] [1, 1] : tensor<16x16xf32> to tensor<4x4xf32>
    %5 = linalg.copy ins(%extracted_slice : tensor<4x4xf32>) outs(%extracted_slice_0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg7[%arg5, %arg6] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<16x16xf32>
    }
  } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  %3 = scf.forall (%arg5, %arg6) in (8, 8) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
    %6 = affine.apply #map1(%arg5)
    %7 = affine.apply #map1(%arg6)
    %extracted_slice_0 = tensor.extract_slice %arg0[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
    %extracted_slice_1 = tensor.extract_slice %arg7[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
    %8 = linalg.matmul ins(%2, %extracted_slice_0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %arg7[%6, %7] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
    }
  } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  return %3 : tensor<128x128xf32>
}

//   CHECK-LABEL: func @fuse_direct_forall_use
//       CHECK:   %[[FUSED_LOOP:.+]] = scf.forall
//       CHECK:     %[[BARRIER:.+]] = iree_gpu.barrier_region
//       CHECK:     linalg.matmul ins(%[[BARRIER]]
//       CHECK:   return %[[FUSED_LOOP]]

// -----

#translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>

func.func @forall_hoist_unit_loop_with_fill(%3: tensor<1x128xf16>, %4: tensor<128x1xf16>) -> tensor<1x1xf32>
    attributes {translation_info = #translation_info} {
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<1x1xf32>
  %cst = arith.constant 0.0 : f32
  %5 = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %8 = scf.for %arg0 = %c0 to %c128 step %c4 iter_args(%arg1 = %5) -> (tensor<1x1xf32>) {
    %11 = scf.forall (%arg2, %arg3) in (1, 1) shared_outs(%arg4 = %arg1) -> (tensor<1x1xf32>) {
      %12 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg0)
      %extracted_slice = tensor.extract_slice %3[0, %12] [1, 4] [1, 1] : tensor<1x128xf16> to tensor<1x4xf16>
      %extracted_slice_0 = tensor.extract_slice %4[%12, 0] [4, 1] [1, 1] : tensor<128x1xf16> to tensor<4x1xf16>
      %14 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<1x4xf16>, tensor<4x1xf16>) outs(%arg4 : tensor<1x1xf32>) -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %14 into %arg4[%arg2, %arg3] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x1xf32>
      }
    } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
    scf.yield %11 : tensor<1x1xf32>
  }
  return %8 : tensor<1x1xf32>
}

// CHECK-LABEL: func @forall_hoist_unit_loop_with_fill
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x1xf32>
//       CHECK:   %[[OUTER_PARALLEL:.+]] = scf.forall {{.*}}shared_outs(%[[ITER:.+]] = %[[EMPTY]])
//       CHECK:     %[[FILL:.+]] = linalg.fill {{.*}} outs(%[[ITER]]
//       CHECK:     %[[LOOP:.+]] = scf.for {{.*}} iter_args(%{{.*}} = %[[FILL]])
//       CHECK:     scf.yield {{.*}} : tensor<1x1xf32>
//       CHECK:   scf.forall.in_parallel
//  CHECK-NEXT:     tensor.parallel_insert_slice %[[LOOP]] into %[[ITER]]
//       CHECK:   return %[[OUTER_PARALLEL]]

// -----

func.func @no_fuse_multi_use(%2: tensor<128x128xf16>, %3: tensor<128x128xf16>) -> tensor<128x128xf16> {
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %empty = tensor.empty() : tensor<128x128xf16>
  %10:2 = scf.forall (%arg5, %arg6) in (32, 32) shared_outs(%arg7 = %empty, %arg8 = %empty) -> (tensor<128x128xf16>, tensor<128x128xf16>) {
    %extracted_slice_1 = tensor.extract_slice %2[%arg5, %arg6] [2, 2] [1, 1] : tensor<128x128xf16> to tensor<2x2xf16>
    %extracted_slice_2 = tensor.extract_slice %arg7[%arg5, %arg6] [2, 2] [1, 1] : tensor<128x128xf16> to tensor<2x2xf16>
    %extracted_slice_3 = tensor.extract_slice %arg8[%arg6, %arg5] [2, 2] [1, 1] : tensor<128x128xf16> to tensor<2x2xf16>
    %16 = linalg.copy ins(%extracted_slice_1 : tensor<2x2xf16>) outs(%extracted_slice_2 : tensor<2x2xf16>) -> tensor<2x2xf16>
    %17 = linalg.transpose ins(%extracted_slice_1 : tensor<2x2xf16>) outs(%extracted_slice_3 : tensor<2x2xf16>) permutation = [1, 0]
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %16 into %arg7[%arg5, %arg6] [2, 2] [1, 1] : tensor<2x2xf16> into tensor<128x128xf16>
      tensor.parallel_insert_slice %17 into %arg8[%arg6, %arg5] [2, 2] [1, 1] : tensor<2x2xf16> into tensor<128x128xf16>
    }
  } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
  %add = linalg.add
    ins(%10#0, %10#1 : tensor<128x128xf16>, tensor<128x128xf16>)
    outs(%empty: tensor<128x128xf16>) -> tensor<128x128xf16>
  return %add : tensor<128x128xf16>
}

// CHECK-LABEL: func @no_fuse_multi_use
//       CHECK:   scf.forall
//       CHECK:     linalg.copy
//       CHECK:     linalg.transpose
//       CHECK:   scf.forall.in_parallel
//       CHECK:   linalg.add
//       CHECK:   return

// -----

#map = affine_map<(d0) -> (d0 * 64)>
func.func @fuse_imperfectly_aligned_unpack(%arg0: tensor<5x31xf16>, %arg1: index) -> tensor<128xf16> {
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<128xf16>
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0] inner_tiles = [31] into %0 : tensor<5x31xf16> -> tensor<128xf16>
  %1 = scf.forall (%arg2) in (2) shared_outs(%arg3 = %0) -> (tensor<128xf16>) {
    %2 = affine.apply #map(%arg2)
    %extracted_slice = tensor.extract_slice %unpack[%2] [64] [1] : tensor<128xf16> to tensor<64xf16>
    %extracted_slice_0 = tensor.extract_slice %arg3[%2] [64] [1] : tensor<128xf16> to tensor<64xf16>
    %3 = linalg.copy ins(%extracted_slice : tensor<64xf16>) outs(%extracted_slice_0 : tensor<64xf16>) -> tensor<64xf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %3 into %arg3[%2] [64] [1] : tensor<64xf16> into tensor<128xf16>
    }
  } {mapping = [#gpu.thread<linear_dim_0>]}
  return %1 : tensor<128xf16>
}

// CHECK-LABEL: func @fuse_imperfectly_aligned_unpack
//       CHECK:   scf.forall
//       CHECK:     linalg.unpack
//       CHECK:     linalg.copy
//       CHECK:   scf.forall.in_parallel
//       CHECK:   return

// -----

#map = affine_map<(d0) -> (d0 * 2)>
func.func @no_fuse_non_contiguous_collapse_shape(%arg0: tensor<8x8xf32>) -> tensor<64xf32> {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = scf.forall (%arg1) in (4) shared_outs(%arg2 = %0) -> (tensor<8x8xf32>) {
    %2 = affine.apply #map(%arg1)
    %extracted_slice = tensor.extract_slice %arg0[%2, 0] [2, 7] [1, 1] : tensor<8x8xf32> to tensor<2x7xf32>
    %extracted_slice_0 = tensor.extract_slice %arg2[%2, 0] [2, 7] [1, 1] : tensor<8x8xf32> to tensor<2x7xf32>
    %3 = linalg.copy ins(%extracted_slice : tensor<2x7xf32>) outs(%extracted_slice_0 : tensor<2x7xf32>) -> tensor<2x7xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %3 into %arg2[%2, 0] [2, 7] [1, 1] : tensor<2x7xf32> into tensor<8x8xf32>
    }
  } {mapping = [#gpu.thread<x>]}
  %collapsed = tensor.collapse_shape %1 [[0, 1]] : tensor<8x8xf32> into tensor<64xf32>
  return %collapsed : tensor<64xf32>
}

// CHECK-LABEL: func @no_fuse_non_contiguous_collapse_shape
//       CHECK:   %[[FORALL_RESULT:.+]] = scf.forall {{.*}} -> (tensor<8x8xf32>) {
//       CHECK:     scf.forall.in_parallel {
//   CHECK-DAG:       tensor.parallel_insert_slice {{.*}} : tensor<2x7xf32> into tensor<8x8xf32>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<x>]}
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[FORALL_RESULT]]
//       CHECK:   return %[[COLLAPSE]]

// -----

#map = affine_map<(d0) -> (d0 * 2)>
func.func @no_fuse_collapse_shape_rank_reduced(%arg0: tensor<8x8xf32>) -> tensor<64xf32> {
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = scf.forall (%arg1) in (8) shared_outs(%arg2 = %0) -> (tensor<8x8xf32>) {
    %2 = affine.apply #map(%arg1)
    %extracted_slice = tensor.extract_slice %arg0[%2, 0] [1, 8] [1, 1] : tensor<8x8xf32> to tensor<8xf32>
    %extracted_slice_0 = tensor.extract_slice %arg2[%2, 0] [1, 8] [1, 1] : tensor<8x8xf32> to tensor<8xf32>
    %3 = linalg.copy ins(%extracted_slice : tensor<8xf32>) outs(%extracted_slice_0 : tensor<8xf32>) -> tensor<8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %3 into %arg2[%2, 0] [1, 8] [1, 1] : tensor<8xf32> into tensor<8x8xf32>
    }
  } {mapping = [#gpu.thread<x>]}
  %collapsed = tensor.collapse_shape %1 [[0, 1]] : tensor<8x8xf32> into tensor<64xf32>
  return %collapsed : tensor<64xf32>
}

// CHECK-LABEL: func @no_fuse_collapse_shape_rank_reduced
//       CHECK:   %[[FORALL_RESULT:.+]] = scf.forall {{.*}} -> (tensor<8x8xf32>) {
//       CHECK:     scf.forall.in_parallel {
//   CHECK-DAG:       tensor.parallel_insert_slice {{.*}} : tensor<8xf32> into tensor<8x8xf32>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<x>]}
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[FORALL_RESULT]]
//       CHECK:   return %[[COLLAPSE]]

// -----

#map = affine_map<(d0) -> (d0 * 2)>
func.func @no_fuse_extract_slice_rank_reduced(%arg0: tensor<4x8xf32>, %size1: index) -> tensor<?xf32> {
  %0 = tensor.empty() : tensor<4x8xf32>
  %1 = scf.forall (%arg2) in (4) shared_outs(%arg3 = %0) -> (tensor<4x8xf32>) {
    %2 = affine.apply #map(%arg2)
    %extracted_slice_0 = tensor.extract_slice %arg0[0, %2] [1, 2] [1, 1] : tensor<4x8xf32> to tensor<2xf32>
    %extracted_slice_1 = tensor.extract_slice %arg3[0, %2] [1, 2] [1, 1] : tensor<4x8xf32> to tensor<2xf32>
    %3 = linalg.copy ins(%extracted_slice_0 : tensor<2xf32>) outs(%extracted_slice_1 : tensor<2xf32>) -> tensor<2xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %3 into %arg3[0, %2] [1, 2] [1, 1] : tensor<2xf32> into tensor<4x8xf32>
    }
  } {mapping = [#gpu.thread<x>]}
  %extracted_slice = tensor.extract_slice %1[0, 0] [1, %size1] [1, 1] : tensor<4x8xf32> to tensor<?xf32>
  return %extracted_slice : tensor<?xf32>
}

// CHECK-LABEL: func @no_fuse_extract_slice_rank_reduced
//       CHECK:   %[[FORALL_RESULT:.+]] = scf.forall {{.*}} -> (tensor<4x8xf32>) {
//       CHECK:     scf.forall.in_parallel {
//   CHECK-DAG:       tensor.parallel_insert_slice {{.*}} : tensor<2xf32> into tensor<4x8xf32>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<x>]}
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[FORALL_RESULT]]
//       CHECK:   return %[[EXTRACT]]

// -----

// Tests `collapse_shape` fusion into `scf.forall` with the `parallel_insert_slice` depending on both
// a block argument and an affine expression on the same block argument. This lead to dominance issues
// before being fixed.

#map = affine_map<(d0) -> (d0 * 2)>
func.func @fuse_collapse_shape(%arg0: tensor<?x?x8xf32>, %arg1 : index, %arg2 : index) -> tensor<?xf32> {
  %0 = tensor.empty(%arg1, %arg2) : tensor<?x?x8xf32>
  %1 = scf.forall (%arg3, %arg4) in (%arg1, %arg2) shared_outs(%arg5 = %0) -> (tensor<?x?x8xf32>) {
    %2 = affine.apply #map(%arg4)
    %extracted_slice = tensor.extract_slice %arg0[%arg3, %2, 0] [1, 1, 8] [1, 1, 1] : tensor<?x?x8xf32> to tensor<1x1x8xf32>
    %extracted_slice_0 = tensor.extract_slice %arg5[%arg3, %2, 0] [1, 1, 8] [1, 1, 1] : tensor<?x?x8xf32> to tensor<1x1x8xf32>
    %3 = linalg.copy ins(%extracted_slice : tensor<1x1x8xf32>) outs(%extracted_slice_0 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %3 into %arg5[%arg3, %2, 0] [1, 1, 8] [1, 1, 1] : tensor<1x1x8xf32> into tensor<?x?x8xf32>
    }
  } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
  %collapsed = tensor.collapse_shape %1 [[0, 1, 2]] : tensor<?x?x8xf32> into tensor<?xf32>
  return %collapsed : tensor<?xf32>
}

// CHECK-LABEL: func @fuse_collapse_shape
//       CHECK:   %[[FORALL_RESULT:.+]] = scf.forall {{.*}} -> (tensor<?xf32>) {
//       CHECK:     %[[COPY:.+]] = linalg.copy
//       CHECK:     %[[COLLAPSE:.+]] = tensor.collapse_shape %[[COPY]]
//       CHECK:     scf.forall.in_parallel {
//   CHECK-DAG:       tensor.parallel_insert_slice %[[COLLAPSE]] into {{.*}} [8] [1] : tensor<8xf32> into tensor<?xf32>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
//       CHECK:   return %[[FORALL_RESULT]]
