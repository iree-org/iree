// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | \
// RUN: FileCheck %s

func.func @sort1D(%1: tensor<4xi32>) -> tensor<4xi32> {
  %2 = iree_linalg_ext.sort dimension(0) outs(%1 : tensor<4xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %3 = arith.cmpi slt, %arg0, %arg1 : i32
    iree_linalg_ext.yield %3 : i1
  } -> tensor<4xi32>
  return %2 : tensor<4xi32>
}


//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [1, 1, 1] subgroup_size = 32>
//       CHECK: func.func @sort1D(
//       CHECK:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.sort {
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [0], workgroup = [0]}>

// -----

func.func @sort2D_static_shape(%1: tensor<2000x30000xi32>) -> tensor<2000x30000xi32> {
  %2 = iree_linalg_ext.sort dimension(1) outs(%1 : tensor<2000x30000xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %3 = arith.cmpi slt, %arg0, %arg1 : i32
    iree_linalg_ext.yield %3 : i1
  } -> tensor<2000x30000xi32>
  return %2 : tensor<2000x30000xi32>
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
//       CHECK: func.func @sort2D_static_shape(
//       CHECK:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.sort {
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [1, 0], workgroup = [1, 0]}>

// -----
func.func @sort3D_dynamic_shape(%4: index, %6: tensor<?x2x4xi32>) -> tensor<?x2x4xi32> {
  %7 = iree_linalg_ext.sort dimension(2) outs(%6 : tensor<?x2x4xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %8 = arith.cmpi slt, %arg0, %arg1 : i32
    iree_linalg_ext.yield %8 : i1
  } -> tensor<?x2x4xi32>
  return %7 : tensor<?x2x4xi32>
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
//       CHECK: func.func @sort3D_dynamic_shape(
//       CHECK:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.sort {
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [1, 1, 0], workgroup = [1, 2, 0]}>

// -----
func.func @sort5D_static_shape(%1: tensor<4x100x100x200x300xi32>) -> tensor<4x100x100x200x300xi32> {
  %2 = iree_linalg_ext.sort dimension(0) outs(%1 : tensor<4x100x100x200x300xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %3 = arith.cmpi sgt, %arg0, %arg1 : i32
    iree_linalg_ext.yield %3 : i1
  } -> tensor<4x100x100x200x300xi32>
  return %2 : tensor<4x100x100x200x300xi32>
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
//   CHECK-DAG: func.func @sort5D_static_shape(
//       CHECK:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.sort {
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [0, 1, 1, 1, 1], workgroup = [0, 1, 1, 1, 1]}>
