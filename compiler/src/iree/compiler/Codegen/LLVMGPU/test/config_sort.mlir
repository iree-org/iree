// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | \
// RUN: FileCheck %s

func.func @sort1D() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi32>>
  %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi32>> -> tensor<4xi32>
  %2 = iree_linalg_ext.sort dimension(0) outs(%1 : tensor<4xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %3 = arith.cmpi slt, %arg0, %arg1 : i32
    iree_linalg_ext.yield %3 : i1
  } -> tensor<4xi32>
  iree_tensor_ext.dispatch.tensor.store %2, %0, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi32>>
  return
}


//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [1, 1, 1] subgroup_size = 32>
//       CHECK: func.func @sort1D()
//       CHECK:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.sort {
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [0], workgroup = [0]}>

// -----

func.func @sort2D_static_shape() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2000x30000xi32>>
  %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2000, 30000], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2000x30000xi32>> -> tensor<2000x30000xi32>
  %2 = iree_linalg_ext.sort dimension(1) outs(%1 : tensor<2000x30000xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %3 = arith.cmpi slt, %arg0, %arg1 : i32
    iree_linalg_ext.yield %3 : i1
  } -> tensor<2000x30000xi32>
  iree_tensor_ext.dispatch.tensor.store %2, %0, offsets = [0, 0], sizes = [2000, 30000], strides = [1, 1] : tensor<2000x30000xi32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2000x30000xi32>>
  return
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
//       CHECK: func.func @sort2D_static_shape()
//       CHECK:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.sort {
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [1, 0], workgroup = [1, 0]}>

// -----
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @sort3D_dynamic_shape() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.index_castui %0 : i32 to index
  %3 = arith.index_castui %1 : i32 to index
  %4 = iree_tensor_ext.dispatch.workload.ordinal %3, 0 : index
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%2) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x2x4xi32>>{%4}
  %6 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0, 0], sizes = [%4, 2, 4], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x2x4xi32>>{%4} -> tensor<?x2x4xi32>
  %7 = iree_linalg_ext.sort dimension(2) outs(%6 : tensor<?x2x4xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %8 = arith.cmpi slt, %arg0, %arg1 : i32
    iree_linalg_ext.yield %8 : i1
  } -> tensor<?x2x4xi32>
  iree_tensor_ext.dispatch.tensor.store %7, %5, offsets = [0, 0, 0], sizes = [%4, 2, 4], strides = [1, 1, 1] : tensor<?x2x4xi32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x2x4xi32>>{%4}
  return
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
//       CHECK: func.func @sort3D_dynamic_shape()
//       CHECK:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.sort {
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [1, 1, 0], workgroup = [1, 2, 0]}>

// -----
func.func @sort5D_static_shape() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x100x100x200x300xi32>>
  %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0, 0], sizes = [4, 100, 100, 200, 300], strides = [1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x100x100x200x300xi32>> -> tensor<4x100x100x200x300xi32>
  %2 = iree_linalg_ext.sort dimension(0) outs(%1 : tensor<4x100x100x200x300xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %3 = arith.cmpi sgt, %arg0, %arg1 : i32
    iree_linalg_ext.yield %3 : i1
  } -> tensor<4x100x100x200x300xi32>
  iree_tensor_ext.dispatch.tensor.store %2, %0, offsets = [0, 0, 0, 0, 0], sizes = [4, 100, 100, 200, 300], strides = [1, 1, 1, 1, 1] : tensor<4x100x100x200x300xi32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x100x100x200x300xi32>>
  return
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
//   CHECK-DAG: func.func @sort5D_static_shape()
//       CHECK:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.sort {
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [0, 1, 1, 1, 1], workgroup = [0, 1, 1, 1, 1]}>
