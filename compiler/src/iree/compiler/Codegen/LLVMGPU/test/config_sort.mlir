// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s

module {
  func.func @_sort1D_dispatch_0_sort_4xi32_dispatch_tensor_store() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !flow.dispatch.tensor<readwrite:tensor<4xi32>>
    %1 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<4xi32>> -> tensor<4xi32>
    %2 = iree_linalg_ext.sort dimension(0) outs(%1 : tensor<4xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %3 = arith.cmpi slt, %arg0, %arg1 : i32
      iree_linalg_ext.yield %3 : i1
    } -> tensor<4xi32>
    flow.dispatch.tensor.store %2, %0, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !flow.dispatch.tensor<readwrite:tensor<4xi32>>
    return
  }
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [1, 1, 1] subgroup_size = 32>
//       CHECK: func.func @_sort1D_dispatch_0_sort_4xi32_dispatch_tensor_store()
//       CHECK:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.sort {
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [0], workgroup = [0]}>

// -----

module {
  func.func @_sort2D_dispatch_0_sort_2x4xi32_dispatch_tensor_store() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !flow.dispatch.tensor<readwrite:tensor<2x4xi32>>
    %1 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 4], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<2x4xi32>> -> tensor<2x4xi32>
    %2 = iree_linalg_ext.sort dimension(1) outs(%1 : tensor<2x4xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %3 = arith.cmpi slt, %arg0, %arg1 : i32
      iree_linalg_ext.yield %3 : i1
    } -> tensor<2x4xi32>
    flow.dispatch.tensor.store %2, %0, offsets = [0, 0], sizes = [2, 4], strides = [1, 1] : tensor<2x4xi32> -> !flow.dispatch.tensor<readwrite:tensor<2x4xi32>>
    return
  }
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
//       CHECK: func.func @_sort2D_dispatch_0_sort_2x4xi32_dispatch_tensor_store()
//       CHECK:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.sort {
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [1], workgroup = [2]}>

// -----

module {
  func.func @_sort3D_dispatch_0_sort_1x2x4xi32_dispatch_tensor_store() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !flow.dispatch.tensor<readwrite:tensor<1x2x4xi32>>
    %1 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 2, 4], strides = [1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<1x2x4xi32>> -> tensor<1x2x4xi32>
    %2 = iree_linalg_ext.sort dimension(2) outs(%1 : tensor<1x2x4xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %3 = arith.cmpi slt, %arg0, %arg1 : i32
      iree_linalg_ext.yield %3 : i1
    } -> tensor<1x2x4xi32>
    flow.dispatch.tensor.store %2, %0, offsets = [0, 0, 0], sizes = [1, 2, 4], strides = [1, 1, 1] : tensor<1x2x4xi32> -> !flow.dispatch.tensor<readwrite:tensor<1x2x4xi32>>
    return
  }
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
//       CHECK: func.func @_sort3D_dispatch_0_sort_1x2x4xi32_dispatch_tensor_store()
//       CHECK:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.sort {
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [1, 1], workgroup = [1, 2]}>

// -----

module {
  func.func @_sort_to_decreasing_seq_dispatch_0_sort_4xi32_dispatch_tensor_store() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !flow.dispatch.tensor<readwrite:tensor<4xi32>>
    %1 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<4xi32>> -> tensor<4xi32>
    %2 = iree_linalg_ext.sort dimension(0) outs(%1 : tensor<4xi32>) {
    ^bb0(%arg0: i32, %arg1: i32):
      %3 = arith.cmpi sgt, %arg0, %arg1 : i32
      iree_linalg_ext.yield %3 : i1
    } -> tensor<4xi32>
    flow.dispatch.tensor.store %2, %0, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !flow.dispatch.tensor<readwrite:tensor<4xi32>>
    return
  }
}

//       CHECK: #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [1, 1, 1] subgroup_size = 32>
//       CHECK-DAG: func.func @_sort_to_decreasing_seq_dispatch_0_sort_4xi32_dispatch_tensor_store()
//       CHECK:     translation_info = #[[$TRANSLATION]]
//       CHECK:   iree_linalg_ext.sort {
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config<{thread = [0], workgroup = [0]}>