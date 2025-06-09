// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-propagate-reshapes-by-expansion))" \
// RUN:   --split-input-file %s --mlir-print-local-scope | FileCheck %s

func.func @reshape_and_lowering_config(%src: tensor<3x4xf16>, %dest: tensor<12xf16>, %dest2: tensor<12xf16>) -> tensor<12xf16> {
  %collapse = tensor.collapse_shape %src [[0, 1]] : tensor<3x4xf16> into tensor<12xf16>
  %copy = linalg.copy ins(%collapse : tensor<12xf16>) outs(%dest: tensor<12xf16>) -> tensor<12xf16>
  %copy2 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%copy : tensor<12xf16>) outs(%dest2: tensor<12xf16>) -> tensor<12xf16>
  return %copy2: tensor<12xf16>
}

// CHECK-LABEL: func @reshape_and_lowering_config
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: tensor<3x4xf16>
//       CHECK:   %[[COPY1:.+]] = linalg.copy{{.*}} ins(%[[SRC]]
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[COPY1]]
//       CHECK:   linalg.copy
//  CHECK-SAME:     lowering_config = #iree_gpu.derived_thread_config
//  CHECK-SAME:     ins(%[[COLLAPSE]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">], flags = Indirect>
func.func @fold_collapse_into_loads_dynamic() -> tensor<?x32xf32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0}
  %2 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, %0, 32], strides = [1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0} -> tensor<2x?x32xf32>
  %3 = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  return %3 : tensor<?x32xf32>
}
// CHECK-LABEL: func @fold_collapse_into_loads_dynamic()
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%[[CONST]]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%[[SHAPE]]}
//       CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[SHAPE]], 32], strides = [1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">], flags = Indirect>
func.func @fold_expand_into_loads_dynamic() -> tensor<2x?x16x32xf32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0}
  %2 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, %0, 32], strides = [1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0} -> tensor<2x?x32xf32>
  %3 = affine.apply affine_map<()[s0] -> (s0 floordiv 2)>()[%0]
  %4 = tensor.expand_shape %2 [[0], [1, 2], [3]] output_shape [2, %3, 16, 32] : tensor<2x?x32xf32> into tensor<2x?x16x32xf32>
  return %4 : tensor<2x?x16x32xf32>
}
// CHECK-LABEL: func @fold_expand_into_loads_dynamic()
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = arith.divsi %[[CONST]], %[[C16]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x?x16x32xf32>>{%[[SHAPE]]}
//       CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [2, %[[SHAPE]], 16, 32], strides = [1, 1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x?x16x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 0], sizes = [%0, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic(
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = arith.divsi %[[CONST]], %[[C2]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0], sizes = [2, %[[SHAPE]], 32], strides = [1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic_rank_reduced_outer(%arg0 : tensor<2x?xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x?xf32> into tensor<?xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 0], sizes = [%0, 1], strides = [1, 1]
      : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic_rank_reduced_outer(
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = arith.divsi %[[CONST]], %[[C2]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0], sizes = [2, %[[SHAPE]], 1], strides = [1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic_rank_reduced_inner(%arg0 : tensor<2x?xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x?xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x?xf32> into tensor<?xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 0], sizes = [1, %0], strides = [1, 1]
      : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x?xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic_rank_reduced_inner(
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = arith.divsi %[[CONST]], %[[C2]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x2x?xf32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0], sizes = [1, 2, %[[SHAPE]]], strides = [1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x2x?xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic_partial_size(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x40xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 0], sizes = [%0, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x40xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic_partial_size(
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = arith.divsi %[[CONST]], %[[C2]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x40xf32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0], sizes = [2, %[[SHAPE]], 32], strides = [1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x40xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic_partial_with_offset(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x40xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 8], sizes = [%0, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x40xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic_partial_with_offset(
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = arith.divsi %[[CONST]], %[[C2]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x40xf32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 8], sizes = [2, %[[SHAPE]], 32], strides = [1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x40xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic_partial_unsupported(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [8, 0], sizes = [%0, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic_partial_unsupported(
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape
//  CHECK-NEXT:   iree_tensor_ext.dispatch.tensor.store %[[COLLAPSE]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic_partial_unsupported(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  %3 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  iree_tensor_ext.dispatch.tensor.store %3, %2, offsets = [0, 0], sizes = [%1, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic_partial_unsupported(
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape
//  CHECK-NEXT:   iree_tensor_ext.dispatch.tensor.store %[[COLLAPSE]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @expand_dest_forall() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %index = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x64x32xf32>>{%index}
  %1 = tensor.empty(%index) : tensor<?x64x32xf32>
  %extra = tensor.empty() : tensor<32x32xf32>
  %2 = scf.forall (%arg0, %arg1) = (0, 0) to (64, 32) step (16, 16)
    shared_outs(%arg2 = %1) -> (tensor<?x64x32xf32>) {
    %extracted_slice = tensor.extract_slice %arg2[%c0, %arg0, %arg1] [1, 16, 16] [1, 1, 1]
         : tensor<?x64x32xf32> to tensor<1x16x16xf32>
    %expanded = tensor.expand_shape %extracted_slice [[0], [1], [2, 3, 4]]
              output_shape [1, 16, 2, 4, 2] : tensor<1x16x16xf32> into tensor<1x16x2x4x2xf32>
    %expanded_barrier = util.optimization_barrier %expanded : tensor<1x16x2x4x2xf32>
    %collapsed = tensor.collapse_shape %expanded_barrier [[0], [1], [2, 3, 4]] : tensor<1x16x2x4x2xf32> into tensor<1x16x16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %collapsed into %arg2[%c0, %arg0, %arg1] [1, 16, 16] [1, 1, 1]
        : tensor<1x16x16xf32> into tensor<?x64x32xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  iree_tensor_ext.dispatch.tensor.store %2, %0, offsets = [0, 0, 0], sizes = [%index, 64, 32], strides = [1, 1, 1]
     : tensor<?x64x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x64x32xf32>>{%index}
  return
}

// CHECK-LABEL: func @expand_dest_forall(
//       CHECK:   %[[LOAD_CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[LOAD_CONST]]) : tensor<?x64x4x4x2xf32>
//       CHECK:   %[[SCFFORALL:.+]] = scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) = (0, 0)
//  CHECK-SAME:       shared_outs(%[[ARG2:.+]] = %[[EMPTY]]) -> (tensor<?x64x4x4x2xf32>) {
//   CHECK-DAG:     %[[OFFSET:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 8)>()[%[[ARG1]]]
//       CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG2]]
//  CHECK-SAME:         [0, %[[ARG0]], %[[OFFSET]], 0, 0] [1, 16, 2, 4, 2] [1, 1, 1, 1, 1]
//  CHECK-SAME:         tensor<?x64x4x4x2xf32> to tensor<1x16x2x4x2xf32>
//       CHECK:     %[[BARRIER:.+]] = util.optimization_barrier %[[EXTRACT]] : tensor<1x16x2x4x2xf32>
//       CHECK:     tensor.parallel_insert_slice %[[BARRIER]] into %[[ARG2]]
//  CHECK-SAME:         [0, %[[ARG0]], %[[OFFSET]], 0, 0] [1, 16, 2, 4, 2] [1, 1, 1, 1, 1]
//  CHECK-SAME:         tensor<1x16x2x4x2xf32> into tensor<?x64x4x4x2xf32>
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[SCFFORALL]], %[[SUBSPAN]]
//  CHECK-SAME:   offsets = [0, 0, 0, 0, 0], sizes = [%[[LOAD_CONST]], 64, 4, 4, 2], strides = [1, 1, 1, 1, 1]
//  CHECK-SAME:   !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x64x4x4x2xf32>>{%[[LOAD_CONST]]}

// -----
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @expand_dest_forall_multiresult() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64)
      offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x32xf32>>
  %2 = tensor.empty() : tensor<32xf32>
  %3 = tensor.empty() : tensor<32x32xf32>
  %4:2 = scf.forall (%arg0) = (0) to (32) step (16)
       shared_outs(%arg1 = %3, %arg2 = %2) -> (tensor<32x32xf32>, tensor<32xf32>) {
    %extracted_slice = tensor.extract_slice %arg2[%arg0] [16] [1] : tensor<32xf32> to tensor<16xf32>
    %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [2, 8]
                : tensor<16xf32> into tensor<2x8xf32>
    %5 = util.optimization_barrier %expanded : tensor<2x8xf32>
    %collapsed = tensor.collapse_shape %5 [[0, 1]] : tensor<2x8xf32> into tensor<16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg1 into %arg1[%c0, %c0] [32, 32] [1, 1]
          : tensor<32x32xf32> into tensor<32x32xf32>
      tensor.parallel_insert_slice %collapsed into %arg2[%arg0] [16] [1]
          : tensor<16xf32> into tensor<32xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>]}
  iree_tensor_ext.dispatch.tensor.store %4#1, %0, offsets = [0], sizes = [32], strides = [1]
    : tensor<32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
  iree_tensor_ext.dispatch.tensor.store %4#0, %1, offsets = [0, 0], sizes = [32, 32], strides = [1, 1]
    : tensor<32x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x32xf32>>
  return
}

// CHECK-LABEL: func @expand_dest_forall_multiresult(
//       CHECK:   %[[SUBSPAN0:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[SUBSPAN1:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[EMPTY0:.+]] = tensor.empty() : tensor<32x32xf32>
//       CHECK:   %[[EMPTY1:.+]] = tensor.empty() : tensor<4x8xf32>
//       CHECK:   %[[SCFFORALL:.+]]:2 = scf.forall (%[[ARG0:.+]]) = (0) to (32) step (16)
//  CHECK-SAME:       shared_outs(%[[ARG1:.+]] = %[[EMPTY0]], %[[ARG2:.+]] = %[[EMPTY1]])
//  CHECK-SAME:       -> (tensor<32x32xf32>, tensor<4x8xf32>) {
//   CHECK-DAG:     %[[OFFSET:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 8)>()[%[[ARG0]]]
//       CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG2]]
//  CHECK-SAME:         [%[[OFFSET]], 0] [2, 8] [1, 1]
//  CHECK-SAME:         tensor<4x8xf32> to tensor<2x8xf32>
//       CHECK:     %[[BARRIER:.+]] = util.optimization_barrier %[[EXTRACT]] : tensor<2x8xf32>
//       CHECK:     tensor.parallel_insert_slice %[[ARG1]] into %[[ARG1]]
//  CHECK-SAME:         tensor<32x32xf32> into tensor<32x32xf32>
//       CHECK:     tensor.parallel_insert_slice %[[BARRIER]] into %[[ARG2]]
//  CHECK-SAME:         [%[[OFFSET]], 0] [2, 8] [1, 1]
//  CHECK-SAME:         tensor<2x8xf32> into tensor<4x8xf32>
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[SCFFORALL]]#1, %[[SUBSPAN0]]
//  CHECK-SAME:   offsets = [0, 0], sizes = [4, 8], strides = [1, 1]
//  CHECK-SAME:   !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x8xf32>>
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[SCFFORALL]]#0, %[[SUBSPAN1]]
//  CHECK-SAME:   offsets = [0, 0], sizes = [32, 32], strides = [1, 1]
//  CHECK-SAME:   !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x32xf32>>


// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @noexpand_dest_forall_dynamicpacked() {
  %index1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %index2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %index3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
  %2 = tensor.empty() : tensor<32xf32>
  %4 = scf.forall (%arg0) = (0) to (32) step (16)
       shared_outs(%arg2 = %2) -> (tensor<32xf32>) {
    %extracted_slice = tensor.extract_slice %arg2[%arg0] [%index1] [1] : tensor<32xf32> to tensor<?xf32>
    %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [%index2, %index3]
                : tensor<?xf32> into tensor<?x?xf32>
    %5 = util.optimization_barrier %expanded : tensor<?x?xf32>
    %collapsed = tensor.collapse_shape %5 [[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %collapsed into %arg2[%arg0] [%index1] [1]
          : tensor<?xf32> into tensor<32xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>]}
  iree_tensor_ext.dispatch.tensor.store %4, %0, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32>
    -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
  return
}

// CHECK-LABEL: func @noexpand_dest_forall_dynamicpacked(
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<32xf32>
//       CHECK:   %[[SCFFORALL:.+]] = scf.forall (%[[ARG0:.+]]) = (0) to (32) step (16)
//  CHECK-SAME:       shared_outs(%[[ARG2:.+]] = %[[EMPTY]]) -> (tensor<32xf32>) {
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[SCFFORALL]], %[[SUBSPAN]]
//  CHECK-SAME:   offsets = [0], sizes = [32], strides = [1] : tensor<32xf32>
//  CHECK-SAME:   !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>

// -----
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @expand_dest_forall_unsupporteduse() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
  %2 = tensor.empty() : tensor<32xf32>
  %4 = scf.forall (%arg0) = (0) to (32) step (16)
       shared_outs(%arg2 = %2) -> (tensor<32xf32>) {
    %extracted_slice = tensor.extract_slice %arg2[%arg0] [16] [1] : tensor<32xf32> to tensor<16xf32>
    %arith_op = arith.negf %extracted_slice : tensor<16xf32>
    %expanded = tensor.expand_shape %arith_op [[0, 1]] output_shape [2, 8]
                : tensor<16xf32> into tensor<2x8xf32>
    %5 = util.optimization_barrier %expanded : tensor<2x8xf32>
    %collapsed = tensor.collapse_shape %5 [[0, 1]] : tensor<2x8xf32> into tensor<16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %collapsed into %arg2[%arg0] [16] [1]
          : tensor<16xf32> into tensor<32xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>]}
  iree_tensor_ext.dispatch.tensor.store %4, %0, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
  return
}

// CHECK-LABEL: func @expand_dest_forall_unsupporteduse(
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<32xf32>
//       CHECK:   %[[SCFFORALL:.+]] = scf.forall (%[[ARG0:.+]]) = (0) to (32) step (16)
//  CHECK-SAME:       shared_outs(%[[ARG2:.+]] = %[[EMPTY]]) -> (tensor<32xf32>) {
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[SCFFORALL]], %[[SUBSPAN]]
//  CHECK-SAME:   offsets = [0], sizes = [32], strides = [1] : tensor<32xf32>
//  CHECK-SAME:   !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>


// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @noexpand_dest_forall_nomapping() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
  %2 = tensor.empty() : tensor<32xf32>
  %4 = scf.forall (%arg0) = (0) to (32) step (16)
       shared_outs(%arg2 = %2) -> (tensor<32xf32>) {
    %extracted_slice = tensor.extract_slice %arg2[%arg0] [16] [1] : tensor<32xf32> to tensor<16xf32>
    %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [2, 8]
                : tensor<16xf32> into tensor<2x8xf32>
    %5 = util.optimization_barrier %expanded : tensor<2x8xf32>
    %collapsed = tensor.collapse_shape %5 [[0, 1]] : tensor<2x8xf32> into tensor<16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %collapsed into %arg2[%arg0] [16] [1]
          : tensor<16xf32> into tensor<32xf32>
    }
  }
  iree_tensor_ext.dispatch.tensor.store %4, %0, offsets = [0], sizes = [32], strides = [1] : tensor<32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>
  return
}

// CHECK-LABEL: func @noexpand_dest_forall_nomapping(
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<32xf32>
//       CHECK:   %[[SCFFORALL:.+]] = scf.forall (%[[ARG0:.+]]) = (0) to (32) step (16)
//  CHECK-SAME:       shared_outs(%[[ARG2:.+]] = %[[EMPTY]]) -> (tensor<32xf32>) {
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[SCFFORALL]], %[[SUBSPAN]]
//  CHECK-SAME:   offsets = [0], sizes = [32], strides = [1] : tensor<32xf32>
//  CHECK-SAME:   !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32xf32>>


// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @noexpand_dest_forall_notfullslicestore() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<34xf32>>
  %2 = tensor.empty() : tensor<32xf32>
  %4 = scf.forall (%arg0) = (0) to (32) step (16)
       shared_outs(%arg2 = %2) -> (tensor<32xf32>) {
    %extracted_slice = tensor.extract_slice %arg2[%arg0] [16] [1] : tensor<32xf32> to tensor<16xf32>
    %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [2, 8]
                : tensor<16xf32> into tensor<2x8xf32>
    %5 = util.optimization_barrier %expanded : tensor<2x8xf32>
    %collapsed = tensor.collapse_shape %5 [[0, 1]] : tensor<2x8xf32> into tensor<16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %collapsed into %arg2[%arg0] [16] [1]
          : tensor<16xf32> into tensor<32xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>]}
  iree_tensor_ext.dispatch.tensor.store %4, %0, offsets = [1], sizes = [32], strides = [1] : tensor<32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<34xf32>>
  return
}

// CHECK-LABEL: func @noexpand_dest_forall_notfullslicestore(
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<32xf32>
//       CHECK:   %[[SCFFORALL:.+]] = scf.forall (%[[ARG0:.+]]) = (0) to (32) step (16)
//  CHECK-SAME:       shared_outs(%[[ARG2:.+]] = %[[EMPTY]]) -> (tensor<32xf32>) {
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[SCFFORALL]], %[[SUBSPAN]]
//  CHECK-SAME:   offsets = [1], sizes = [32], strides = [1] : tensor<32xf32>
//  CHECK-SAME:   !iree_tensor_ext.dispatch.tensor<writeonly:tensor<34xf32>>

// -----
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @expand_dest_forall_chained() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %index = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x64x32xf32>>{%index}
  %1 = tensor.empty(%index) : tensor<?x64x32xf32>
  %extra = tensor.empty() : tensor<32x32xf32>
  %2 = scf.forall (%arg0, %arg1) = (0, 0) to (64, 32) step (16, 16)
    shared_outs(%arg2 = %1) -> (tensor<?x64x32xf32>) {
    %extracted_slice = tensor.extract_slice %arg2[%c0, %arg0, %arg1] [1, 16, 16] [1, 1, 1]
         : tensor<?x64x32xf32> to tensor<1x16x16xf32>
    %expanded = tensor.expand_shape %extracted_slice [[0], [1], [2, 3, 4]]
              output_shape [1, 16, 2, 4, 2] : tensor<1x16x16xf32> into tensor<1x16x2x4x2xf32>
    %expanded2 = tensor.expand_shape %expanded [[0], [1, 2], [3], [4], [5]]
              output_shape [1, 8, 2, 2, 4, 2] : tensor<1x16x2x4x2xf32> into tensor<1x8x2x2x4x2xf32>
    %expanded_barrier = util.optimization_barrier %expanded2 : tensor<1x8x2x2x4x2xf32>
    %collapsed = tensor.collapse_shape %expanded_barrier [[0], [1, 2], [3], [4], [5]] :  tensor<1x8x2x2x4x2xf32> into tensor<1x16x2x4x2xf32>
    %collapsed2 = tensor.collapse_shape %collapsed [[0], [1], [2, 3, 4]] : tensor<1x16x2x4x2xf32> into tensor<1x16x16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %collapsed2 into %arg2[%c0, %arg0, %arg1] [1, 16, 16] [1, 1, 1]
        : tensor<1x16x16xf32> into tensor<?x64x32xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  iree_tensor_ext.dispatch.tensor.store %2, %0, offsets = [0, 0, 0], sizes = [%index, 64, 32], strides = [1, 1, 1]
     : tensor<?x64x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x64x32xf32>>{%index}
  return
}

// CHECK-LABEL: func @expand_dest_forall_chained(
//       CHECK:   %[[LOAD_CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[LOAD_CONST]]) : tensor<?x32x2x4x4x2xf32>
//       CHECK:   %[[SCFFORALL:.+]] = scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) = (0, 0)
//  CHECK-SAME:       shared_outs(%[[ARG2:.+]] = %[[EMPTY]]) -> (tensor<?x32x2x4x4x2xf32>) {
//   CHECK-DAG:     %[[OFFSET0:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 8)>()[%[[ARG1]]]
//   CHECK-DAG:     %[[OFFSET1:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 2)>()[%[[ARG0]]]
//       CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG2]]
//  CHECK-SAME:         [0, %[[OFFSET1]], 0, %[[OFFSET0]], 0, 0] [1, 8, 2, 2, 4, 2] [1, 1, 1, 1, 1, 1]
//  CHECK-SAME:         tensor<?x32x2x4x4x2xf32> to tensor<1x8x2x2x4x2xf32>
//       CHECK:     %[[BARRIER:.+]] = util.optimization_barrier %[[EXTRACT]] : tensor<1x8x2x2x4x2xf32>
//       CHECK:     tensor.parallel_insert_slice %[[BARRIER]] into %[[ARG2]]
//  CHECK-SAME:         [0, %[[OFFSET1]], 0, %[[OFFSET0]], 0, 0] [1, 8, 2, 2, 4, 2] [1, 1, 1, 1, 1, 1]
//  CHECK-SAME:         tensor<1x8x2x2x4x2xf32> into tensor<?x32x2x4x4x2xf32>
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[SCFFORALL]], %[[SUBSPAN]]
//  CHECK-SAME:   offsets = [0, 0, 0, 0, 0, 0], sizes = [%[[LOAD_CONST]], 32, 2, 4, 4, 2], strides = [1, 1, 1, 1, 1, 1]
//  CHECK-SAME:   !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32x2x4x4x2xf32>>{%[[LOAD_CONST]]}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @expand_dest_forall_no_crash_issue_20736(%arg0: tensor<16x8x48x32x3x96xbf16>, %arg1: tensor<3x96x1x3x3x96xbf16>) {
  // IR that previously crashed during pattern application.
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x8x48x32x3x96xbf16>>
  %1 = tensor.empty() : tensor<16x8x48x32x3x96xbf16>
  %2 = scf.forall (%arg2, %arg3, %arg4, %arg5, %arg6) = (0, 0, 0, 0, 0) to (16, 8, 48, 3, 96) step (1, 1, 16, 1, 32) shared_outs(%arg7 = %1) -> (tensor<16x8x48x32x3x96xbf16>) {
    %extracted_slice = tensor.extract_slice %arg7[%arg2, %arg3, %arg4, 0, %arg5, %arg6] [1, 1, 16, 32, 1, 32] [1, 1, 1, 1, 1, 1] : tensor<16x8x48x32x3x96xbf16> to tensor<1x1x16x32x1x32xbf16>
    %expanded = tensor.expand_shape %extracted_slice [[0], [1], [2], [3, 4], [5], [6, 7]] output_shape [1, 1, 16, 2, 16, 1, 2, 16] : tensor<1x1x16x32x1x32xbf16> into tensor<1x1x16x2x16x1x2x16xbf16>
    %3 = util.optimization_barrier %expanded : tensor<1x1x16x2x16x1x2x16xbf16>
    %collapsed = tensor.collapse_shape %3 [[0], [1], [2], [3, 4], [5], [6, 7]] : tensor<1x1x16x2x16x1x2x16xbf16> into tensor<1x1x16x32x1x32xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %collapsed into %arg7[%arg2, %arg3, %arg4, 0, %arg5, %arg6] [1, 1, 16, 32, 1, 32] [1, 1, 1, 1, 1, 1] : tensor<1x1x16x32x1x32xbf16> into tensor<16x8x48x32x3x96xbf16>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<z:2>, #iree_codegen.workgroup_mapping<z:1>, #iree_codegen.workgroup_mapping<z>, #iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  iree_tensor_ext.dispatch.tensor.store %2, %0, offsets = [0, 0, 0, 0, 0, 0], sizes = [16, 8, 48, 32, 3, 96], strides = [1, 1, 1, 1, 1, 1] : tensor<16x8x48x32x3x96xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x8x48x32x3x96xbf16>>
  return
}
// CHECK-LABEL: func @expand_dest_forall_no_crash_issue_20736(
//       CHECK:   scf.forall
//   CHECK-NOT:     tensor.collapse_shape
//       CHECK:     tensor.parallel_insert_slice
