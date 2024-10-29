// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-propagate-reshapes-by-expansion), cse)" \
// RUN:   --split-input-file %s --mlir-print-local-scope | FileCheck %s

func.func @reshape_and_lowering_config(%src: tensor<3x4xf16>, %dest: tensor<12xf16>, %dest2: tensor<12xf16>) -> tensor<12xf16> {
  %collapse = tensor.collapse_shape %src [[0, 1]] : tensor<3x4xf16> into tensor<12xf16>
  %copy = linalg.copy ins(%collapse : tensor<12xf16>) outs(%dest: tensor<12xf16>) -> tensor<12xf16>
  %copy2 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%copy : tensor<12xf16>) outs(%dest2: tensor<12xf16>) -> tensor<12xf16>
  return %copy2: tensor<12xf16>
}

// CHECK-LABEL: func @reshape_and_lowering_config
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: tensor<3x4xf16>
//       CHECK:   %[[COPY1:.+]] = linalg.generic {{.*}} ins(%[[SRC]]
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
      flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0}
  %2 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, %0, 32], strides = [1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0} -> tensor<2x?x32xf32>
  %3 = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  return %3 : tensor<?x32xf32>
}
// CHECK-LABEL: func @fold_collapse_into_loads_dynamic()
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%[[CONST]]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x32xf32>>{%[[SHAPE]]}
//       CHECK:   %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[SHAPE]], 32], strides = [1, 1]
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">], flags = Indirect>
func.func @fold_expand_into_loads_dynamic() -> tensor<2x?x16x32xf32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0}
  %2 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, %0, 32], strides = [1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0} -> tensor<2x?x32xf32>
  %3 = affine.apply affine_map<()[s0] -> (s0 floordiv 2)>()[%0]
  %4 = tensor.expand_shape %2 [[0], [1, 2], [3]] output_shape [2, %3, 16, 32] : tensor<2x?x32xf32> into tensor<2x?x16x32xf32>
  return %4 : tensor<2x?x16x32xf32>
}
// CHECK-LABEL: func @fold_expand_into_loads_dynamic()
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = arith.divui %[[CONST]], %[[C16]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<2x?x16x32xf32>>{%[[SHAPE]]}
//       CHECK:   %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [2, %[[SHAPE]], 16, 32], strides = [1, 1, 1, 1]
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<2x?x16x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !flow.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  flow.dispatch.tensor.store %2, %1, offsets = [0, 0], sizes = [%0, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic(
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = arith.divui %[[CONST]], %[[C2]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !flow.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}
//       CHECK:   flow.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0], sizes = [2, %[[SHAPE]], 32], strides = [1, 1, 1]
//  CHECK-SAME:       !flow.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @expand_dest_forall_workgroup_mapped() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  %1 = tensor.empty() : tensor<2048x10240xf32>
  %2 = scf.forall (%arg0, %arg1) = (0, 0) to (2048, 10240) step (128, 128)
    shared_outs(%arg2 = %1) -> (tensor<2048x10240xf32>) {
    %extracted_slice = tensor.extract_slice %arg2[%arg0, %arg1] [128, 128] [1, 1]
         : tensor<2048x10240xf32> to tensor<128x128xf32>
    %3 = tensor.empty() : tensor<8x8x16x8x2xf32>
    %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<8x8x16x8x2xf32>) -> tensor<8x8x16x8x2xf32>
    %5 = tensor.empty() : tensor<8x16x8x8x2xf32>
    %transposed = linalg.transpose ins(%4 : tensor<8x8x16x8x2xf32>)
        outs(%5 : tensor<8x16x8x8x2xf32>) permutation = [0, 2, 1, 3, 4]
    %expanded = tensor.expand_shape %extracted_slice [[0, 1], [2, 3, 4]]
              output_shape [8, 16, 8, 8, 2] : tensor<128x128xf32> into tensor<8x16x8x8x2xf32>
    %6 = linalg.copy ins(%transposed : tensor<8x16x8x8x2xf32>)
         outs(%expanded : tensor<8x16x8x8x2xf32>) -> tensor<8x16x8x8x2xf32>
    %collapsed = tensor.collapse_shape %6 [[0, 1], [2, 3, 4]] : tensor<8x16x8x8x2xf32> into tensor<128x128xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %collapsed into %arg2[%arg0, %arg1] [128, 128] [1, 1]
        : tensor<128x128xf32> into tensor<2048x10240xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  flow.dispatch.tensor.store %2, %0, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1]
     : tensor<2048x10240xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  return
}

// CHECK-LABEL: func @expand_dest_forall_workgroup_mapped(
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<128x16x640x8x2xf32>
//       CHECK:   %[[SCFFORALL:.+]] = scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) = (0, 0)
//  CHECK-SAME:       shared_outs(%[[ARG2:.+]] = %[[EMPTY]]) -> (tensor<128x16x640x8x2xf32>) {
//   CHECK-DAG:     %[[OFFSET1:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%[[ARG0]]]
//   CHECK-DAG:     %[[OFFSET2:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%[[ARG1]]]
//       CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG2]]
//  CHECK-SAME:         [%[[OFFSET1]], 0, %[[OFFSET2]], 0, 0] [8, 16, 8, 8, 2] [1, 1, 1, 1, 1]
//  CHECK-SAME:         tensor<128x16x640x8x2xf32> to tensor<8x16x8x8x2xf32>
//       CHECK:     tensor.parallel_insert_slice %{{.+}} into %[[ARG2]]
//  CHECK-SAME:         [%[[OFFSET1]], 0, %[[OFFSET2]], 0, 0] [8, 16, 8, 8, 2] [1, 1, 1, 1, 1]
//  CHECK-SAME:         tensor<8x16x8x8x2xf32> into tensor<128x16x640x8x2xf32>
//       CHECK:   flow.dispatch.tensor.store %[[SCFFORALL]], %[[SUBSPAN]]
//  CHECK-SAME:   offsets = [0, 0, 0, 0, 0], sizes = [128, 16, 640, 8, 2], strides = [1, 1, 1, 1, 1]
//  CHECK-SAME:   !flow.dispatch.tensor<writeonly:tensor<128x16x640x8x2xf32>>

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @no_expand_dest_forall_not_workgroup_mapped() {
  %cst = arith.constant 0.000000e+00 : f16
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  %1 = tensor.empty() : tensor<2048x10240xf32>
  %2 = scf.forall (%arg0, %arg1) = (0, 0) to (2048, 10240) step (128, 128)
    shared_outs(%arg2 = %1) -> (tensor<2048x10240xf32>) {
    %extracted_slice = tensor.extract_slice %arg2[%arg0, %arg1] [128, 128] [1, 1]
         : tensor<2048x10240xf32> to tensor<128x128xf32>
    %3 = tensor.empty() : tensor<8x8x16x8x2xf32>
    %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<8x8x16x8x2xf32>) -> tensor<8x8x16x8x2xf32>
    %5 = tensor.empty() : tensor<8x16x8x8x2xf32>
    %transposed = linalg.transpose ins(%4 : tensor<8x8x16x8x2xf32>)
        outs(%5 : tensor<8x16x8x8x2xf32>) permutation = [0, 2, 1, 3, 4]
    %expanded = tensor.expand_shape %extracted_slice [[0, 1], [2, 3, 4]]
              output_shape [8, 16, 8, 8, 2] : tensor<128x128xf32> into tensor<8x16x8x8x2xf32>
    %6 = linalg.copy ins(%transposed : tensor<8x16x8x8x2xf32>)
         outs(%expanded : tensor<8x16x8x8x2xf32>) -> tensor<8x16x8x8x2xf32>
    %collapsed = tensor.collapse_shape %6 [[0, 1], [2, 3, 4]] : tensor<8x16x8x8x2xf32> into tensor<128x128xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %collapsed into %arg2[%arg0, %arg1] [128, 128] [1, 1]
        : tensor<128x128xf32> into tensor<2048x10240xf32>
    }
  }
  flow.dispatch.tensor.store %2, %0, offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1]
     : tensor<2048x10240xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
  return
}

// CHECK-LABEL: func @no_expand_dest_forall_not_workgroup_mapped(
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<2048x10240xf32>
//       CHECK:   %[[SCFFORALL:.+]] = scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) = (0, 0)
//  CHECK-SAME:       shared_outs(%[[ARG2:.+]] = %[[EMPTY]]) -> (tensor<2048x10240xf32>) {
//       CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG2]]
//  CHECK-SAME:         [%[[ARG0]], %[[ARG1]]] [128, 128] [1, 1]
//  CHECK-SAME:         tensor<2048x10240xf32> to tensor<128x128xf32>
//       CHECK:     tensor.parallel_insert_slice %{{.+}} into %[[ARG2]]
//  CHECK-SAME:         [%[[ARG0]], %[[ARG1]]] [128, 128] [1, 1]
//  CHECK-SAME:         tensor<128x128xf32> into tensor<2048x10240xf32>
//       CHECK:   flow.dispatch.tensor.store %[[SCFFORALL]], %[[SUBSPAN]]
//  CHECK-SAME:   offsets = [0, 0], sizes = [2048, 10240], strides = [1, 1]
//  CHECK-SAME:   !flow.dispatch.tensor<writeonly:tensor<2048x10240xf32>>
