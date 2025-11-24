// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-type-propagation))" --split-input-file %s \
// RUN: | FileCheck %s --implicit-check-not=packed_storage

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @generic_op_illegal_operand() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d} -> tensor<?xi1, #iree_encoding.packed_storage>
  %4 = tensor.empty(%d) : tensor<?xi8>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%2 : tensor<?xi1, #iree_encoding.packed_storage>) outs(%4 : tensor<?xi8>) {
      ^bb0(%arg0 : i1, %arg1 : i8):
        %6 = arith.extui %arg0 : i1 to i8
        linalg.yield %6 : i8
    } -> tensor<?xi8>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: @generic_op_illegal_operand
// CHECK-DAG:   %[[D:.*]] = hal.interface.constant.load
// CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan {{.*}} binding(0)
// CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan {{.*}} binding(1)
// CHECK-DAG:   %[[INTENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN]]
// CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%[[D]]) : tensor<?xi8>
// CHECK:       %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[INTENSOR]] : tensor<?xi1>) outs(%[[INIT]] : tensor<?xi8>)
// CHECK:       ^bb0(%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i8):
// CHECK:         %[[EXTUI:.+]] = arith.extui %[[ARG0]] : i1 to i8
// CHECK:         linalg.yield %[[EXTUI]] : i8
// CHECK:       iree_tensor_ext.dispatch.tensor.store %[[GENERIC]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @generic_op_illegal_result() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %3 = tensor.empty(%d) : tensor<?xi1, #iree_encoding.packed_storage>
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%2 : tensor<?xi8>) outs(%3 : tensor<?xi1, #iree_encoding.packed_storage>) {
      ^bb0(%arg0 : i8, %arg1 : i1):
        %5 = arith.trunci %arg0 : i8 to i1
        linalg.yield %5 : i1
    } -> tensor<?xi1, #iree_encoding.packed_storage>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi1, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  return
}
// CHECK-LABEL: @generic_op_illegal_result
// CHECK-DAG:   %[[D:.*]] = hal.interface.constant.load
// CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan {{.*}} binding(0)
// CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan {{.*}} binding(1)
// CHECK-DAG:   %[[INTENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN]]
// CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%[[D]]) : tensor<?xi1>
// CHECK:       %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[INTENSOR]] : tensor<?xi8>) outs(%[[INIT]] : tensor<?xi1>)
// CHECK:       ^bb0(%[[ARG0:.*]]: i8, %[[ARG1:.*]]: i1):
// CHECK:         %[[TRUNC:.+]] = arith.trunci %[[ARG0]] : i8 to i1
// CHECK:         linalg.yield %[[TRUNC]] : i1
// CHECK:       iree_tensor_ext.dispatch.tensor.store %[[GENERIC]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @tensor_extract() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %offset = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %size = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d} -> tensor<?xi1, #iree_encoding.packed_storage>
  %3 = tensor.extract_slice %2[%offset] [%size] [1] : tensor<?xi1, #iree_encoding.packed_storage> to tensor<?xi1, #iree_encoding.packed_storage>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [%offset], sizes=[%size], strides=[1] : tensor<?xi1, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  return
}
// CHECK-LABEL: @tensor_extract
// CHECK-DAG:   %[[D:.*]] = hal.interface.constant.load
// CHECK-DAG:   %[[OFFSET:.*]] = hal.interface.constant.load
// CHECK-DAG:   %[[SIZE:.*]] = hal.interface.constant.load
// CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan {{.*}} binding(0)
// CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan {{.*}} binding(1)
// CHECK-DAG:   %[[INTENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN]]
// CHECK:       %[[EXTRACT:.+]] = tensor.extract_slice %[[INTENSOR]]{{\[}}%[[OFFSET]]] {{\[}}%[[SIZE]]] [1]
// CHECK:       iree_tensor_ext.dispatch.tensor.store %[[EXTRACT]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @tensor_insert() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %offset = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %size = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [%offset], sizes=[%size], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d} -> tensor<?xi1, #iree_encoding.packed_storage>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d} -> tensor<?xi1, #iree_encoding.packed_storage>
  %5 = tensor.insert_slice %3 into %4[%offset] [%size] [1] : tensor<?xi1, #iree_encoding.packed_storage> into tensor<?xi1, #iree_encoding.packed_storage>
  iree_tensor_ext.dispatch.tensor.store %5, %2, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi1, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  return
}
// CHECK-LABEL: @tensor_insert
// CHECK-DAG:   %[[D:.*]] = hal.interface.constant.load
// CHECK-DAG:   %[[OFFSET:.*]] = hal.interface.constant.load
// CHECK-DAG:   %[[SIZE:.*]] = hal.interface.constant.load
// CHECK-DAG:   %[[IN1:.+]] = hal.interface.binding.subspan {{.*}} binding(0)
// CHECK-DAG:   %[[IN2:.+]] = hal.interface.binding.subspan {{.*}} binding(1)
// CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan {{.*}} binding(2)
// CHECK-DAG:   %[[IN1TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN1]]
// CHECK-DAG:   %[[IN2TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN2]]
// CHECK:       %[[INSERT:.+]] = tensor.insert_slice %[[IN1TENSOR]] into %[[IN2TENSOR]]{{\[}}%[[OFFSET]]] {{\[}}%[[SIZE]]] [1]
// CHECK:       iree_tensor_ext.dispatch.tensor.store %[[INSERT]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @for_loop() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %lb = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %step = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets=[0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d} -> tensor<?xi1, #iree_encoding.packed_storage>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets=[0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d} -> tensor<?xi1, #iree_encoding.packed_storage>
  %c0 = arith.constant 0 : index
  %4 = scf.for %arg0 = %c0 to %d step %step iter_args(%arg1 = %3) -> tensor<?xi1, #iree_encoding.packed_storage> {
    %7 = tensor.extract_slice %2[%arg0][%step][1] : tensor<?xi1, #iree_encoding.packed_storage> to tensor<?xi1, #iree_encoding.packed_storage>
    %8 = tensor.insert_slice %7 into %arg1[%arg0][%step][1] : tensor<?xi1, #iree_encoding.packed_storage> into tensor<?xi1, #iree_encoding.packed_storage>
    scf.yield %8 : tensor<?xi1, #iree_encoding.packed_storage>
  }
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets=[0], sizes=[%d], strides=[1]: tensor<?xi1, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  return
}
// CHECK-LABEL: @for_loop
// CHECK-DAG:   %[[D:.*]] = hal.interface.constant.load
// CHECK-DAG:   %[[LB:.*]] = hal.interface.constant.load
// CHECK-DAG:   %[[STEP:.*]] = hal.interface.constant.load
// CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan {{.*}} binding(0)
// CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan {{.*}} binding(1)
// CHECK-DAG:   %[[INTENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN]]
// CHECK-DAG:   %[[OUTTENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUT]]
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[FOR:.+]] = scf.for %[[ARG0:.*]] = %[[C0]] to %[[D]] step %[[STEP]] iter_args(%[[ARG1:.+]] = %[[OUTTENSOR]])
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[INTENSOR]]{{\[}}%[[ARG0]]] {{\[}}%[[STEP]]] [1]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[SLICE]] into %[[ARG1]]{{\[}}%[[ARG0]]] {{\[}}%[[STEP]]] [1]
// CHECK:         scf.yield %[[INSERT]]
// CHECK:       iree_tensor_ext.dispatch.tensor.store %[[FOR]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @fill_op() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  %1 = tensor.empty(%d) : tensor<?xi1, #iree_encoding.packed_storage>
  %false = arith.constant false
  %2 = linalg.fill ins(%false : i1) outs(%1 : tensor<?xi1, #iree_encoding.packed_storage>) -> tensor<?xi1, #iree_encoding.packed_storage>
  iree_tensor_ext.dispatch.tensor.store %2, %0, offsets=[0], sizes=[%d], strides=[1] : tensor<?xi1, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi1, #iree_encoding.packed_storage>>{%d}
  return
}
// CHECK-LABEL: @fill_op
// CHECK-DAG:   %[[D:.*]] = hal.interface.constant.load
// CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan {{.*}} binding(0)
// CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%[[D]]) : tensor<?xi1>
// CHECK-DAG:   %[[FALSE:.+]] = arith.constant false
// CHECK:       %[[FILL:.+]] = linalg.fill ins(%[[FALSE]] : i1) outs(%[[INIT]] : tensor<?xi1>)
// CHECK:       iree_tensor_ext.dispatch.tensor.store %[[FILL]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @tensor_extract() {
  %c0 = arith.constant 0 : index
  %c13 = arith.constant 13 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<14xi1, #iree_encoding.packed_storage>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14xi1, #iree_encoding.packed_storage>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [14], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<14xi1, #iree_encoding.packed_storage>> -> tensor<14xi1, #iree_encoding.packed_storage>
  %3 = tensor.empty() : tensor<14xi1, #iree_encoding.packed_storage>
  %4 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      outs(%3 : tensor<14xi1, #iree_encoding.packed_storage>) {
  ^bb0(%out: i1):
    %5 = linalg.index 0 : index
    %6 = arith.subi %c13, %5 : index
    %extracted = tensor.extract %3[%6] : tensor<14xi1, #iree_encoding.packed_storage>
    linalg.yield %extracted : i1
  } -> tensor<14xi1, #iree_encoding.packed_storage>
  iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0], sizes = [14], strides = [1]
      : tensor<14xi1, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14xi1, #iree_encoding.packed_storage>>
  return
}
// CHECK-LABEL: @tensor_extract
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C13:.*]] = arith.constant 13 : index
// CHECK-DAG:   %[[BINDING:.+]] = hal.interface.binding.subspan {{.*}} binding(0)
// CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan {{.*}} binding(1)
// CHECK-DAG:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[BINDING]]
// CHECK-DAG:   %[[INIT:.+]] = tensor.empty() : tensor<14xi1>
// CHECK:       %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:    outs(%[[INIT]] : tensor<14xi1>)
// CHECK:       ^bb0(%[[OUT_ARG:.*]]: i1):
// CHECK:         %[[INDEX:.+]] = linalg.index 0 : index
// CHECK:         %[[SUBI:.+]] = arith.subi %[[C13]], %[[INDEX]] : index
// CHECK:         %[[EXTRACTED:.+]] = tensor.extract %[[INIT]]{{\[}}%[[SUBI]]] : tensor<14xi1>
// CHECK:         linalg.yield %[[EXTRACTED]] : i1
// CHECK:       iree_tensor_ext.dispatch.tensor.store %[[GENERIC]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @named_op() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8xi1, #iree_encoding.packed_storage>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8xi1, #iree_encoding.packed_storage>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8x8xi1, #iree_encoding.packed_storage>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8xi1, #iree_encoding.packed_storage>> -> tensor<8x8xi1, #iree_encoding.packed_storage>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8xi1, #iree_encoding.packed_storage>> -> tensor<8x8xi1, #iree_encoding.packed_storage>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8x8xi1, #iree_encoding.packed_storage>> -> tensor<8x8xi1, #iree_encoding.packed_storage>
  %gemm = linalg.matmul ins(%3, %4 : tensor<8x8xi1, #iree_encoding.packed_storage>, tensor<8x8xi1, #iree_encoding.packed_storage>)
      outs(%5 : tensor<8x8xi1, #iree_encoding.packed_storage>) -> tensor<8x8xi1, #iree_encoding.packed_storage>
  iree_tensor_ext.dispatch.tensor.store %gemm, %2, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : tensor<8x8xi1, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8x8xi1, #iree_encoding.packed_storage>>
  return
}
// CHECK-LABEL: @named_op
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ARG0:.+]] = hal.interface.binding.subspan {{.*}} binding(0)
// CHECK-DAG:   %[[ARG1:.+]] = hal.interface.binding.subspan {{.*}} binding(1)
// CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan {{.*}} binding(2)
// CHECK-DAG:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ARG0]]
// CHECK-DAG:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ARG1]]
// CHECK-DAG:   %[[INIT:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUT]]
// CHECK:       %[[GEMM:.+]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<8x8xi1>, tensor<8x8xi1>) outs(%[[INIT]] : tensor<8x8xi1>)
// CHECK:       iree_tensor_ext.dispatch.tensor.store %[[GEMM]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @scatter() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8xi1, #iree_encoding.packed_storage>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x1xi32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3xi1, #iree_encoding.packed_storage>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [8], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8xi1, #iree_encoding.packed_storage>> -> tensor<8xi1, #iree_encoding.packed_storage>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8, 1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x1xi32>> -> tensor<8x1xi32>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [3], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3xi1, #iree_encoding.packed_storage>> -> tensor<3xi1, #iree_encoding.packed_storage>
  %6 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false) ins(%3, %4 : tensor<8xi1, #iree_encoding.packed_storage>, tensor<8x1xi32>) outs(%5 : tensor<3xi1, #iree_encoding.packed_storage>) {
  ^bb0(%arg0: i1, %arg1: i1):
    %10 = arith.minui %arg1, %arg0 : i1
    iree_linalg_ext.yield %10 : i1
  } -> tensor<3xi1, #iree_encoding.packed_storage>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0], sizes = [3], strides = [1] : tensor<3xi1, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3xi1, #iree_encoding.packed_storage>>
  return
}

// CHECK-LABEL: @scatter
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[UPDATES:.+]] = hal.interface.binding.subspan {{.*}} binding(0)
// CHECK-DAG:   %[[INDICES:.+]] = hal.interface.binding.subspan {{.*}} binding(1)
// CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan {{.*}} binding(2)
// CHECK-DAG:   %[[UPDATES_TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[UPDATES]]
// CHECK-DAG:   %[[INDICES_TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[INDICES]]
// CHECK-DAG:   %[[OUT_TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUT]]
// CHECK:       %[[SCATTER:.+]] = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false) ins(%[[UPDATES_TENSOR]], %[[INDICES_TENSOR]] : tensor<8xi1>, tensor<8x1xi32>) outs(%[[OUT_TENSOR]] : tensor<3xi1>)
// CHECK:       ^bb0(%[[ARG0:.*]]: i1, %[[ARG1:.*]]: i1):
// CHECK:         %[[MIN:.+]] = arith.minui %[[ARG1]], %[[ARG0]] : i1
// CHECK:         iree_linalg_ext.yield %[[MIN]] : i1
// CHECK:       iree_tensor_ext.dispatch.tensor.store %[[SCATTER]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @branch_op() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i1, #iree_encoding.packed_storage>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i1, #iree_encoding.packed_storage>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i1, #iree_encoding.packed_storage>>
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i8
  %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i1, #iree_encoding.packed_storage>> -> tensor<i1, #iree_encoding.packed_storage>
  %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i1, #iree_encoding.packed_storage>> -> tensor<i1, #iree_encoding.packed_storage>
  %6 = arith.trunci %3 : i8 to i1
  cf.cond_br %6, ^bb1(%4 : tensor<i1, #iree_encoding.packed_storage>), ^bb1(%5 : tensor<i1, #iree_encoding.packed_storage>)
^bb1(%arg1 : tensor<i1, #iree_encoding.packed_storage>):
  iree_tensor_ext.dispatch.tensor.store %arg1, %2, offsets = [], sizes = [], strides = [] : tensor<i1, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i1, #iree_encoding.packed_storage>>
  return
}
// CHECK-LABEL: @branch_op
// CHECK-DAG:   %[[IN0:.+]] = hal.interface.binding.subspan {{.*}} binding(0)
// CHECK-DAG:   %[[IN1:.+]] = hal.interface.binding.subspan {{.*}} binding(1)
// CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan {{.*}} binding(1)
// CHECK-DAG:   %[[COND:.+]] = hal.interface.constant.load
// CHECK-DAG:   %[[TENSOR0:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN0]]
// CHECK-DAG:   %[[TENSOR1:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN1]]
// CHECK:       %[[TRUNC:.+]] = arith.trunci %[[COND]] : i8 to i1
// CHECK:       cf.cond_br %[[TRUNC]], ^bb1(%[[TENSOR0]] : tensor<i1>), ^bb1(%[[TENSOR1]] : tensor<i1>)
// CHECK:     ^bb1(%[[ARG1:.*]]: tensor<i1>):
// CHECK:       iree_tensor_ext.dispatch.tensor.store %[[ARG1]], %[[OUT]]
