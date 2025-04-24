// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-type-propagation))" --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @generic_op_illegal_operand() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %3 = arith.trunci %2 : tensor<?xi8> to tensor<?xi1>
  %4 = tensor.empty(%d) : tensor<?xi8>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%3 : tensor<?xi1>) outs(%4 : tensor<?xi8>) {
      ^bb0(%arg0 : i1, %arg1 : i8):
        %6 = arith.extui %arg0 : i1 to i8
        linalg.yield %6 : i8
    } -> tensor<?xi8>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @generic_op_illegal_operand()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?xi8>
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[INTENSOR]] : tensor<?xi8>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?xi8>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i8, %[[ARG1:[a-zA-Z0-9]+]]: i8)
//   CHECK-DAG:       %[[TRUNC:.+]] = arith.trunci %[[ARG0]] : i8 to i1
//   CHECK-DAG:       %[[EXTUI:.+]] = arith.extui %[[TRUNC]] : i1 to i8
//       CHECK:       linalg.yield %[[EXTUI]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[GENERIC]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @generic_op_illegal_operand_i7() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %3 = arith.trunci %2 : tensor<?xi8> to tensor<?xi7>
  %4 = tensor.empty(%d) : tensor<?xi8>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%3 : tensor<?xi7>) outs(%4 : tensor<?xi8>) {
      ^bb0(%arg0 : i7, %arg1 : i8):
        %6 = arith.extui %arg0 : i7 to i8
        linalg.yield %6 : i8
    } -> tensor<?xi8>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @generic_op_illegal_operand_i7()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?xi8>
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[INTENSOR]] : tensor<?xi8>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?xi8>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i8, %[[ARG1:[a-zA-Z0-9]+]]: i8)
//   CHECK-DAG:       %[[TRUNC:.+]] = arith.trunci %[[ARG0]] : i8 to i7
//   CHECK-DAG:       %[[EXTUI:.+]] = arith.extui %[[TRUNC]] : i7 to i8
//       CHECK:       linalg.yield %[[EXTUI]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[GENERIC]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @generic_op_illegal_operand_i33() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi64>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi64>>{%d}
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi64>>{%d} -> tensor<?xi64>
  %3 = arith.trunci %2 : tensor<?xi64> to tensor<?xi33>
  %4 = tensor.empty(%d) : tensor<?xi64>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%3 : tensor<?xi33>) outs(%4 : tensor<?xi64>) {
      ^bb0(%arg0 : i33, %arg1 : i64):
        %6 = arith.extui %arg0 : i33 to i64
        linalg.yield %6 : i64
    } -> tensor<?xi64>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi64> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi64>>{%d}
  return
}
// CHECK-LABEL: func.func @generic_op_illegal_operand_i33()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?xi64>
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[INTENSOR]] : tensor<?xi64>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?xi64>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i64, %[[ARG1:[a-zA-Z0-9]+]]: i64)
//   CHECK-DAG:       %[[TRUNC:.+]] = arith.trunci %[[ARG0]] : i64 to i33
//   CHECK-DAG:       %[[EXTUI:.+]] = arith.extui %[[TRUNC]] : i33 to i64
//       CHECK:       linalg.yield %[[EXTUI]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[GENERIC]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @generic_op_illegal_result() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %3 = tensor.empty(%d) : tensor<?xi1>
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%2 : tensor<?xi8>) outs(%3 : tensor<?xi1>) {
      ^bb0(%arg0 : i8, %arg1 : i1):
        %5 = arith.trunci %arg0 : i8 to i1
        linalg.yield %5 : i1
    } -> tensor<?xi1>
  %5 = arith.extui %4 : tensor<?xi1> to tensor<?xi8>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @generic_op_illegal_result()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?xi8>
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[INTENSOR]] : tensor<?xi8>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?xi8>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i8, %[[ARG1:[a-zA-Z0-9]+]]: i8)
//   CHECK-DAG:       %[[TRUNC:.+]] = arith.trunci %[[ARG0]] : i8 to i1
//   CHECK-DAG:       %[[EXTUI:.+]] = arith.extui %[[TRUNC]] : i1 to i8
//       CHECK:       linalg.yield %[[EXTUI]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[GENERIC]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @tensor_extract() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %offset = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %size = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %3 = tensor.extract_slice %2[%offset] [%size] [1] : tensor<?xi8> to tensor<?xi8>
  %4 = arith.trunci %3 : tensor<?xi8> to tensor<?xi1>
  %5 = arith.extui %4 : tensor<?xi1> to tensor<?xi8>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [%offset], sizes=[%size], strides=[1] : tensor<?xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @tensor_extract()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN]]
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[INTENSOR]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[EXTRACT]], %[[OUT]]

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
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [%offset], sizes=[%size], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %5 = arith.trunci %3 : tensor<?xi8> to tensor<?xi1>
  %6 = arith.trunci %4 : tensor<?xi8> to tensor<?xi1>
  %7 = tensor.insert_slice %5 into %6[%offset] [%size] [1] : tensor<?xi1> into tensor<?xi1>
  %8 = arith.extui %7 : tensor<?xi1> to tensor<?xi8>
  iree_tensor_ext.dispatch.tensor.store %8, %2, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @tensor_insert()
//   CHECK-DAG:   %[[IN1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[IN2:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//   CHECK-DAG:   %[[IN1TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN1]]
//   CHECK-DAG:   %[[IN2TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN2]]
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice %[[IN1TENSOR]] into %[[IN2TENSOR]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[INSERT]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @for_loop() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %lb = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %step = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets=[0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets=[0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %4 = arith.trunci %2 : tensor<?xi8> to tensor<?xi1>
  %5 = arith.trunci %3 : tensor<?xi8> to tensor<?xi1>
  %c0 = arith.constant 0 : index
  %6 = scf.for %arg0 = %c0 to %d step %step iter_args(%arg1 = %5) -> tensor<?xi1> {
    %7 = tensor.extract_slice %4[%arg0][%step][1] : tensor<?xi1> to tensor<?xi1>
    %8 = tensor.insert_slice %7 into %arg1[%arg0][%step][1] : tensor<?xi1> into tensor<?xi1>
    scf.yield %8 : tensor<?xi1>
  }
  %8 = arith.extui %6 : tensor<?xi1> to tensor<?xi8>
  iree_tensor_ext.dispatch.tensor.store %8, %1, offsets=[0], sizes=[%d], strides=[1]: tensor<?xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @for_loop()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[OUTTENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUT]]
//       CHECK:   %[[FOR:.+]] = scf.for
//  CHECK-SAME:       iter_args(%[[ARG1:.+]] = %[[OUTTENSOR]])
//       CHECK:     %[[SLICE:.+]] = tensor.extract_slice %[[INTENSOR]]
//       CHECK:     %[[INSERT:.+]] = tensor.insert_slice %[[SLICE]] into %[[ARG1]]
//       CHECK:     scf.yield %[[INSERT]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[FOR]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @fill_op() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %1 = tensor.empty(%d) : tensor<?xi1>
  %false = arith.constant false
  %2 = linalg.fill ins(%false : i1) outs(%1 : tensor<?xi1>) -> tensor<?xi1>
  %3 = arith.extui %2 : tensor<?xi1> to tensor<?xi8>
  iree_tensor_ext.dispatch.tensor.store %3, %0, offsets=[0], sizes=[%d], strides=[1] : tensor<?xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @fill_op()
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty
//   CHECK-DAG:   %[[FALSE:.+]] = arith.constant false
//   CHECK-DAG:   %[[EXT_SCALAR:.+]] = arith.extui %[[FALSE]]
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       ins(%[[EXT_SCALAR]] :
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[FILL]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0) -> (d0)>
func.func @constant_op() {
  %a = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi32>>
  %b = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi32>>
  %c = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi32>>
  %at = iree_tensor_ext.dispatch.tensor.load %a, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi32>> -> tensor<4xi32>
  %bt = iree_tensor_ext.dispatch.tensor.load %b, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi32>> -> tensor<4xi32>
  %select = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
  %init = tensor.empty() : tensor<4xi32>
  %result = linalg.generic {
      indexing_maps = [#map, #map, #map, #map],
      iterator_types = ["parallel"]}
      ins(%select, %at, %bt : tensor<4xi1>, tensor<4xi32>, tensor<4xi32>)
      outs(%init : tensor<4xi32>) {
    ^bb0(%b0 : i1, %b1 : i32, %b2 : i32, %b3 : i32) :
      %0 = arith.select %b0, %b1, %b2 : i32
      linalg.yield %0 : i32
  } -> tensor<4xi32>
  iree_tensor_ext.dispatch.tensor.store %result, %c, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi32>>
  return
}
// CHECK-LABEL: func.func @constant_op()
//       CHECK:   %[[CONST:.+]] = arith.constant dense<[1, 0, 1, 0]> : tensor<4xi8>
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%[[CONST]]
//  CHECK-NEXT:   ^bb0
//  CHECK-SAME:       %[[B0:.+]]: i8
//       CHECK:     %[[TRUNC:.+]] = arith.trunci %[[B0]] : i8 to i1
//       CHECK:     arith.select %[[TRUNC]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0) -> (d0)>
func.func @constant_splat_op() {
  %a = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi32>>
  %b = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi32>>
  %c = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi32>>
  %at = iree_tensor_ext.dispatch.tensor.load %a, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi32>> -> tensor<4xi32>
  %bt = iree_tensor_ext.dispatch.tensor.load %b, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi32>> -> tensor<4xi32>
  %select = arith.constant dense<true> : tensor<4xi1>
  %init = tensor.empty() : tensor<4xi32>
  %result = linalg.generic {
      indexing_maps = [#map, #map, #map, #map],
      iterator_types = ["parallel"]}
      ins(%select, %at, %bt : tensor<4xi1>, tensor<4xi32>, tensor<4xi32>)
      outs(%init : tensor<4xi32>) {
    ^bb0(%b0 : i1, %b1 : i32, %b2 : i32, %b3 : i32) :
      %0 = arith.select %b0, %b1, %b2 : i32
      linalg.yield %0 : i32
  } -> tensor<4xi32>
  iree_tensor_ext.dispatch.tensor.store %result, %c, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi32>>
  return
}
// CHECK-LABEL: func.func @constant_splat_op()
//       CHECK:   arith.constant dense<1> : tensor<4xi8>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @tensor_extract() {
  %c0 = arith.constant 0 : index
  %c13 = arith.constant 13 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<14xi8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14xi8>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [14], strides = [1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<14xi8>> -> tensor<14xi8>
  %3 = arith.trunci %2 : tensor<14xi8> to tensor<14xi1>
  %4 = tensor.empty() : tensor<14xi1>
  %5 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      outs(%4 : tensor<14xi1>) {
  ^bb0(%out: i1):
    %7 = linalg.index 0 : index
    %8 = arith.subi %c13, %7 : index
    %extracted = tensor.extract %3[%8] : tensor<14xi1>
    linalg.yield %extracted : i1
  } -> tensor<14xi1>
  %6 = arith.extui %5 : tensor<14xi1> to tensor<14xi8>
  iree_tensor_ext.dispatch.tensor.store %6, %1, offsets = [0], sizes = [14], strides = [1]
      : tensor<14xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14xi8>>
  return
}
// CHECK-LABEL: func @tensor_extract()
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<14xi8>>
//       CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[BINDING]]
//       CHECK:   %[[EXTRACTED:.+]] = tensor.extract %[[LOAD]]
//       CHECK:   arith.trunci %[[EXTRACTED]] : i8 to i1

// -----

func.func @named_op(%arg0 : tensor<?x?xi8>, %arg1 : tensor<?x?xi8>) -> tensor<?x?xi8> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %false = arith.constant false
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xi8>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xi8>
  %empty = tensor.empty(%d0, %d1) : tensor<?x?xi1>
  %fill = linalg.fill ins(%false : i1) outs(%empty : tensor<?x?xi1>) -> tensor<?x?xi1>
  %arg0_i1 = arith.trunci %arg0 : tensor<?x?xi8> to tensor<?x?xi1>
  %arg1_i1 = arith.trunci %arg1 : tensor<?x?xi8> to tensor<?x?xi1>
  %gemm = linalg.matmul ins(%arg0_i1, %arg1_i1 : tensor<?x?xi1>, tensor<?x?xi1>)
      outs(%fill : tensor<?x?xi1>) -> tensor<?x?xi1>
  %result_i8 = arith.extui %gemm : tensor<?x?xi1> to tensor<?x?xi8>
  return %result_i8 : tensor<?x?xi8>
}
//      CHECK: func @named_op(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xi8>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xi8>
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<?x?xi8>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   %[[GEMM:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[FILL]] :
//      CHECK:   return %[[GEMM]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @scatter() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8xi8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x1xi32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3xi8>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [8], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8xi8>> -> tensor<8xi8>
  %4 = arith.trunci %3 : tensor<8xi8> to tensor<8xi1>
  %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [8, 1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x1xi32>> -> tensor<8x1xi32>
  %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [3], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3xi8>> -> tensor<3xi8>
  %7 = arith.trunci %6 : tensor<3xi8> to tensor<3xi1>
  %8 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false) ins(%4, %5 : tensor<8xi1>, tensor<8x1xi32>) outs(%7 : tensor<3xi1>) {
  ^bb0(%arg0: i1, %arg1: i1):
    %10 = arith.minui %arg1, %arg0 : i1
    iree_linalg_ext.yield %10 : i1
  } -> tensor<3xi1>
  %9 = arith.extui %8 : tensor<3xi1> to tensor<3xi8>
  iree_tensor_ext.dispatch.tensor.store %9, %2, offsets = [0], sizes = [3], strides = [1] : tensor<3xi8> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3xi8>>
  return
}

// CHECK-LABEL: func.func @scatter()
//   CHECK-DAG:   %[[UPDATES:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[INDICES:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
//   CHECK-DAG:   %[[UPDATES_TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[UPDATES]]
//   CHECK-DAG:   %[[INDICES_TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[INDICES]]
//   CHECK-DAG:   %[[OUT_TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[OUT]]
//       CHECK:   %[[SCATTER:.+]] = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false)
//  CHECK-SAME:       ins(%[[UPDATES_TENSOR]], %[[INDICES_TENSOR]] : tensor<8xi8>, tensor<8x1xi32>)
//  CHECK-SAME:       outs(%[[OUT_TENSOR]] : tensor<3xi8>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i8, %[[ARG1:[a-zA-Z0-9]+]]: i8)
//   CHECK-DAG:       %[[TRUNC0:.+]] = arith.trunci %[[ARG0]] : i8 to i1
//   CHECK-DAG:       %[[TRUNC1:.+]] = arith.trunci %[[ARG1]] : i8 to i1
//   CHECK-DAG:       %[[MIN:.+]] = arith.minui %[[TRUNC1]], %[[TRUNC0]] : i1
//   CHECK-DAG:       %[[EXTUI:.+]] = arith.extui %[[MIN]] : i1 to i8
//       CHECK:       iree_linalg_ext.yield %[[EXTUI]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[SCATTER]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @sort() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1xi8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1xi32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [1], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1xi8>> -> tensor<1xi8>
  %3 = arith.trunci %2 : tensor<1xi8> to tensor<1xi1>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [1], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1xi32>> -> tensor<1xi32>
  %5:2 = iree_linalg_ext.sort dimension(0) outs(%3, %4 : tensor<1xi1>, tensor<1xi32>) {
  ^bb0(%arg0: i1, %arg1: i1, %arg2: i32, %arg3: i32):
    %6 = arith.cmpi ult, %arg0, %arg1 : i1
    iree_linalg_ext.yield %6 : i1
  } -> tensor<1xi1>, tensor<1xi32>
  iree_tensor_ext.dispatch.tensor.store %5#1, %1, offsets = [0], sizes = [1], strides = [1] : tensor<1xi32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1xi32>>
  return
}

// CHECK-LABEL: func.func @sort()
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[A:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[B:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[A_TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[A]]
//   CHECK-DAG:   %[[B_TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[B]]
//       CHECK:   %[[SORT:.+]]:2 = iree_linalg_ext.sort dimension(0)
//  CHECK-SAME:       outs(%[[A_TENSOR]], %[[B_TENSOR]] : tensor<1xi8>, tensor<1xi32>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i8, %[[ARG1:[a-zA-Z0-9]+]]: i8, %[[ARG2:[a-zA-Z0-9]+]]: i32, %[[ARG3:[a-zA-Z0-9]+]]: i32)
//   CHECK-DAG:       %[[TRUNC_A_1:.+]] = arith.trunci %[[ARG0]] : i8 to i1
//   CHECK-DAG:       %[[TRUNC_A_2:.+]] = arith.trunci %[[ARG1]] : i8 to i1
//   CHECK-DAG:       %[[CMPI:.+]] = arith.cmpi ult, %[[TRUNC_A_1]], %[[TRUNC_A_2]] : i1
//       CHECK:       iree_linalg_ext.yield %[[CMPI]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[SORT]]#1, %[[B]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @sort_secondary() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1xi32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1xi8>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [1], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1xi32>> -> tensor<1xi32>
  %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [1], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1xi8>> -> tensor<1xi8>
  %4 = arith.trunci %3 : tensor<1xi8> to tensor<1xi1>
  %5:2 = iree_linalg_ext.sort dimension(0) outs(%2, %4 : tensor<1xi32>, tensor<1xi1>) {
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i1, %arg3: i1):
    %6 = arith.cmpi ult, %arg0, %arg1 : i32
    iree_linalg_ext.yield %6 : i1
  } -> tensor<1xi32>, tensor<1xi1>
  %7 = arith.extui %5#1 : tensor<1xi1> to tensor<1xi8>
  iree_tensor_ext.dispatch.tensor.store %7, %1, offsets = [0], sizes = [1], strides = [1] : tensor<1xi8> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1xi8>>
  return
}

// CHECK-LABEL: func.func @sort_secondary()
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[A:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[B:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[A_TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[A]]
//   CHECK-DAG:   %[[B_TENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[B]]
//       CHECK:   %[[SORT:.+]]:2 = iree_linalg_ext.sort dimension(0)
//  CHECK-SAME:       outs(%[[A_TENSOR]], %[[B_TENSOR]] : tensor<1xi32>, tensor<1xi8>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i32, %[[ARG1:[a-zA-Z0-9]+]]: i32, %[[ARG2:[a-zA-Z0-9]+]]: i8, %[[ARG3:[a-zA-Z0-9]+]]: i8)
//   CHECK-DAG:       %[[CMPI:.+]] = arith.cmpi ult, %[[ARG0]], %[[ARG1]] : i32
//       CHECK:       iree_linalg_ext.yield %[[CMPI]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[SORT]]#1, %[[B]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @branch_op() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i8>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i8>>
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i8
  %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i8>> -> tensor<i8>
  %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i8>> -> tensor<i8>
  %6 = arith.trunci %3 : i8 to i1
  %7 = arith.trunci %4 : tensor<i8> to tensor<i1>
  %8 = arith.trunci %5 : tensor<i8> to tensor<i1>
  cf.cond_br %6, ^bb1(%7 : tensor<i1>), ^bb1(%8 : tensor<i1>)
^bb1(%arg1 : tensor<i1>):
  %9 = arith.extui %arg1 : tensor<i1> to tensor<i8>
  iree_tensor_ext.dispatch.tensor.store %9, %2, offsets = [], sizes = [], strides = [] : tensor<i8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i8>>
  return
}
// CHECK-LABEL: func @branch_op()
//       CHECK:   cf.cond_br %{{.+}}, ^bb1(%{{.+}} : tensor<i8>), ^bb1(%{{.+}} : tensor<i8>)
//       CHECK: ^bb1(%{{.+}}: tensor<i8>)
