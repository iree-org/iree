// RUN: iree-opt --iree-codegen-type-propagation --split-input-file %s | FileCheck %s

func.func @generic_op_illegal_operand() {
  %d = hal.interface.constant.load[0] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
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
  flow.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi8> -> !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @generic_op_illegal_operand()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = flow.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?xi8>
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[INTENSOR]] : tensor<?xi8>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?xi8>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i8, %[[ARG1:[a-zA-Z0-9]+]]: i8)
//   CHECK-DAG:       %[[TRUNC:.+]] = arith.trunci %[[ARG0]] : i8 to i1
//   CHECK-DAG:       %[[EXTUI:.+]] = arith.extui %[[TRUNC]] : i1 to i8
//       CHECK:       linalg.yield %[[EXTUI]]
//       CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[OUT]]

// -----

func.func @generic_op_illegal_operand_i7() {
  %d = hal.interface.constant.load[0] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
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
  flow.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi8> -> !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @generic_op_illegal_operand_i7()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = flow.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?xi8>
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[INTENSOR]] : tensor<?xi8>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?xi8>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i8, %[[ARG1:[a-zA-Z0-9]+]]: i8)
//   CHECK-DAG:       %[[TRUNC:.+]] = arith.trunci %[[ARG0]] : i8 to i7
//   CHECK-DAG:       %[[EXTUI:.+]] = arith.extui %[[TRUNC]] : i7 to i8
//       CHECK:       linalg.yield %[[EXTUI]]
//       CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[OUT]]

// -----

func.func @generic_op_illegal_operand_i33() {
  %d = hal.interface.constant.load[0] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xi64>>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?xi64>>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:tensor<?xi64>>{%d} -> tensor<?xi64>
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
  flow.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi64> -> !flow.dispatch.tensor<writeonly:tensor<?xi64>>{%d}
  return
}
// CHECK-LABEL: func.func @generic_op_illegal_operand_i33()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = flow.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?xi64>
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[INTENSOR]] : tensor<?xi64>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?xi64>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i64, %[[ARG1:[a-zA-Z0-9]+]]: i64)
//   CHECK-DAG:       %[[TRUNC:.+]] = arith.trunci %[[ARG0]] : i64 to i33
//   CHECK-DAG:       %[[EXTUI:.+]] = arith.extui %[[TRUNC]] : i33 to i64
//       CHECK:       linalg.yield %[[EXTUI]]
//       CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[OUT]]


// -----

func.func @generic_op_illegal_result() {
  %d = hal.interface.constant.load[0] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
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
  flow.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi8> -> !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @generic_op_illegal_result()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = flow.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?xi8>
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[INTENSOR]] : tensor<?xi8>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?xi8>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i8, %[[ARG1:[a-zA-Z0-9]+]]: i8)
//   CHECK-DAG:       %[[TRUNC:.+]] = arith.trunci %[[ARG0]] : i8 to i1
//   CHECK-DAG:       %[[EXTUI:.+]] = arith.extui %[[TRUNC]] : i1 to i8
//       CHECK:       linalg.yield %[[EXTUI]]
//       CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[OUT]]

// -----

func.func @tensor_extract() {
  %d = hal.interface.constant.load[0] : index
  %offset = hal.interface.constant.load[1] : index
  %size = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %3 = tensor.extract_slice %2[%offset] [%size] [1] : tensor<?xi8> to tensor<?xi8>
  %4 = arith.trunci %3 : tensor<?xi8> to tensor<?xi1>
  %5 = arith.extui %4 : tensor<?xi1> to tensor<?xi8>
  flow.dispatch.tensor.store %5, %1, offsets = [%offset], sizes=[%size], strides=[1] : tensor<?xi8> -> !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @tensor_extract()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = flow.dispatch.tensor.load %[[IN]]
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[INTENSOR]]
//       CHECK:   flow.dispatch.tensor.store %[[EXTRACT]], %[[OUT]]

// -----

func.func @tensor_insert() {
  %d = hal.interface.constant.load[0] : index
  %offset = hal.interface.constant.load[1] : index
  %size = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %3 = flow.dispatch.tensor.load %0, offsets = [%offset], sizes=[%size], strides=[1] : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %5 = arith.trunci %3 : tensor<?xi8> to tensor<?xi1>
  %6 = arith.trunci %4 : tensor<?xi8> to tensor<?xi1>
  %7 = tensor.insert_slice %5 into %6[%offset] [%size] [1] : tensor<?xi1> into tensor<?xi1>
  %8 = arith.extui %7 : tensor<?xi1> to tensor<?xi8>
  flow.dispatch.tensor.store %8, %2, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi8> -> !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @tensor_insert()
//   CHECK-DAG:   %[[IN1:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[IN2:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(2)
//   CHECK-DAG:   %[[IN1TENSOR:.+]] = flow.dispatch.tensor.load %[[IN1]]
//   CHECK-DAG:   %[[IN2TENSOR:.+]] = flow.dispatch.tensor.load %[[IN2]]
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice %[[IN1TENSOR]] into %[[IN2TENSOR]]
//       CHECK:   flow.dispatch.tensor.store %[[INSERT]], %[[OUT]]

// -----

func.func @for_loop() {
  %d = hal.interface.constant.load[0] : index
  %lb = hal.interface.constant.load[1] : index
  %step = hal.interface.constant.load[2] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets=[0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %3 = flow.dispatch.tensor.load %1, offsets=[0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d} -> tensor<?xi8>
  %4 = arith.trunci %2 : tensor<?xi8> to tensor<?xi1>
  %5 = arith.trunci %3 : tensor<?xi8> to tensor<?xi1>
  %c0 = arith.constant 0 : index
  %6 = scf.for %arg0 = %c0 to %d step %step iter_args(%arg1 = %5) -> tensor<?xi1> {
    %7 = tensor.extract_slice %4[%arg0][%step][1] : tensor<?xi1> to tensor<?xi1>
    %8 = tensor.insert_slice %7 into %arg1[%arg0][%step][1] : tensor<?xi1> into tensor<?xi1>
    scf.yield %8 : tensor<?xi1>
  }
  %8 = arith.extui %6 : tensor<?xi1> to tensor<?xi8>
  flow.dispatch.tensor.store %8, %1, offsets=[0], sizes=[%d], strides=[1]: tensor<?xi8> -> !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @for_loop()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = flow.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[OUTTENSOR:.+]] = flow.dispatch.tensor.load %[[OUT]]
//       CHECK:   %[[FOR:.+]] = scf.for
//  CHECK-SAME:       iter_args(%[[ARG1:.+]] = %[[OUTTENSOR]])
//       CHECK:     %[[SLICE:.+]] = tensor.extract_slice %[[INTENSOR]]
//       CHECK:     %[[INSERT:.+]] = tensor.insert_slice %[[SLICE]] into %[[ARG1]]
//       CHECK:     scf.yield %[[INSERT]]
//       CHECK:   flow.dispatch.tensor.store %[[FOR]], %[[OUT]]

// -----

func.func @fill_op() {
  %d = hal.interface.constant.load[0] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  %1 = tensor.empty(%d) : tensor<?xi1>
  %false = arith.constant false
  %2 = linalg.fill ins(%false : i1) outs(%1 : tensor<?xi1>) -> tensor<?xi1>
  %3 = arith.extui %2 : tensor<?xi1> to tensor<?xi8>
  flow.dispatch.tensor.store %3, %0, offsets=[0], sizes=[%d], strides=[1] : tensor<?xi8> -> !flow.dispatch.tensor<writeonly:tensor<?xi8>>{%d}
  return
}
// CHECK-LABEL: func.func @fill_op()
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty
//   CHECK-DAG:   %[[FALSE:.+]] = arith.constant false
//   CHECK-DAG:   %[[EXT_SCALAR:.+]] = arith.extui %[[FALSE]]
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       ins(%[[EXT_SCALAR]] :
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   flow.dispatch.tensor.store %[[FILL]], %[[OUT]]

// -----

#map = affine_map<(d0) -> (d0)>
func.func @constant_op() {
  %a = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4xi32>>
  %b = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4xi32>>
  %c = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<4xi32>>
  %at = flow.dispatch.tensor.load %a, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xi32>> -> tensor<4xi32>
  %bt = flow.dispatch.tensor.load %b, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xi32>> -> tensor<4xi32>
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
  flow.dispatch.tensor.store %result, %c, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !flow.dispatch.tensor<writeonly:tensor<4xi32>>
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

#map = affine_map<(d0) -> (d0)>
func.func @constant_splat_op() {
  %a = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4xi32>>
  %b = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4xi32>>
  %c = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<4xi32>>
  %at = flow.dispatch.tensor.load %a, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xi32>> -> tensor<4xi32>
  %bt = flow.dispatch.tensor.load %b, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xi32>> -> tensor<4xi32>
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
  flow.dispatch.tensor.store %result, %c, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !flow.dispatch.tensor<writeonly:tensor<4xi32>>
  return
}
// CHECK-LABEL: func.func @constant_splat_op()
//       CHECK:   arith.constant dense<1> : tensor<4xi8>

// -----

func.func @tensor_extract() {
  %c0 = arith.constant 0 : index
  %c13 = arith.constant 13 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<14xi8>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<writeonly:tensor<14xi8>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [14], strides = [1]
      : !flow.dispatch.tensor<readonly:tensor<14xi8>> -> tensor<14xi8>
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
  flow.dispatch.tensor.store %6, %1, offsets = [0], sizes = [14], strides = [1]
      : tensor<14xi8> -> !flow.dispatch.tensor<writeonly:tensor<14xi8>>
  return
}
// CHECK-LABEL: func @tensor_extract()
//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<14xi8>>
//       CHECK:   %[[LOAD:.+]] = flow.dispatch.tensor.load %[[BINDING]]
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
