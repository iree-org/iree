// RUN: iree-opt -iree-codegen-type-propagation -split-input-file %s | FileCheck %s

func.func @generic_op_illegal_operand() {
  %d = hal.interface.constant.load[0] : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?xi8>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?xi8>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:?xi8>{%d} -> tensor<?xi8>
  %3 = arith.trunci %2 : tensor<?xi8> to tensor<?xi1>
  %4 = linalg.init_tensor [%d] : tensor<?xi8>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%3 : tensor<?xi1>) outs(%4 : tensor<?xi8>) {
      ^bb0(%arg0 : i1, %arg1 : i8):
        %6 = arith.extui %arg0 : i1 to i8
        linalg.yield %6 : i8
    } -> tensor<?xi8>
  flow.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi8> -> !flow.dispatch.tensor<writeonly:?xi8>{%d}
  return
}
// CHECK-LABEL: func @generic_op_illegal_operand()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = flow.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[INIT:.+]] = linalg.init_tensor [%{{.+}}] : tensor<?xi8>
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?xi8>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?xi8>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:?xi8>{%d} -> tensor<?xi8>
  %3 = arith.trunci %2 : tensor<?xi8> to tensor<?xi7>
  %4 = linalg.init_tensor [%d] : tensor<?xi8>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%3 : tensor<?xi7>) outs(%4 : tensor<?xi8>) {
      ^bb0(%arg0 : i7, %arg1 : i8):
        %6 = arith.extui %arg0 : i7 to i8
        linalg.yield %6 : i8
    } -> tensor<?xi8>
  flow.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi8> -> !flow.dispatch.tensor<writeonly:?xi8>{%d}
  return
}
// CHECK-LABEL: func @generic_op_illegal_operand_i7()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = flow.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[INIT:.+]] = linalg.init_tensor [%{{.+}}] : tensor<?xi8>
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?xi64>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?xi64>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:?xi64>{%d} -> tensor<?xi64>
  %3 = arith.trunci %2 : tensor<?xi64> to tensor<?xi33>
  %4 = linalg.init_tensor [%d] : tensor<?xi64>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%3 : tensor<?xi33>) outs(%4 : tensor<?xi64>) {
      ^bb0(%arg0 : i33, %arg1 : i64):
        %6 = arith.extui %arg0 : i33 to i64
        linalg.yield %6 : i64
    } -> tensor<?xi64>
  flow.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi64> -> !flow.dispatch.tensor<writeonly:?xi64>{%d}
  return
}
// CHECK-LABEL: func @generic_op_illegal_operand_i33()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = flow.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[INIT:.+]] = linalg.init_tensor [%{{.+}}] : tensor<?xi64>
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?xi8>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?xi8>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:?xi8>{%d} -> tensor<?xi8>
  %3 = linalg.init_tensor [%d] : tensor<?xi1>
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%2 : tensor<?xi8>) outs(%3 : tensor<?xi1>) {
      ^bb0(%arg0 : i8, %arg1 : i1):
        %5 = arith.trunci %arg0 : i8 to i1
        linalg.yield %5 : i1
    } -> tensor<?xi1>
  %5 = arith.extui %4 : tensor<?xi1> to tensor<?xi8>
  flow.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi8> -> !flow.dispatch.tensor<writeonly:?xi8>{%d}
  return
}
// CHECK-LABEL: func @generic_op_illegal_result()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = flow.dispatch.tensor.load %[[IN]]
//   CHECK-DAG:   %[[INIT:.+]] = linalg.init_tensor [%{{.+}}] : tensor<?xi8>
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?xi8>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?xi8>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:?xi8>{%d} -> tensor<?xi8>
  %3 = tensor.extract_slice %2[%offset] [%size] [1] : tensor<?xi8> to tensor<?xi8>
  %4 = arith.trunci %3 : tensor<?xi8> to tensor<?xi1>
  %5 = arith.extui %4 : tensor<?xi1> to tensor<?xi8>
  flow.dispatch.tensor.store %5, %1, offsets = [%offset], sizes=[%size], strides=[1] : tensor<?xi8> -> !flow.dispatch.tensor<writeonly:?xi8>{%d}
  return   
}
// CHECK-LABEL: func @tensor_extract()
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?xi8>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:?xi8>{%d}
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?xi8>{%d}
  %3 = flow.dispatch.tensor.load %0, offsets = [%offset], sizes=[%size], strides=[1] : !flow.dispatch.tensor<readonly:?xi8>{%d} -> tensor<?xi8>
  %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:?xi8>{%d} -> tensor<?xi8>
  %5 = arith.trunci %3 : tensor<?xi8> to tensor<?xi1>
  %6 = arith.trunci %4 : tensor<?xi8> to tensor<?xi1>
  %7 = tensor.insert_slice %5 into %6[%offset] [%size] [1] : tensor<?xi1> into tensor<?xi1>
  %8 = arith.extui %7 : tensor<?xi1> to tensor<?xi8>
  flow.dispatch.tensor.store %8, %2, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi8> -> !flow.dispatch.tensor<writeonly:?xi8>{%d}
  return
}
// CHECK-LABEL: func @tensor_insert()
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:?xi8>{%d}
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?xi8>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets=[0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:?xi8>{%d} -> tensor<?xi8>
  %3 = flow.dispatch.tensor.load %1, offsets=[0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<writeonly:?xi8>{%d} -> tensor<?xi8>
  %4 = arith.trunci %2 : tensor<?xi8> to tensor<?xi1>
  %5 = arith.trunci %3 : tensor<?xi8> to tensor<?xi1>
  %c0 = arith.constant 0 : index
  %6 = scf.for %arg0 = %c0 to %d step %step iter_args(%arg1 = %5) -> tensor<?xi1> {
    %7 = tensor.extract_slice %4[%arg0][%step][1] : tensor<?xi1> to tensor<?xi1>
    %8 = tensor.insert_slice %7 into %arg1[%arg0][%step][1] : tensor<?xi1> into tensor<?xi1>
    scf.yield %8 : tensor<?xi1>
  }
  %8 = arith.extui %6 : tensor<?xi1> to tensor<?xi8>
  flow.dispatch.tensor.store %8, %1, offsets=[0], sizes=[%d], strides=[1]: tensor<?xi8> -> !flow.dispatch.tensor<writeonly:?xi8>{%d}
  return
}
// CHECK-LABEL: func @for_loop()
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
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<writeonly:?xi8>{%d}
  %1 = linalg.init_tensor [%d] : tensor<?xi1>
  %false = arith.constant false
  %2 = linalg.fill ins(%false : i1) outs(%1 : tensor<?xi1>) -> tensor<?xi1>
  %3 = arith.extui %2 : tensor<?xi1> to tensor<?xi8>
  flow.dispatch.tensor.store %3, %0, offsets=[0], sizes=[%d], strides=[1] : tensor<?xi8> -> !flow.dispatch.tensor<writeonly:?xi8>{%d}
  return
}
// CHECK-LABEL: func @fill_op()
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[INIT:.+]] = linalg.init_tensor
//   CHECK-DAG:   %[[FALSE:.+]] = arith.constant false
//   CHECK-DAG:   %[[EXT_SCALAR:.+]] = arith.extui %[[FALSE]]
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       ins(%[[EXT_SCALAR]] :
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   flow.dispatch.tensor.store %[[FILL]], %[[OUT]]
