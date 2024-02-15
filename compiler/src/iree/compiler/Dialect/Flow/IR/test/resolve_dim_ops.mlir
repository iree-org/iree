// RUN: iree-opt -resolve-ranked-shaped-type-result-dims -split-input-file %s | FileCheck %s

util.func public @tensor_load_op() -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = hal.interface.constant.load[0] : index
  %1 = hal.interface.constant.load[1] : index
  %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
      : !flow.dispatch.tensor<readonly:tensor<?x1x1x?xf32>>{%0, %1}
  %3 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [%0, 1, 1, %1], strides = [1, 1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x1x1x?xf32>>{%0, %1} -> tensor<?x?xf32>
  %4 = tensor.dim %3, %c0 : tensor<?x?xf32>
  %5 = tensor.dim %3, %c1 : tensor<?x?xf32>
  util.return %4, %5 : index, index
}
// CHECK-LABEL: util.func public @tensor_load_op()
//   CHECK-DAG:   %[[D0:.+]] = hal.interface.constant.load[0]
//   CHECK-DAG:   %[[D1:.+]] = hal.interface.constant.load[1]
//       CHECK:   util.return %[[D0]], %[[D1]]
