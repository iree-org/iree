// RUN: iree-opt --split-input-file --iree-flow-dump-executable-functions %s | FileCheck %s

// Test by dumping to stdout.

// CHECK-LABEL: flow.executable public @single_executable_ex_0
flow.executable public @single_executable_ex_0 {
  flow.executable.export @forward_dispatch_0
  builtin.module {
    func.func @forward_dispatch_0(%arg0: !flow.dispatch.tensor<readonly:tensor<1048576xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<1048576xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<1048576xf32>>) {
      %0 = flow.dispatch.tensor.load %arg0, offsets = [0], sizes = [1048576], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1048576xf32>> -> tensor<1048576xf32>
      %1 = flow.dispatch.tensor.load %arg1, offsets = [0], sizes = [1048576], strides = [1] : !flow.dispatch.tensor<readonly:tensor<1048576xf32>> -> tensor<1048576xf32>
      %2 = tensor.empty() : tensor<1048576xf32>
      %3 = linalg.add ins(%0, %1 : tensor<1048576xf32>, tensor<1048576xf32>) outs(%2 : tensor<1048576xf32>) -> tensor<1048576xf32>
      flow.dispatch.tensor.store %3, %arg2, offsets = [0], sizes = [1048576], strides = [1] : tensor<1048576xf32> -> !flow.dispatch.tensor<writeonly:tensor<1048576xf32>>
      return
    }
  }
}

// CHECK-LABEL: util.func public @single_executable_ex_0_forward_dispatch_0
// CHECK-SAME:      %[[ARG0:[A-Za-z0-9]+]]: tensor<1048576xf32>, %[[ARG1:[A-Za-z0-9]+]]: tensor<1048576xf32>
// CHECK-SAME:      -> tensor<1048576xf32> {
// CHECK:         %[[DISPATCH:.+]] = flow.dispatch @single_executable_ex_0::@forward_dispatch_0
// CHECK-SAME:      %[[ARG0]], %[[ARG1]]) : (tensor<1048576xf32>, tensor<1048576xf32>) -> tensor<1048576xf32>
// CHECK:         util.return %[[DISPATCH]] : tensor<1048576xf32>
