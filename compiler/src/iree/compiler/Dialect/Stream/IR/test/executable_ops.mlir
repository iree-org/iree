// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: stream.executable private @executable
stream.executable private @executable {
  // CHECK-NEXT: stream.executable.export public @dispatch
  stream.executable.export public @dispatch
  // CHECK-NEXT: builtin.module
  builtin.module  {
    // CHECK-NEXT: func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: index) {
    func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: index) {
      %c0 = arith.constant 0 : index
      // CHECK-DAG: = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readwrite:?x5x64xf32>{%arg2}
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readwrite:?x5x64xf32>{%arg2}
      // CHECK-DAG: = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:?x5x4xf32>{%arg2}
      %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:?x5x4xf32>{%arg2}
      // CHECK: return
      return
    }
  }
}
