// RUN: iree-opt --split-input-file --verify-diagnostics %s

flow.executable @dispatch_with_result_ex {
  builtin.module {
    // expected-error @+1 {{flow dispatch functions should not have a result}}
    func.func @dispatch_with_result(%arg0: tensor<1xf32>) -> tensor<1xf32> {
      return %arg0 : tensor<1xf32>
    }
  }
  flow.executable.export @dispatch_with_result
}
