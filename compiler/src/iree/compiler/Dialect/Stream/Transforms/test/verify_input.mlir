// RUN: iree-opt --iree-stream-verify-input --split-input-file %s --verify-diagnostics

// expected-error @+2 {{cannot be converted to stream because it contains public functions with a result}}
// expected-error @+1 {{illegal for this phase of lowering in the stream dialect}}
flow.executable @dispatch_with_result_ex {
  builtin.module {
    func.func @dispatch_with_result(%arg0: tensor<1xf32>) -> tensor<1xf32> {
      return %arg0 : tensor<1xf32>
    }
  }
  flow.executable.export @dispatch_with_result
}
