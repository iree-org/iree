// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops{filter-ops='tensor.extract_slice'}), canonicalize, cse)" --verify-diagnostics %s

// expected-error@+1 {{operation 'tensor.extract_slice' does not implement AggregatedOpInterface}}
func.func @attention(%0: tensor<f32>
) -> tensor<f32>{
    return %0 : tensor<f32>
}
