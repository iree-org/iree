// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-aggregated-ops{filter-ops='iree_linalg_ext.nonsense_op'}), canonicalize, cse)" --verify-diagnostics %s

// expected-error@+1 {{operation 'iree_linalg_ext.nonsense_op' does not exist}}
func.func @attention(%0: tensor<f32>
) -> tensor<f32>{
    return %0 : tensor<f32>
}
