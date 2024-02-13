// PDL pattern spec to match an MLP and offload to an external function
//
// ```
// void mlp_external(void *params, void *context, void *reserved)
// ```
//
// which is the expected signature of an external function implemented
// provided by a system plugin. See
// samples/custom_dispatch/cpu/plugin/system_plugin.c for an example.
//
// The `params` is the following struct
//
// ```
// struct mlp_params_t {
//   const float *restrict lhs;
//   size_t lhs_offset;
//   const float *restrict rhs;
//   size_t rhs_offset;
//   int32_t M;
//   int32_t N;
//   int32_t K;
//   float *restrict result;
//   size_t result_offset;
// };
// ```
//
// In MLIR this corresponds to the function
//
// ```
// func.func @mlp_external(%lhs : memref<..xf32>, %rhs : memref<..xf32>,
//     %M: i32, %N : i32, %K : i32, %result : memref<..xf32>)
// ```
//
// Note: In the above struct a `pointer, offset` pair represents a buffer
// passed into the external function. So any access to `lhs`, `rhs` and
// `result` is valid only if accessed as `lhs[lhs_offset + ...]`,
// `rhs[rhs_offset + ]` and `result[result_offset + ...]`.
pdl.pattern @mlp : benefit(1) {

  // PDL matcher to match the MLP computation. This pattern is expected to
  // match
  //
  // ```
  // %result = func.call @mlp_external(%lhs : tensor<...xf32>,
  //     %rhs : tensor<..xf32>, %M : i32, %N : i32, %K : i32) -> tensor<..xf32>
  // ```
  %lhs = pdl.operand
  %rhs = pdl.operand
  %lhs_type = pdl.type
  %rhs_type = pdl.type
  %lhs_torch = pdl.operation "torch_c.from_builtin_tensor"(%lhs : !pdl.value) -> (%lhs_type : !pdl.type)
  %lhs_val = pdl.result 0 of %lhs_torch
  %rhs_torch = pdl.operation "torch_c.from_builtin_tensor"(%rhs : !pdl.value) -> (%rhs_type : !pdl.type)
  %rhs_val = pdl.result 0 of %rhs_torch
  %matmul_type = pdl.type
  %matmul = pdl.operation "torch.aten.mm"(%lhs_val, %rhs_val : !pdl.value, !pdl.value) -> (%matmul_type : !pdl.type)
  %matmul_result = pdl.result 0 of %matmul
  %relu = pdl.operation "torch.aten.relu"(%matmul_result : !pdl.value) -> (%matmul_type : !pdl.type)
  %result_type = pdl.type
  %relu_val = pdl.result 0 of %relu
  %cast = pdl.operation "torch_c.to_builtin_tensor"(%relu_val : !pdl.value) -> (%result_type : !pdl.type)
  
  pdl.rewrite %matmul {
    // The pattern above matched `%result`, `%lhs`, `%rhs` needed for the
    // external function call. The values of `%M`, `%N` and `%K` need to
    // be generated. 
    %zero_val = pdl.attribute = 0 : index
    %one_val = pdl.attribute = 1 : index
    %index_type = pdl.type : index
    %zero_op = pdl.operation "arith.constant" {"value" = %zero_val} -> (%index_type : !pdl.type)
    %zero = pdl.result 0 of %zero_op
    %one_op = pdl.operation "arith.constant" {"value" = %one_val} -> (%index_type : !pdl.type)
    %one = pdl.result 0 of %one_op
    %i32_type = pdl.type : i32
    %m_op = pdl.operation "tensor.dim"(%lhs, %zero : !pdl.value, !pdl.value)
    %m = pdl.result 0 of %m_op
    %n_op = pdl.operation "tensor.dim"(%rhs, %one : !pdl.value, !pdl.value)
    %n = pdl.result 0 of %n_op
    %k_op = pdl.operation "tensor.dim"(%lhs, %one : !pdl.value, !pdl.value)
    %k = pdl.result 0 of %k_op
    %m_i32_op = pdl.operation "arith.index_cast"(%m : !pdl.value) -> (%i32_type : !pdl.type)
    %m_i32 = pdl.result 0 of %m_i32_op
    %n_i32_op = pdl.operation "arith.index_cast"(%n : !pdl.value) -> (%i32_type : !pdl.type)
    %n_i32 = pdl.result 0 of %n_i32_op
    %k_i32_op = pdl.operation "arith.index_cast"(%k : !pdl.value) -> (%i32_type : !pdl.type)
    %k_i32 = pdl.result 0 of %k_i32_op

    %replaced_values_dims = pdl.range %m, %n : !pdl.value, !pdl.value
    %input_values = pdl.range %lhs, %rhs : !pdl.value, !pdl.value
    %replaced_value = pdl.result 0 of %cast
    %replaced_values = pdl.range %replaced_value : !pdl.value
    %other_operands = pdl.range %m_i32, %n_i32, %k_i32 : !pdl.value, !pdl.value, !pdl.value

    // The `rewriteAsFlowDispatch` is a rewrite function that allows
    // converting the matched dag into a call to the external function call
    // provided by a system plugin. The rewrite method expects the following
    // arguments
    // - the root of the matched DAG. This op will be erased after the call.
    // - `fn_name` the name of the function that is provided externally
    //   (using a plugin).
    // - `input_values` are values that are captures as the part of the match
    //   and are inputs to the match.
    // - `replaced_values` are the values that are captured as part of the
    //   match and are replaced by the `flow.dispatch`. The `flow.dispatch`
    //   returns as many values as `replaced_values` (and of same type).
    // - `replaced_values_dims` are the values for the dynamic dimensions of
    //   all the `tensor` values in `replaced_values`. For matches that could
    //   be static or dynamic, it should be assumed that the shape is dynamic
    //   and the value needs to be passed to the rewrite function.
    // - `other_operands` same as `input_values`, but kept separate to allow
    //   flexibility of where the results are passed through the ABI boundary.
    %fn_name = pdl.attribute = "mlp_external"
    pdl.apply_native_rewrite "rewriteAsFlowDispatch"(
        %cast, %fn_name, %input_values, %replaced_values, %replaced_values_dims, %other_operands
        : !pdl.operation, !pdl.attribute, !pdl.range<value>, !pdl.range<value>, !pdl.range<value>, !pdl.range<value>)
  }
}

