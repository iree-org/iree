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
//   bool doRelu;
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
  %empty = pdl.operand
  %lhs_type = pdl.type : tensor<?x?xf32>
  %rhs_type = pdl.type : tensor<?x?xf32>
  %matmul_type = pdl.type : tensor<?x?xf32>

  %zero_val_f32 = pdl.attribute = 0.000000e+00 : f32
  %index_type = pdl.type : index
  %f32_type = pdl.type : f32


  %zero_f32_op = pdl.operation "arith.constant" {"value" = %zero_val_f32} -> (%f32_type : !pdl.type)
  %zero_f32 = pdl.result 0 of %zero_f32_op


  %fill_op = pdl.operation "linalg.fill" (%zero_f32, %empty : !pdl.value, !pdl.value) -> (%matmul_type : !pdl.type)
  %fill = pdl.result 0 of %fill_op
  %matmul = pdl.operation "linalg.matmul" (%lhs, %rhs, %fill : !pdl.value, !pdl.value, !pdl.value) -> (%matmul_type : !pdl.type)

  pdl.rewrite %matmul {
    // The pattern above matched `%result`, `%lhs`, `%rhs` needed for the
    // external function call. The values of `%M`, `%N` and `%K` need to
    // be generated.
    %i32_type = pdl.type : i32
    %bool_type = pdl.type : i1
    %zero_val = pdl.attribute = 0 : index
    %one_val = pdl.attribute = 1 : index
    %zero_op = pdl.operation "arith.constant" {"value" = %zero_val} -> (%index_type : !pdl.type)
    %zero = pdl.result 0 of %zero_op
    %one_op = pdl.operation "arith.constant" {"value" = %one_val} -> (%index_type : !pdl.type)
    %one = pdl.result 0 of %one_op
    %m_op = pdl.operation "tensor.dim"(%lhs, %zero : !pdl.value, !pdl.value) -> (%index_type : !pdl.type)
    %m = pdl.result 0 of %m_op
    %n_op = pdl.operation "tensor.dim"(%rhs, %one : !pdl.value, !pdl.value) -> (%index_type : !pdl.type)
    %n = pdl.result 0 of %n_op
    %k_op = pdl.operation "tensor.dim"(%lhs, %one : !pdl.value, !pdl.value)
    %k = pdl.result 0 of %k_op
    %m_i32_op = pdl.operation "arith.index_cast"(%m : !pdl.value) -> (%i32_type : !pdl.type)
    %m_i32 = pdl.result 0 of %m_i32_op
    %n_i32_op = pdl.operation "arith.index_cast"(%n : !pdl.value) -> (%i32_type : !pdl.type)
    %n_i32 = pdl.result 0 of %n_i32_op
    %k_i32_op = pdl.operation "arith.index_cast"(%k : !pdl.value) -> (%i32_type : !pdl.type)
    %k_i32 = pdl.result 0 of %k_i32_op

    %false_val = pdl.attribute = 0 : i1
    %do_relu_op = pdl.operation "arith.constant" {"value" = %false_val} -> (%bool_type : !pdl.type)
    %do_relu = pdl.result 0 of %do_relu_op

    %replaced_values_dims = pdl.range %m, %n : !pdl.value, !pdl.value
    %input_values = pdl.range %lhs, %rhs : !pdl.value, !pdl.value
    %replaced_value = pdl.result 0 of %matmul
    %replaced_values = pdl.range %replaced_value : !pdl.value
    %other_operands = pdl.range %m_i32, %n_i32, %k_i32, %do_relu : !pdl.value, !pdl.value, !pdl.value, !pdl.value

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
        %matmul, %fn_name, %input_values, %replaced_values, %replaced_values_dims, %other_operands
        : !pdl.operation, !pdl.attribute, !pdl.range<value>, !pdl.range<value>, !pdl.range<value>, !pdl.range<value>)
  }
}
