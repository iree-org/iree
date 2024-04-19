// RUN: not iree-compile --compile-to=input --split-input-file %s

vhlo.func_v1 @vhlo_stablehlo_func(%arg0: !vhlo.tensor_v1<!vhlo.i32_v1>) -> (!vhlo.tensor_v1<!vhlo.i32_v1>) {
  // expected-error @+1 {{failed to materialize conversion for result #0 of operation 'vhlo.constant_v1' that remained live after conversion}}
  %0 = "vhlo.constant_v1"() <{value = #vhlo.tensor_v1<dense<1> : tensor<i32>>}> : () -> !vhlo.tensor_v1<!vhlo.i32_v1>
  // expected-note @+1 {{see existing live user here: %2 = "builtin.unrealized_conversion_cast"(%1) : (!vhlo.tensor_v1<!vhlo.i32_v1>) -> tensor<i32>}}
  %1 = builtin.unrealized_conversion_cast %0 : !vhlo.tensor_v1<!vhlo.i32_v1> to tensor<i32>
  %2 = stablehlo.abs %1 : tensor<i32>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<i32> to !vhlo.tensor_v1<!vhlo.i32_v1>
  "vhlo.return_v1"(%3) : (!vhlo.tensor_v1<!vhlo.i32_v1>) -> ()
} {arg_attrs = #vhlo.array_v1<[]>, res_attrs = #vhlo.array_v1<[]>, sym_visibility = #vhlo.string_v1<"">}
