// Test for importing functions from this spec to a payload module.
// Tested in `transform_symbol_importing.mlir`
module attributes {transform.with_named_sequence} {
  util.func private @some_external_function(%arg0: tensor<?xf32>) -> tensor<?xf32>

  util.func @some_function(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    util.return %arg0 : tensor<?xf32>
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %new_func = transform.util.import_symbol @some_function into %module : (!transform.any_op) -> !transform.any_op

    %func = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op
    %module_2 = transform.util.get_nearest_symbol_table %func : (!transform.any_op) -> !transform.any_op
    %new_func_2 = transform.util.import_symbol @some_external_function into %module_2 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
