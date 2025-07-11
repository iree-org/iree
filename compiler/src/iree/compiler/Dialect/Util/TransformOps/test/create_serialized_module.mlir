// RUN: iree-opt --transform-interpreter %s --split-input-file | FileCheck %s

module @my_module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %syms = transform.util.create_serialized_module {
      ^bb0(%m: !transform.any_op):
        transform.annotate %m "util.hello_world" : !transform.any_op
    } -> !transform.any_param
    %container = transform.util.get_nearest_symbol_table %arg1 : (!transform.any_op) -> !transform.any_op
    transform.util.deserialize_module %syms into %container : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: module @my_module
//       CHECK:   transform.named_sequence @__transform_main
//       CHECK:   module attributes {util.hello_world}

// -----

module @my_module attributes { transform.with_named_sequence } {
  util.func private @some_func()
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %syms = transform.util.create_serialized_module {
      ^bb0(%m: !transform.any_op):
        transform.util.import_symbol @some_func into %m if undefined : (!transform.any_op) -> !transform.any_op
        transform.annotate %m "util.has_some_func" : !transform.any_op
    } -> !transform.any_param
    %container = transform.util.get_nearest_symbol_table %arg1 : (!transform.any_op) -> !transform.any_op
    transform.util.deserialize_module %syms into %container : !transform.any_param, !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: module @my_module
//       CHECK:   transform.named_sequence @__transform_main
//       CHECK:   module attributes {util.has_some_func}
//  CHECK-NEXT:     util.func private @some_func
