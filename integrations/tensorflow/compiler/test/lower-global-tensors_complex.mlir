// RUN: iree-tf-opt -pass-pipeline=iree-tf-saved-model-lower-global-tensors -verify-diagnostics -split-input-file <%s | IreeFileCheck %s

// -----

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {

// CHECK: flow.variable [[V:@[a-z_]+]] mutable dense<1.000000e+00> : tensor<1xf32>
// CHECK: func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}) attributes {tf_saved_model.exported_names = ["f"]} {
// CHECK:   br ^bb1(%arg0 : tensor<?xf32>)
// CHECK: ^bb1([[T:%.+]]: tensor<?xf32>):
// CHECK:   flow.variable.store [[T]], [[V]] : tensor<?xf32>
// CHECK:   return

  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}, %arg1: tensor<*x!tf.resource> {tf_saved_model.bound_input = @v}) attributes {tf_saved_model.exported_names = ["f"]} {
    br ^bb1(%arg0, %arg1 : tensor<?xf32>, tensor<*x!tf.resource>)
  ^bb1(%t: tensor<?xf32>, %r: tensor<*x!tf.resource>):
    "tf.AssignVariableOp"(%r, %t) : (tensor<*x!tf.resource>, tensor<?xf32>) -> ()
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v1", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}, %v: tensor<*x!tf.resource> {tf_saved_model.bound_input = @v}, %v1: tensor<*x!tf.resource> {tf_saved_model.bound_input = @v1}) -> (tensor<?xf32> {tf_saved_model.index_path = [0]}) attributes {tf_saved_model.exported_names = ["f"]} {
    %pred = constant false
    cond_br %pred, ^bb1(%v : tensor<*x!tf.resource>), ^bb1(%v1 : tensor<*x!tf.resource>)
  ^bb1(%either: tensor<*x!tf.resource>):
    // expected-error@+1 {{cannot prove resource op uses a single global tensor: potential global tensors: 'v', 'v1'}}
    %ret = "tf.ReadVariableOp"(%either) : (tensor<*x!tf.resource>) -> tensor<?xf32>
    return %ret : tensor<?xf32>
  }
}

// -----

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {

  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}, %arg1: tensor<*x!tf.resource> {tf_saved_model.bound_input = @v}) attributes {tf_saved_model.exported_names = ["f"]} {
    br ^bb1(%arg1, %arg1, %arg1 : tensor<*x!tf.resource>, tensor<*x!tf.resource>, tensor<*x!tf.resource>)
  ^bb1(%0: tensor<*x!tf.resource>, %1: tensor<*x!tf.resource>, %2: tensor<*x!tf.resource>):
    "tf.AssignVariableOp"(%0, %arg0) : (tensor<*x!tf.resource>, tensor<?xf32>) -> ()
    br ^bb1(%1, %2, %0 : tensor<*x!tf.resource>, tensor<*x!tf.resource>, tensor<*x!tf.resource>)
  }
}
