// RUN: iree-tf-opt -iree-tf-saved-model-lower-global-tensors -split-input-file -verify-diagnostics %s

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func @f(%arg0: tensor<!tf.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    // expected-error@+1 {{could not lower resource op to flow: tf.SomeUnknownVariableOp}}
    "tf.SomeUnknownVariableOp"(%arg0) : (tensor<!tf.resource<tensor<?xf32>>>) -> ()
    return
  }
}
