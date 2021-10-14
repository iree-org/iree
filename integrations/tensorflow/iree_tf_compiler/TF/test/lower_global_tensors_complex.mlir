// RUN: iree-tf-opt -iree-tf-saved-model-lower-global-tensors -verify-diagnostics -split-input-file %s | IreeFileCheck %s

// TODO(silvasean): Make this interprocedural.

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {

// CHECK:      util.global private mutable [[V:@.+]] : tensor<?xf32> = dense<1.000000e+00> : tensor<1xf32>
// CHECK:      func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}) attributes {tf_saved_model.exported_names = ["f"]} {
// CHECK-NEXT:   [[PTR:%.+]] = util.global.address [[V]] : !util.ptr<tensor<?xf32>>
// CHECK-NEXT:   br ^bb1([[PTR]] : !util.ptr<tensor<?xf32>>)
// CHECK-NEXT: ^bb1([[PTR1:%.+]]: !util.ptr<tensor<?xf32>>):   // pred: ^bb0
// CHECK-NEXT:   util.global.store.indirect %arg0, [[PTR1]] : tensor<?xf32> -> !util.ptr<tensor<?xf32>>
// CHECK-NEXT:   return


  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}, %arg1: tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v}) attributes {tf_saved_model.exported_names = ["f"]} {
    br ^bb1(%arg1 : tensor<!tf_type.resource<tensor<?xf32>>>)
  ^bb1(%r: tensor<!tf_type.resource<tensor<?xf32>>>):
    "tf.AssignVariableOp"(%r, %arg0) : (tensor<!tf_type.resource<tensor<?xf32>>>, tensor<?xf32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {

// CHECK:      util.global private mutable [[V:@.+]] : tensor<?xf32> = dense<1.000000e+00> : tensor<1xf32>
// CHECK:      util.global private mutable [[V1:@.+]] : tensor<?xf32> = dense<1.000000e+00> : tensor<1xf32>
// CHECK:      func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}) -> (tensor<?xf32> {tf_saved_model.index_path = [0]}) attributes {tf_saved_model.exported_names = ["f"]} {
// CHECK-NEXT:   [[PTR0:%.+]] = util.global.address [[V]] : !util.ptr<tensor<?xf32>>
// CHECK-NEXT:   [[PTR1:%.+]] = util.global.address [[V1]] : !util.ptr<tensor<?xf32>>
// CHECK-NEXT:   %[[FALSE:.+]] = arith.constant false
// CHECK-NEXT:   cond_br %[[FALSE]], ^bb1([[PTR0]] : !util.ptr<tensor<?xf32>>), ^bb1([[PTR1]] : !util.ptr<tensor<?xf32>>)
// CHECK-NEXT: ^bb1([[PTR:%.+]]: !util.ptr<tensor<?xf32>>):   // 2 preds: ^bb0, ^bb0
// CHECK-NEXT:   [[T:%.+]] = util.global.load.indirect [[PTR]] : !util.ptr<tensor<?xf32>> -> tensor<?xf32>
// CHECK-NEXT:   return [[T]] : tensor<?xf32>

  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v1", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}, %v: tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v}, %v1: tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v1}) -> (tensor<?xf32> {tf_saved_model.index_path = [0]}) attributes {tf_saved_model.exported_names = ["f"]} {
    %pred = arith.constant false
    cond_br %pred, ^bb1(%v : tensor<!tf_type.resource<tensor<?xf32>>>), ^bb1(%v1 : tensor<!tf_type.resource<tensor<?xf32>>>)
  ^bb1(%either: tensor<!tf_type.resource<tensor<?xf32>>>):
    %ret = "tf.ReadVariableOp"(%either) : (tensor<!tf_type.resource<tensor<?xf32>>>) -> tensor<?xf32>
    return %ret : tensor<?xf32>
  }
}

// -----

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {

// CHECK:      util.global private mutable [[V:@.+]] : tensor<?xf32> = dense<1.000000e+00> : tensor<1xf32>
// CHECK:      func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}) attributes {tf_saved_model.exported_names = ["f"]} {
// CHECK-NEXT:   [[PTR:%.+]] = util.global.address [[V]] : !util.ptr<tensor<?xf32>>
// CHECK-NEXT:   br ^bb1([[PTR]], [[PTR]], [[PTR]] : !util.ptr<tensor<?xf32>>, !util.ptr<tensor<?xf32>>, !util.ptr<tensor<?xf32>>)
// CHECK-NEXT: ^bb1([[PTR0:%.+]]: !util.ptr<tensor<?xf32>>, [[PTR1:%.+]]: !util.ptr<tensor<?xf32>>, [[PTR2:%.+]]: !util.ptr<tensor<?xf32>>):       // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:   util.global.store.indirect %arg0, [[PTR0]] : tensor<?xf32> -> !util.ptr<tensor<?xf32>>
// CHECK-NEXT:   br ^bb1([[PTR1]], [[PTR2]], [[PTR0]] : !util.ptr<tensor<?xf32>>, !util.ptr<tensor<?xf32>>, !util.ptr<tensor<?xf32>>)

  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}, %arg1: tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v}) attributes {tf_saved_model.exported_names = ["f"]} {
    br ^bb1(%arg1, %arg1, %arg1 : tensor<!tf_type.resource<tensor<?xf32>>>, tensor<!tf_type.resource<tensor<?xf32>>>, tensor<!tf_type.resource<tensor<?xf32>>>)
  ^bb1(%0: tensor<!tf_type.resource<tensor<?xf32>>>, %1: tensor<!tf_type.resource<tensor<?xf32>>>, %2: tensor<!tf_type.resource<tensor<?xf32>>>):
    "tf.AssignVariableOp"(%0, %arg0) : (tensor<!tf_type.resource<tensor<?xf32>>>, tensor<?xf32>) -> ()
    br ^bb1(%1, %2, %0 : tensor<!tf_type.resource<tensor<?xf32>>>, tensor<!tf_type.resource<tensor<?xf32>>>, tensor<!tf_type.resource<tensor<?xf32>>>)
  }
}
