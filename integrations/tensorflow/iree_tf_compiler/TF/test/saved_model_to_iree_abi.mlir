// RUN: iree-tf-opt %s -tf-saved-model-to-iree-abi -split-input-file -verify-diagnostics | IreeFileCheck %s

// CHECK-LABEL: module @binary_func
// Should just be a pass through.
// CHECK: func @binary_func
// CHECK-SAME{LITERAL}: iree.abi = "{\22a\22:[[\22ndarray\22,\22f32\22,1,16],[\22ndarray\22,\22f32\22,1,16]],\22r\22:[[\22ndarray\22,\22f32\22,1,16],[\22ndarray\22,\22f32\22,1,16]],\22v\22:1}"
// CHECK: %[[ARG0_TENSOR:.*]] = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<16xf32>
// CHECK: %[[ARG1_TENSOR:.*]] = hal.tensor.cast %arg1 : !hal.buffer_view -> tensor<16xf32>
// CHECK: %[[R:.*]]:2 = call @__inference_binary_func_70(%[[ARG0_TENSOR]], %[[ARG1_TENSOR]])
// CHECK: %[[R0_BV:.*]] = hal.tensor.cast %[[R]]#0 : tensor<16xf32> -> !hal.buffer_view
// CHECK: %[[R1_BV:.*]] = hal.tensor.cast %[[R]]#1 : tensor<16xf32> -> !hal.buffer_view
// CHECK: return %[[R0_BV]], %[[R1_BV]] : !hal.buffer_view, !hal.buffer_view
// CHECK: func private @__inference_binary_func_70
// CHECK-NOT: tf_saved_model
module @binary_func attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 729 : i32}, tf_saved_model.semantics}  {
  func @__inference_binary_func_70(%arg0: tensor<16xf32> {tf._user_specified_name = "a", tf_saved_model.index_path = [0]}, %arg1: tensor<16xf32> {tf._user_specified_name = "b", tf_saved_model.index_path = [1]}) -> (tensor<16xf32> {tf_saved_model.index_path = [0]}, tensor<16xf32> {tf_saved_model.index_path = [1]}) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<16>, #tf_type.shape<16>], tf_saved_model.exported_names = ["binary_func"]} {
    %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<16xf32>) -> tensor<16xf32>
    %1 = "tf.Identity"(%arg1) {device = ""} : (tensor<16xf32>) -> tensor<16xf32>
    return %0, %1 : tensor<16xf32>, tensor<16xf32>
  }
}

// -----
// CHECK-LABEL: module @unary_func
// CHECK: func @unary_func
// CHECK-SAME{LITERAL}: iree.abi = "{\22a\22:[[\22ndarray\22,\22f32\22,1,16]],\22r\22:[[\22ndarray\22,\22f32\22,1,16]],\22v\22:1}"
// CHECK: %[[ARG0_TENSOR:.*]] = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<16xf32>
// CHECK: %[[R:.*]] = call @__inference_unary_func_240(%[[ARG0_TENSOR]])
// CHECK: %[[R0_BV:.*]] = hal.tensor.cast %[[R]] : tensor<16xf32> -> !hal.buffer_view
// CHECK: return %[[R0_BV]] : !hal.buffer_view
// CHECK: func private @__inference_unary_func_240
  // CHECK-NOT: tf_saved_model
module @unary_func attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 729 : i32}, tf_saved_model.semantics}  {
  func @__inference_unary_func_240(%arg0: tensor<16xf32> {tf._user_specified_name = "a", tf_saved_model.index_path = [0]}) -> (tensor<16xf32> {tf_saved_model.index_path = []}) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<16>], tf_saved_model.exported_names = ["unary_func"]} {
    %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<16xf32>) -> tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}

// -----
// CHECK-LABEL: module @return_list
// CHECK: func @return_list
// CHECK-SAME{LITERAL}: iree.abi = "{\22a\22:[[\22ndarray\22,\22f32\22,1,16],[\22ndarray\22,\22f32\22,1,16]],\22r\22:[[\22ndarray\22,\22f32\22,1,16],[\22ndarray\22,\22f32\22,1,16]],\22v\22:1}"
// CHECK: %[[ARG0_TENSOR:.*]] = hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<16xf32>
// CHECK: %[[ARG1_TENSOR:.*]] = hal.tensor.cast %arg1 : !hal.buffer_view -> tensor<16xf32>
// CHECK: %[[R:.+]]:2 = call @__inference_return_list_260(%[[ARG0_TENSOR]], %[[ARG1_TENSOR]])
// CHECK: %[[R0_BV:.*]] = hal.tensor.cast %[[R]]#0 : tensor<16xf32> -> !hal.buffer_view
// CHECK: %[[R1_BV:.*]] = hal.tensor.cast %[[R]]#1 : tensor<16xf32> -> !hal.buffer_view
// CHECK: return %[[R0_BV]], %[[R1_BV]] : !hal.buffer_view, !hal.buffer_view
// CHECK: func private @__inference_return_list_260
// CHECK-NOT: tf_saved_model
module @return_list attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 729 : i32}, tf_saved_model.semantics}  {
  func @__inference_return_list_260(%arg0: tensor<16xf32> {tf._user_specified_name = "a", tf_saved_model.index_path = [0]}, %arg1: tensor<16xf32> {tf._user_specified_name = "b", tf_saved_model.index_path = [1]}) -> (tensor<16xf32> {tf_saved_model.index_path = [0]}, tensor<16xf32> {tf_saved_model.index_path = [1]}) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<16>, #tf_type.shape<16>], tf_saved_model.exported_names = ["return_list"]} {
    %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<16xf32>) -> tensor<16xf32>
    %1 = "tf.Identity"(%arg1) {device = ""} : (tensor<16xf32>) -> tensor<16xf32>
    return %0, %1 : tensor<16xf32>, tensor<16xf32>
  }
}

// -----
// CHECK-LABEL: module @dict_nest
// CHECK: func @dict_nest
// CHECK-SAME{LITERAL}: iree.abi = "{\22a\22:[[\22sdict\22,[\22dict\22,[\22sdict\22,[\22a\22,[\22ndarray\22,\22f32\22,1,16]],[\22b\22,[\22ndarray\22,\22f32\22,1,16]]]],[\22list\22,[\22slist\22,[\22ndarray\22,\22f32\22,1,16],[\22ndarray\22,\22f32\22,1,16]]]],[\22ndarray\22,\22f32\22,0]],\22r\22:[[\22sdict\22,[\22dict\22,[\22sdict\22,[\22a\22,[\22ndarray\22,\22f32\22,1,16]],[\22b\22,[\22ndarray\22,\22f32\22,1,16]]]],[\22list\22,[\22stuple\22,[\22ndarray\22,\22f32\22,1,16],[\22ndarray\22,\22f32\22,1,16]]]]],\22v\22:1}"
// CHECK: %[[c0:.+]] = constant 0 : index
// CHECK: %[[L0:.+]] = iree.list.get %arg0[%[[c0]]] : !iree.list<?> -> !iree.list<?>
// CHECK: %[[c0_0:.+]] = constant 0 : index
// CHECK: %[[L1:.+]] = iree.list.get %[[L0]][%[[c0_0]]] : !iree.list<?> -> !hal.buffer_view
// CHECK: %[[L1_TENSOR:.+]] = hal.tensor.cast %[[L1]] : !hal.buffer_view -> tensor<16xf32>
// CHECK: %[[c1:.+]] = constant 1 : index
// CHECK: %[[L2:.+]] = iree.list.get %[[L0]][%[[c1]]] : !iree.list<?> -> !hal.buffer_view
// CHECK: %[[L2_TENSOR:.+]] = hal.tensor.cast %[[L2]] : !hal.buffer_view -> tensor<16xf32>
// CHECK: %[[c1_1:.+]] = constant 1 : index
// CHECK: %[[L3:.+]] = iree.list.get %arg0[%[[c1_1]]] : !iree.list<?> -> !iree.list<?>
// CHECK: %[[c0_2:.+]] = constant 0 : index
// CHECK: %[[L4:.+]] = iree.list.get %[[L3]][%[[c0_2]]] : !iree.list<?> -> !hal.buffer_view
// CHECK: %[[L4_TENSOR:.+]] = hal.tensor.cast %[[L4]] : !hal.buffer_view -> tensor<16xf32>
// CHECK: %[[c1_3:.+]] = constant 1 : index
// CHECK: %[[L5:.+]] = iree.list.get %[[L3]][%[[c1_3]]] : !iree.list<?> -> !hal.buffer_view
// CHECK: %[[L5_TENSOR:.+]] = hal.tensor.cast %[[L5]] : !hal.buffer_view -> tensor<16xf32>
// CHECK: %[[ARG1_TENSOR:.+]] = hal.tensor.cast %arg1 : !hal.buffer_view -> tensor<f32>
// CHECK: %[[RESULT:.+]]:4 = call @__inference_dict_nest_190(%[[L1_TENSOR]], %[[L2_TENSOR]], %[[L4_TENSOR]], %[[L5_TENSOR]], %[[ARG1_TENSOR]]) : (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<f32>) -> (tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>)
// CHECK: %[[c2:.+]] = constant 2 : index
// CHECK: %[[R7:.+]] = iree.list.create %[[c2]] : !iree.list<?>
// CHECK: %[[c2_4:.+]] = constant 2 : index
// CHECK: %[[R8:.+]] = iree.list.create %[[c2_4]] : !iree.list<?>
// CHECK: %[[R0_BV:.+]] = hal.tensor.cast %[[RESULT]]#0 : tensor<16xf32> -> !hal.buffer_view
// CHECK: %[[c0_5:.+]] = constant 0 : index
// CHECK: iree.list.set %[[R8]][%[[c0_5]]], %[[R0_BV]] : !hal.buffer_view -> !iree.list<?>
// CHECK: %[[R1_BV:.+]] = hal.tensor.cast %[[RESULT]]#1 : tensor<16xf32> -> !hal.buffer_view
// CHECK: %[[c1_6:.+]] = constant 1 : index
// CHECK: iree.list.set %[[R8]][%[[c1_6]]], %[[R1_BV]] : !hal.buffer_view -> !iree.list<?>
// CHECK: %[[c0_7:.+]] = constant 0 : index
// CHECK: iree.list.set %[[R7]][%[[c0_7]]], %[[R8]] : !iree.list<?> -> !iree.list<?>
// CHECK: %[[c2_8:.+]] = constant 2 : index
// CHECK: %[[R9:.+]] = iree.list.create %[[c2_8]] : !iree.list<?>
// CHECK: %[[R2_BV:.+]] = hal.tensor.cast %[[RESULT]]#2 : tensor<16xf32> -> !hal.buffer_view
// CHECK: %[[c0_9:.+]] = constant 0 : index
// CHECK: iree.list.set %[[R9]][%[[c0_9]]], %[[R2_BV]] : !hal.buffer_view -> !iree.list<?>
// CHECK: %[[R3_BV:.+]] = hal.tensor.cast %[[RESULT]]#3 : tensor<16xf32> -> !hal.buffer_view
// CHECK: %[[c1_10:.+]] = constant 1 : index
// CHECK: iree.list.set %[[R9]][%[[c1_10]]], %[[R3_BV]] : !hal.buffer_view -> !iree.list<?>
// CHECK: %[[c1_11:.+]] = constant 1 : index
// CHECK: iree.list.set %[[R7]][%[[c1_11]]], %[[R9]] : !iree.list<?> -> !iree.list<?>
// return %[[R7]] : !iree.list<?>
// CHECK: func private @__inference_dict_nest_190
// CHECK-NOT: tf_saved_model
module @dict_nest attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 729 : i32}, tf_saved_model.semantics}  {
  func @__inference_dict_nest_190(%arg0: tensor<16xf32> {tf._user_specified_name = "mapping", tf_saved_model.index_path = [0, "dict", "a"]}, %arg1: tensor<16xf32> {tf._user_specified_name = "mapping", tf_saved_model.index_path = [0, "dict", "b"]}, %arg2: tensor<16xf32> {tf._user_specified_name = "mapping", tf_saved_model.index_path = [0, "list", 0]}, %arg3: tensor<16xf32> {tf._user_specified_name = "mapping", tf_saved_model.index_path = [0, "list", 1]}, %arg4: tensor<f32> {tf._user_specified_name = "scalar", tf_saved_model.index_path = [1]}) -> (tensor<16xf32> {tf_saved_model.index_path = ["dict", "a"]}, tensor<16xf32> {tf_saved_model.index_path = ["dict", "b"]}, tensor<16xf32> {tf_saved_model.index_path = ["list", 0]}, tensor<16xf32> {tf_saved_model.index_path = ["list", 1]}) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<16>, #tf_type.shape<16>, #tf_type.shape<16>, #tf_type.shape<16>, #tf_type.shape<>], tf_saved_model.exported_names = ["dict_nest"]} {
    %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<16xf32>) -> tensor<16xf32>
    %1 = "tf.Identity"(%arg1) {device = ""} : (tensor<16xf32>) -> tensor<16xf32>
    %2 = "tf.Identity"(%arg2) {device = ""} : (tensor<16xf32>) -> tensor<16xf32>
    %3 = "tf.Identity"(%arg3) {device = ""} : (tensor<16xf32>) -> tensor<16xf32>
    return %0, %1, %2, %3 : tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>
  }
}

// -----
// CHECK-LABEL: module @kwargs
// CHECK: func @dict_nest
// CHECK-SAME{LITERAL}: iree.abi = "{\22a\22:[[\22sdict\22,[\22list\22,[\22slist\22,[\22ndarray\22,\22f32\22,1,16],[\22ndarray\22,\22f32\22,1,16]]]],[\22ndarray\22,\22f32\22,0],[\22sdict_kwargs\22,[\22a\22,[\22ndarray\22,\22f32\22,1,16]],[\22b\22,[\22ndarray\22,\22f32\22,1,16]]]],\22r\22:[[\22sdict\22,[\22dict\22,[\22sdict\22,[\22a\22,[\22ndarray\22,\22f32\22,1,16]],[\22b\22,[\22ndarray\22,\22f32\22,1,16]]]],[\22list\22,[\22stuple\22,[\22ndarray\22,\22f32\22,1,16],[\22ndarray\22,\22f32\22,1,16]]]]],\22v\22:1}"
module @kwargs attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 729 : i32}, tf_saved_model.semantics}  {
  func @__inference_dict_nest_190(%arg0: tensor<16xf32> {tf_saved_model.index_path = ["a"]}, %arg1: tensor<16xf32> {tf_saved_model.index_path = ["b"]}, %arg2: tensor<16xf32> {tf._user_specified_name = "mapping", tf_saved_model.index_path = [0, "list", 0]}, %arg3: tensor<16xf32> {tf._user_specified_name = "mapping", tf_saved_model.index_path = [0, "list", 1]}, %arg4: tensor<f32> {tf._user_specified_name = "scalar", tf_saved_model.index_path = [1]}) -> (tensor<16xf32> {tf_saved_model.index_path = ["dict", "a"]}, tensor<16xf32> {tf_saved_model.index_path = ["dict", "b"]}, tensor<16xf32> {tf_saved_model.index_path = ["list", 0]}, tensor<16xf32> {tf_saved_model.index_path = ["list", 1]}) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<16>, #tf_type.shape<16>, #tf_type.shape<16>, #tf_type.shape<16>, #tf_type.shape<>], tf_saved_model.exported_names = ["dict_nest"]} {
    %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<16xf32>) -> tensor<16xf32>
    %1 = "tf.Identity"(%arg1) {device = ""} : (tensor<16xf32>) -> tensor<16xf32>
    %2 = "tf.Identity"(%arg2) {device = ""} : (tensor<16xf32>) -> tensor<16xf32>
    %3 = "tf.Identity"(%arg3) {device = ""} : (tensor<16xf32>) -> tensor<16xf32>
    return %0, %1, %2, %3 : tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>
  }
}
