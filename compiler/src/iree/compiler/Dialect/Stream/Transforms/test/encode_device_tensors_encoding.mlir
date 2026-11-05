// RUN: iree-opt --split-input-file --iree-stream-encode-device-tensors %s | FileCheck %s

// CHECK-LABEL: @convert_load_i1
stream.executable private @convert_load_i1 {
  stream.executable.export public @dispatch
  builtin.module {
     util.func public @dispatch(%arg0: !stream.binding) {
      %c0 = arith.constant 0 : index
      // CHECK: %[[BINDING_0:.*]] = stream.binding.subspan %arg0{{.+}} : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi1, #iree_encoding.packed_storage>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi1, #iree_encoding.packed_storage>>
      // CHECK: %[[DISPATCH_0:.*]] = iree_tensor_ext.dispatch.tensor.load %[[BINDING_0]], offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi1, #iree_encoding.packed_storage>> -> tensor<?xi1, #iree_encoding.packed_storage>
      %tile = iree_tensor_ext.dispatch.tensor.load %binding, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi1, #iree_encoding.packed_storage>> -> tensor<?xi1, #iree_encoding.packed_storage>
      // CHECK: util.optimization_barrier %[[DISPATCH_0]] : tensor<?xi1, #iree_encoding.packed_storage>
      util.optimization_barrier %tile : tensor<?xi1, #iree_encoding.packed_storage>
      util.return
    }
  }
}

// -----

// CHECK-LABEL: @convert_store_i1
stream.executable private @convert_store_i1 {
  stream.executable.export public @dispatch
  builtin.module {
     util.func public @dispatch(%arg0: !stream.binding) {
      // CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant dense<[false, false, true, true]> : tensor<4xi1, #iree_encoding.packed_storage>
      %c0 = arith.constant 0 : index
      // CHECK-DAG: %[[BINDING_0:.*]] = stream.binding.subspan %arg0{{.+}} : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi1, #iree_encoding.packed_storage>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi1, #iree_encoding.packed_storage>>
      %cst = arith.constant dense<[false, false, true, true]> : tensor<4xi1, #iree_encoding.packed_storage>
      // CHECK-NEXT: iree_tensor_ext.dispatch.tensor.store %[[CONSTANT_0]], %[[BINDING_0]], offsets = [0], sizes = [4], strides = [1] : tensor<4xi1, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi1, #iree_encoding.packed_storage>>
      iree_tensor_ext.dispatch.tensor.store %cst, %binding, offsets = [0], sizes = [4], strides = [1] : tensor<4xi1, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi1, #iree_encoding.packed_storage>>
      util.return
    }
  }
}

// -----

// CHECK-LABEL: @convert_multi_i1
stream.executable private @convert_multi_i1 {
  stream.executable.export public @dispatch
  builtin.module {
     util.func public @dispatch(%arg0: !stream.binding, %arg1: !stream.binding) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      // CHECK: %[[BINDING_0:.*]] = stream.binding.subspan %arg0{{.+}} : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi1, #iree_encoding.packed_storage>>
      %binding0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi1, #iree_encoding.packed_storage>>
      // CHECK: %[[BINDING_1:.*]] = stream.binding.subspan %arg1{{.+}} : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi1, #iree_encoding.packed_storage>>
      %binding1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi1, #iree_encoding.packed_storage>>
      // CHECK: %[[DISPATCH_0:.*]] = iree_tensor_ext.dispatch.tensor.load %[[BINDING_0]], offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi1, #iree_encoding.packed_storage>> -> tensor<?xi1, #iree_encoding.packed_storage>
      %tile0 = iree_tensor_ext.dispatch.tensor.load %binding0, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi1, #iree_encoding.packed_storage>> -> tensor<?xi1, #iree_encoding.packed_storage>
      // CHECK: %[[DISPATCH_1:.*]] = iree_tensor_ext.dispatch.tensor.load %[[BINDING_1]], offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi1, #iree_encoding.packed_storage>> -> tensor<?xi1, #iree_encoding.packed_storage>
      %tile1 = iree_tensor_ext.dispatch.tensor.load %binding1, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi1, #iree_encoding.packed_storage>> -> tensor<?xi1, #iree_encoding.packed_storage>
      // CHECK: %[[ORI_0:.*]] = arith.ori %[[DISPATCH_0]], %[[DISPATCH_1]] : tensor<?xi1, #iree_encoding.packed_storage>
      %result = arith.ori %tile0, %tile1 : tensor<?xi1, #iree_encoding.packed_storage>
      // CHECK-NEXT: iree_tensor_ext.dispatch.tensor.store %[[ORI_0]], %[[BINDING_1]], {{.+}} : tensor<?xi1, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi1, #iree_encoding.packed_storage>>
      iree_tensor_ext.dispatch.tensor.store %result, %binding1, offsets = [0], sizes = [%c4], strides = [1] : tensor<?xi1, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4xi1, #iree_encoding.packed_storage>>
      util.return
    }
  }
}

// -----

// Check that i4 are packed and not extended to a full byte. This is also the default behavior without the 'packed_storage' encoding,
// so just making sure it still works with the encoding attached.

// CHECK-LABEL: @convert_load_i4
stream.executable private @convert_load_i4 {
  stream.executable.export public @dispatch
  builtin.module {
     util.func public @dispatch(%arg0: !stream.binding) {
      %c0 = arith.constant 0 : index
      // CHECK: %[[BINDING_0:.*]] = stream.binding.subspan %arg0{{.+}} : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi4, #iree_encoding.packed_storage>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi4, #iree_encoding.packed_storage>>
      // CHECK: %[[DISPATCH_0:.*]] = iree_tensor_ext.dispatch.tensor.load %[[BINDING_0]], offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi4, #iree_encoding.packed_storage>> -> tensor<?xi4, #iree_encoding.packed_storage>
      %tile = iree_tensor_ext.dispatch.tensor.load %binding, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi4, #iree_encoding.packed_storage>> -> tensor<?xi4, #iree_encoding.packed_storage>
      // CHECK: util.optimization_barrier %[[DISPATCH_0]] : tensor<?xi4, #iree_encoding.packed_storage>
      util.optimization_barrier %tile : tensor<?xi4, #iree_encoding.packed_storage>
      util.return
    }
  }
}

// -----

// CHECK-LABEL: @convert_store_i4

stream.executable private @convert_store_i4 {
  stream.executable.export public @dispatch
  builtin.module {
     util.func public @dispatch(%arg0: !stream.binding) {
      // CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant dense<[0, 7, 2, 5]> : tensor<4xi4, #iree_encoding.packed_storage>
      %c0 = arith.constant 0 : index
      // CHECK-DAG: %[[BINDING_0:.*]] = stream.binding.subspan %arg0{{.+}} : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi4, #iree_encoding.packed_storage>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi4, #iree_encoding.packed_storage>>
      %cst = arith.constant dense<[0, 7, 2, 5]> : tensor<4xi4, #iree_encoding.packed_storage>
      // CHECK-NEXT: iree_tensor_ext.dispatch.tensor.store %[[CONSTANT_0]], %[[BINDING_0]], offsets = [0], sizes = [4], strides = [1] : tensor<4xi4, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi4, #iree_encoding.packed_storage>>
      iree_tensor_ext.dispatch.tensor.store %cst, %binding, offsets = [0], sizes = [4], strides = [1] : tensor<4xi4, #iree_encoding.packed_storage> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi4, #iree_encoding.packed_storage>>
      util.return
    }
  }
}

// -----

// CHECK-LABEL: @unpacked_load_i5
// CHECK: stream.binding.subspan {{.*}}tensor<4xi8>
// CHECK: %[[LOAD:.*]] = iree_tensor_ext.dispatch.tensor.load {{.*}} -> tensor<4xi8
// CHECK: arith.trunci %[[LOAD]] : tensor<4xi8{{.*}}> to tensor<4xi5, #iree_encoding.unpacked_storage>
stream.executable private @unpacked_load_i5 {
  stream.executable.export public @dispatch
  builtin.module {
    util.func public @dispatch(%binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %buffer = stream.binding.subspan %binding[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi5, #iree_encoding.unpacked_storage>>
      %value = iree_tensor_ext.dispatch.tensor.load %buffer, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi5, #iree_encoding.unpacked_storage>> -> tensor<4xi5, #iree_encoding.unpacked_storage>
      util.optimization_barrier %value : tensor<4xi5, #iree_encoding.unpacked_storage>
      util.return
    }
  }
}

// -----

// CHECK-LABEL: @unpacked_store_i5
// CHECK: stream.binding.subspan {{.*}}tensor<4xi8>
// CHECK: %[[EXT:.*]] = arith.extui {{.*}} : tensor<4xi5, #iree_encoding.unpacked_storage> to tensor<4xi8
// CHECK: iree_tensor_ext.dispatch.tensor.store %[[EXT]]
stream.executable private @unpacked_store_i5 {
  stream.executable.export public @dispatch
  builtin.module {
    util.func public @dispatch(%binding: !stream.binding, %value: tensor<4xi5, #iree_encoding.unpacked_storage>) {
      %c0 = arith.constant 0 : index
      %buffer = stream.binding.subspan %binding[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi5, #iree_encoding.unpacked_storage>>
      iree_tensor_ext.dispatch.tensor.store %value, %buffer, offsets = [0], sizes = [4], strides = [1] : tensor<4xi5, #iree_encoding.unpacked_storage> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi5, #iree_encoding.unpacked_storage>>
      util.return
    }
  }
}
