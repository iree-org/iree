// RUN: iree-opt --split-input-file --iree-stream-encode-device-tensors %s | FileCheck %s

// CHECK-LABEL: @convert_load_i1
stream.executable private @convert_load_i1 {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding) {
      %c0 = arith.constant 0 : index
      // CHECK: %[[BINDING:.+]] = stream.binding.subspan {{.+}} -> !flow.dispatch.tensor<readonly:tensor<4xi8>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<4xi1>>
      // CHECK: %[[TILE_I8:.+]] = flow.dispatch.tensor.load %[[BINDING]], {{.+}} : !flow.dispatch.tensor<readonly:tensor<4xi8>> -> tensor<?xi8>
      // CHECK: %[[TILE_I1:.+]] = arith.trunci %[[TILE_I8]] : tensor<?xi8> to tensor<?xi1>
      %tile = flow.dispatch.tensor.load %binding, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xi1>> -> tensor<?xi1>
      // CHECK: util.optimization_barrier %[[TILE_I1]]
      util.optimization_barrier %tile : tensor<?xi1>
      return
    }
  }
}

// -----

// CHECK-LABEL: @convert_store_i1
stream.executable private @convert_store_i1 {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding) {
      %c0 = arith.constant 0 : index
      // CHECK-DAG: %[[TILE_I8:.+]] = arith.constant dense<[0, 0, 1, 1]> : tensor<4xi8>
      // CHECK-DAG: %[[BINDING:.+]] = stream.binding.subspan {{.+}} -> !flow.dispatch.tensor<writeonly:tensor<4xi8>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<4xi1>>
      %cst = arith.constant dense<[false, false, true, true]> : tensor<4xi1>
      // CHECK-NEXT: flow.dispatch.tensor.store %[[TILE_I8]], %[[BINDING]], {{.+}} : tensor<4xi8> -> !flow.dispatch.tensor<writeonly:tensor<4xi8>>
      flow.dispatch.tensor.store %cst, %binding, offsets = [0], sizes = [4], strides = [1] : tensor<4xi1> -> !flow.dispatch.tensor<writeonly:tensor<4xi1>>
      return
    }
  }
}

// -----

// CHECK-LABEL: @convert_multi_i1
stream.executable private @convert_multi_i1 {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding) {
      %c0 = arith.constant 0 : index
      // CHECK-DAG: %[[BINDING0:.+]] = stream.binding.subspan %arg0{{.+}} -> !flow.dispatch.tensor<readonly:tensor<4xi8>>
      %binding0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<4xi1>>
      // CHECK-DAG: %[[BINDING1:.+]] = stream.binding.subspan %arg1{{.+}} -> !flow.dispatch.tensor<readwrite:tensor<4xi8>>
      %binding1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<4xi1>>
      // CHECK: %[[TILE0_I8:.+]] = flow.dispatch.tensor.load %[[BINDING0]], {{.+}} : !flow.dispatch.tensor<readonly:tensor<4xi8>> -> tensor<?xi8>
      // CHECK: %[[TILE0_I1:.+]] = arith.trunci %[[TILE0_I8]] : tensor<?xi8> to tensor<?xi1>
      %tile0 = flow.dispatch.tensor.load %binding0, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xi1>> -> tensor<?xi1>
      // CHECK: %[[TILE1_I8:.+]] = flow.dispatch.tensor.load %[[BINDING1]], {{.+}} : !flow.dispatch.tensor<readwrite:tensor<4xi8>> -> tensor<?xi8>
      // CHECK: %[[TILE1_I1:.+]] = arith.trunci %[[TILE1_I8]] : tensor<?xi8> to tensor<?xi1>
      %tile1 = flow.dispatch.tensor.load %binding1, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<4xi1>> -> tensor<?xi1>
      // CHECK: %[[RESULT_I1:.+]] = arith.ori %[[TILE0_I1]], %[[TILE1_I1]] : tensor<?xi1>
      %result = arith.ori %tile0, %tile1 : tensor<?xi1>
      // CHECK: %[[RESULT_I8:.+]] = arith.extui %[[RESULT_I1]] : tensor<?xi1> to tensor<?xi8>
      // CHECK-NEXT: flow.dispatch.tensor.store %[[RESULT_I8]], %[[BINDING1]], {{.+}} : tensor<?xi8> -> !flow.dispatch.tensor<readwrite:tensor<4xi8>>
      flow.dispatch.tensor.store %result, %binding1, offsets = [0], sizes = [4], strides = [1] : tensor<?xi1> -> !flow.dispatch.tensor<readwrite:tensor<4xi1>>
      return
    }
  }
}

// -----

// CHECK-LABEL: @convert_load_i33
stream.executable private @convert_load_i33 {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding) {
      %c0 = arith.constant 0 : index
      // CHECK: %[[BINDING:.+]] = stream.binding.subspan {{.+}} -> !flow.dispatch.tensor<readonly:tensor<4xi64>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<4xi33>>
      // CHECK: %[[TILE_I8:.+]] = flow.dispatch.tensor.load %[[BINDING]], {{.+}} : !flow.dispatch.tensor<readonly:tensor<4xi64>> -> tensor<?xi64>
      // CHECK: %[[TILE_I1:.+]] = arith.trunci %[[TILE_I8]] : tensor<?xi64> to tensor<?xi33>
      %tile = flow.dispatch.tensor.load %binding, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xi33>> -> tensor<?xi33>
      // CHECK: util.optimization_barrier %[[TILE_I1]]
      util.optimization_barrier %tile : tensor<?xi33>
      return
    }
  }
}

// -----

// CHECK-LABEL: @convert_store_i33

stream.executable private @convert_store_i33 {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding) {
      // CHECK: %[[CST:.+]] = arith.constant dense<[0, 7, 2, 5]> : tensor<4xi64>
      %c0 = arith.constant 0 : index
      // CHECK: %[[BINDING:.+]] = stream.binding.subspan {{.+}} -> !flow.dispatch.tensor<writeonly:tensor<4xi64>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<4xi33>>
      %cst = arith.constant dense<[0, 7, 2, 5]> : tensor<4xi33>
      // CHECK: flow.dispatch.tensor.store %[[CST]], %[[BINDING]], {{.+}} : tensor<4xi64> -> !flow.dispatch.tensor<writeonly:tensor<4xi64>>
      flow.dispatch.tensor.store %cst, %binding, offsets = [0], sizes = [4], strides = [1] : tensor<4xi33> -> !flow.dispatch.tensor<writeonly:tensor<4xi33>>
      return
    }
  }
}
