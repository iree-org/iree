// RUN: iree-opt --split-input-file --iree-stream-encode-device-tensors --verify-diagnostics %s | FileCheck %s

// Ensures that loading a non-power-of-two type (i3) is expanded to a full byte
// because we don't currently do unaligned sub-byte packing.

// CHECK-LABEL: @subspanLoadI3
stream.executable private @subspanLoadI3 {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding) {
      %c0 = arith.constant 0 : index
      // CHECK: %[[BINDING:.+]] = stream.binding.subspan {{.+}} -> !flow.dispatch.tensor<readonly:tensor<4xi8>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<4xi3>>
      // CHECK: %[[TILE_I8:.+]] = flow.dispatch.tensor.load %[[BINDING]], offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xi8>> -> tensor<?xi8>
      // CHECK: %[[TILE_I3:.+]] = arith.trunci %[[TILE_I8]] : tensor<?xi8> to tensor<?xi3>
      %tile = flow.dispatch.tensor.load %binding, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xi3>> -> tensor<?xi3>
      // CHECK: util.optimization_barrier %[[TILE_I3]] : tensor<?xi3>
      util.optimization_barrier %tile : tensor<?xi3>
      return
    }
  }
}

// -----

// Ensures that storing a non-power-of-two type (i3) is expanded to a full byte
// because we don't currently do unaligned sub-byte packing.

// CHECK-LABEL: @subspanStoreI3
stream.executable private @subspanStoreI3 {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding) {
      // CHECK: %[[CST:.+]] = arith.constant dense<[0, 7, 2, 5]> : tensor<4xi8>
      %c0 = arith.constant 0 : index
      // CHECK: %[[BINDING:.+]] = stream.binding.subspan {{.+}} -> !flow.dispatch.tensor<writeonly:tensor<4xi8>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<4xi3>>
      %cst = arith.constant dense<[0, 7, 2, 5]> : tensor<4xi3>
      // CHECK: flow.dispatch.tensor.store %[[CST]], %[[BINDING]], {{.+}} : tensor<4xi8> -> !flow.dispatch.tensor<writeonly:tensor<4xi8>>
      flow.dispatch.tensor.store %cst, %binding, offsets = [0], sizes = [4], strides = [1] : tensor<4xi3> -> !flow.dispatch.tensor<writeonly:tensor<4xi3>>
      return
    }
  }
}

// -----

// CHECK-LABEL: @subspanLoadI4
stream.executable private @subspanLoadI4 {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding) {
      %c0 = arith.constant 0 : index
      // CHECK: %[[BINDING:.+]] = stream.binding.subspan {{.+}} -> !flow.dispatch.tensor<readonly:tensor<8xi4>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<8xi4>>
      // CHECK: %[[TILE_I4:.+]] = flow.dispatch.tensor.load %[[BINDING]], {{.+}} : !flow.dispatch.tensor<readonly:tensor<8xi4>> -> tensor<?xi4>
      %tile = flow.dispatch.tensor.load %binding, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<8xi4>> -> tensor<?xi4>
      // CHECK: util.optimization_barrier %[[TILE_I4]]
      util.optimization_barrier %tile : tensor<?xi4>
      return
    }
  }
}

// -----

// CHECK-LABEL: @subspanStoreI4
stream.executable private @subspanStoreI4 {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding) {
      %c0 = arith.constant 0 : index
      // CHECK: %[[TILE_I4:.+]] = arith.constant dense<[5, -1, 0, 3, 1, 7, -8, 4]> : tensor<8xi4>
      %cst = arith.constant dense<[5, 15, 0, 3, 1, 7, 8, 4]> : tensor<8xi4>
      // CHECK: %[[BINDING:.+]] = stream.binding.subspan {{.+}} : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<8xi4>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<8xi4>>
      // CHECK: flow.dispatch.tensor.store %[[TILE_I4]], %[[BINDING]], offsets = [0], sizes = [8], strides = [1] : tensor<8xi4> -> !flow.dispatch.tensor<writeonly:tensor<8xi4>>
      flow.dispatch.tensor.store %cst, %binding, offsets = [0], sizes = [8], strides = [1] : tensor<8xi4> -> !flow.dispatch.tensor<writeonly:tensor<8xi4>>
      return
    }
  }
}

// -----

// CHECK-LABEL: @subspanLoadI8
stream.executable private @subspanLoadI8 {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding) {
      %c0 = arith.constant 0 : index
      // CHECK: %[[BINDING:.+]] = stream.binding.subspan {{.+}} -> !flow.dispatch.tensor<readonly:tensor<4xi8>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<4xi8>>
      // CHECK: %[[TILE_I8:.+]] = flow.dispatch.tensor.load %[[BINDING]], {{.+}} : !flow.dispatch.tensor<readonly:tensor<4xi8>> -> tensor<?xi8>
      %tile = flow.dispatch.tensor.load %binding, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xi8>> -> tensor<?xi8>
      // CHECK: util.optimization_barrier %[[TILE_I8]]
      util.optimization_barrier %tile : tensor<?xi8>
      return
    }
  }
}

// -----

// CHECK-LABEL: @subspanStoreI8
stream.executable private @subspanStoreI8 {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding) {
      %c0 = arith.constant 0 : index
      // CHECK-DAG: %[[TILE_I8:.+]] = arith.constant dense<[25, 8, 0, -1]> : tensor<4xi8>
      // CHECK-DAG: %[[BINDING:.+]] = stream.binding.subspan {{.+}} -> !flow.dispatch.tensor<writeonly:tensor<4xi8>>
      %binding = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<4xi8>>
      %cst = arith.constant dense<[25, 8, 0, 255]> : tensor<4xi8>
      // CHECK-NEXT: flow.dispatch.tensor.store %[[TILE_I8]], %[[BINDING]], {{.+}} : tensor<4xi8> -> !flow.dispatch.tensor<writeonly:tensor<4xi8>>
      flow.dispatch.tensor.store %cst, %binding, offsets = [0], sizes = [4], strides = [1] : tensor<4xi8> -> !flow.dispatch.tensor<writeonly:tensor<4xi8>>
      return
    }
  }
}
