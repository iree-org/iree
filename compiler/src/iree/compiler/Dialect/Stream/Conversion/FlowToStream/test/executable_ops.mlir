// RUN: iree-opt --split-input-file --iree-stream-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @extern_executable
flow.executable private @extern_executable {
  // CHECK: stream.executable.export public @dispatch
  flow.executable.export public @dispatch
  // CHECK-NOT: builtin.module
}

// -----

// CHECK-LABEL: @workgroup_count_region
flow.executable private @workgroup_count_region {
  // CHECK-NEXT: stream.executable.export public @dispatch
  flow.executable.export public @dispatch
      // CHECK-SAME: workgroups(%[[ARG0:.+]]: index) -> (index, index, index) {
      workgroups(%arg0: index) -> (index, index, index) {
        // CHECK-NEXT: stream.return %[[ARG0]], %[[ARG0]], %[[ARG0]] : index, index, index
        flow.return %arg0, %arg0, %arg0 : index, index, index
      }
  builtin.module {
    // CHECK: util.func public @dispatch()
    util.func public @dispatch() {
      util.return
    }
  }
}

// -----

// CHECK-LABEL: @rank_0_binding
flow.executable private @rank_0_binding {
  flow.executable.export public @dispatch
  builtin.module {
    // CHECK: util.func public @dispatch(%[[INPUT:.+]]: !stream.binding)
    util.func public @dispatch(%input: !iree_tensor_ext.dispatch.tensor<readonly:tensor<i64>>) {
      // CHECK: %[[SUBSPAN:.+]] = stream.binding.subspan %[[INPUT]][%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<i64>>
      // CHECK: = iree_tensor_ext.dispatch.tensor.load %[[SUBSPAN]]
      %tied_input = iree_tensor_ext.dispatch.tensor.load %input, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i64>> -> tensor<i64>
      util.optimization_barrier %tied_input : tensor<i64>
      util.return
    }
  }
}

// -----

// CHECK-LABEL: @static_bindings
flow.executable private @static_bindings {
  flow.executable.export public @dispatch
  builtin.module {
    // CHECK: util.func public @dispatch(%[[INPUT:.+]]: !stream.binding, %[[OUTPUT:.+]]: !stream.binding)
    util.func public @dispatch(%input: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4xf32>>, %output: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>) {
      // CHECK-DAG: %[[TIED_INPUT:.+]] = stream.binding.subspan %[[INPUT]][%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4xf32>>
      // CHECK-DAG: %[[TIED_OUTPUT:.+]] = stream.binding.subspan %[[OUTPUT]][%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
      %tied_input = flow.dispatch.tie_shape %input : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4xf32>>
      %tied_output = flow.dispatch.tie_shape %output : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>

      // CHECK: %[[TILE:.+]] = iree_tensor_ext.dispatch.tensor.load %[[TIED_INPUT]]
      // CHECK: iree_tensor_ext.dispatch.tensor.store %[[TILE]], %[[TIED_OUTPUT]]
      %tile = iree_tensor_ext.dispatch.tensor.load %tied_input, offsets = [0, 0], sizes = [1, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4xf32>> -> tensor<4xf32>
      iree_tensor_ext.dispatch.tensor.store %tile, %tied_output, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
      util.return
    }
  }
}

// -----

// CHECK-LABEL: @dynamic_bindings
flow.executable private @dynamic_bindings {
  flow.executable.export public @dispatch
  builtin.module {
    // CHECK: util.func public @dispatch(%[[DIM:.+]]: index, %[[INPUT:.+]]: !stream.binding, %[[OUTPUT:.+]]: !stream.binding)
    util.func public @dispatch(%dim: index, %input: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?xf32>>, %output: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>) {
      // CHECK-DAG: %[[TIED_INPUT:.+]] = stream.binding.subspan %[[INPUT]][%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?xf32>>{%[[DIM]]}
      // CHECK-DAG: %[[TIED_OUTPUT:.+]] = stream.binding.subspan %[[OUTPUT]][%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%[[DIM]]}
      %tied_input = flow.dispatch.tie_shape %input : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?xf32>>{%dim}
      %tied_output = flow.dispatch.tie_shape %output : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%dim}

      // CHECK: %[[TILE:.+]] = iree_tensor_ext.dispatch.tensor.load %[[TIED_INPUT]]
      // CHECK: iree_tensor_ext.dispatch.tensor.store %[[TILE]], %[[TIED_OUTPUT]]
      %tile = iree_tensor_ext.dispatch.tensor.load %tied_input, offsets = [0, 0], sizes = [1, %dim], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?xf32>>{%dim} -> tensor<?xf32>
      iree_tensor_ext.dispatch.tensor.store %tile, %tied_output, offsets = [0], sizes = [%dim], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%dim}
      util.return
    }
  }
}

// -----

// CHECK-LABEL: @indirect_dynamic_bindings
flow.executable private @indirect_dynamic_bindings {
  flow.executable.export public @dispatch
  builtin.module {
    // CHECK: util.func public @dispatch(%[[DIM_TENSOR:.+]]: !stream.binding, %[[INPUT:.+]]: !stream.binding, %[[OUTPUT:.+]]: !stream.binding)
    util.func public @dispatch(%dim_tensor: !iree_tensor_ext.dispatch.tensor<readonly:tensor<i64>>, %input: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?xf32>>, %output: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>) {
      // CHECK: %[[DIM_SUBSPAN:.+]] = stream.binding.subspan %[[DIM_TENSOR]][%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<i64>>
      // CHECK: %[[DIM_TILE:.+]] = iree_tensor_ext.dispatch.tensor.load %[[DIM_SUBSPAN]]
      // CHECK: %[[DIM_I64:.+]] = tensor.extract %[[DIM_TILE]][] : tensor<i64>
      // CHECK: %[[DIM:.+]] = arith.index_cast %[[DIM_I64]] : i64 to index
      %dim_tile = iree_tensor_ext.dispatch.tensor.load %dim_tensor, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i64>> -> tensor<i64>
      %dim_i64 = tensor.extract %dim_tile[] : tensor<i64>
      %dim = arith.index_cast %dim_i64 : i64 to index

      // CHECK-DAG: %[[TIED_INPUT:.+]] = stream.binding.subspan %[[INPUT]][%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?xf32>>{%[[DIM]]}
      // CHECK-DAG: %[[TIED_OUTPUT:.+]] = stream.binding.subspan %[[OUTPUT]][%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%[[DIM]]}
      %tied_input = flow.dispatch.tie_shape %input : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?xf32>>{%dim}
      %tied_output = flow.dispatch.tie_shape %output : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%dim}

      // CHECK: %[[TILE:.+]] = iree_tensor_ext.dispatch.tensor.load %[[TIED_INPUT]]
      // CHECK: iree_tensor_ext.dispatch.tensor.store %[[TILE]], %[[TIED_OUTPUT]]
      %tile = iree_tensor_ext.dispatch.tensor.load %tied_input, offsets = [0, 0], sizes = [1, %dim], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?xf32>>{%dim} -> tensor<?xf32>
      iree_tensor_ext.dispatch.tensor.store %tile, %tied_output, offsets = [0], sizes = [%dim], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%dim}
      util.return
    }
  }
}

// -----

// CHECK-LABEL: @nested_bindings
flow.executable private @nested_bindings {
  flow.executable.export public @dispatch
  builtin.module {
    // CHECK: util.func public @dispatch(%[[DIM:.+]]: index, %[[INPUT:.+]]: !stream.binding, %[[OUTPUT:.+]]: !stream.binding)
    util.func public @dispatch(%dim: index, %input: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?xf32>>, %output: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>) {
      // CHECK-DAG: stream.dispatch.workgroup.size[0] : index
      %workgroup_size_0 = flow.dispatch.workgroup.size[0] : index
      // CHECK-DAG: stream.dispatch.workgroup.id[0] : index
      %workgroup_id_0 = flow.dispatch.workgroup.id[0] : index
      // CHECK-DAG: stream.dispatch.workgroup.count[0] : index
      %workgroup_count_0 = flow.dispatch.workgroup.count[0] : index

      // CHECK-DAG: %[[TIED_INPUT:.+]] = stream.binding.subspan %[[INPUT]][%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?xf32>>{%[[DIM]]}
      %tied_input = flow.dispatch.tie_shape %input : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?xf32>>{%dim}
      // CHECK-DAG: %[[TIED_OUTPUT:.+]] = stream.binding.subspan %[[OUTPUT]][%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%[[DIM]]}
      %tied_output = flow.dispatch.tie_shape %output : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%dim}

      %5 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_0, %workgroup_id_0]
      %6 = affine.apply affine_map<()[s0, s1] -> (s1 * s0)>()[%workgroup_size_0, %workgroup_count_0]
      scf.for %arg3 = %5 to %dim step %6 {
        %7 = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%arg3)[%workgroup_size_0, %dim]
        // CHECK: %[[TILE:.+]] = iree_tensor_ext.dispatch.tensor.load %[[TIED_INPUT]]
        // CHECK: iree_tensor_ext.dispatch.tensor.store %[[TILE]], %[[TIED_OUTPUT]]
        %tile = iree_tensor_ext.dispatch.tensor.load %tied_input, offsets = [0, %arg3], sizes = [1, %7], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?xf32>>{%dim} -> tensor<?xf32>
        iree_tensor_ext.dispatch.tensor.store %tile, %tied_output, offsets = [%arg3], sizes = [%7], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%dim}
      }
      util.return
    }
  }
}
