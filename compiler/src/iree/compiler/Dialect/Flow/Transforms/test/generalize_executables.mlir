// RUN: iree-opt --split-input-file --iree-flow-generalize-executables %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-flow-generalize-executables, iree-flow-deduplicate-executables)" %s | FileCheck %s --check-prefix=DEDUP

// Two executables that read a global at different static offsets: the offset
// becomes a dispatch operand (inserted after the operands, before the result
// binding) and each call site passes its value; deduplication then keeps one.

flow.executable private @offset_ex_0 {
  flow.executable.export public @offset_entry_0 workgroups() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    flow.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @offset_entry_0(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [2, 0], sizes = [1, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>> -> tensor<1x4xf32>
      iree_tensor_ext.dispatch.tensor.store %0, %arg1, offsets = [0, 0], sizes = [1, 4], strides = [1, 1] : tensor<1x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4xf32>>
      return
    }
  }
}
flow.executable private @offset_ex_1 {
  flow.executable.export public @offset_entry_1 workgroups() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    flow.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @offset_entry_1(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [6, 0], sizes = [1, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>> -> tensor<1x4xf32>
      iree_tensor_ext.dispatch.tensor.store %0, %arg1, offsets = [0, 0], sizes = [1, 4], strides = [1, 1] : tensor<1x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4xf32>>
      return
    }
  }
}
util.global private @cache : tensor<8x4xf32>
util.func public @offset() -> (tensor<1x4xf32>, tensor<1x4xf32>) {
  %cache = util.global.load @cache : tensor<8x4xf32>
  %0 = flow.dispatch @offset_ex_0::@offset_entry_0(%cache) : (tensor<8x4xf32>) -> tensor<1x4xf32>
  %1 = flow.dispatch @offset_ex_1::@offset_entry_1(%cache) : (tensor<8x4xf32>) -> tensor<1x4xf32>
  util.return %0, %1 : tensor<1x4xf32>, tensor<1x4xf32>
}

// CHECK-LABEL: flow.executable private @offset_ex_0
//       CHECK:   func.func @offset_entry_0(%[[SRC:.+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>>, %[[OFF:.+]]: index, %{{.+}}: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4xf32>>)
//       CHECK:     iree_tensor_ext.dispatch.tensor.load %[[SRC]], offsets = [%[[OFF]], 0]
// CHECK-LABEL: flow.executable private @offset_ex_1
//       CHECK:   func.func @offset_entry_1(%{{.+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>>, %{{.+}}: index,
// CHECK-LABEL: util.func public @offset
//       CHECK:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   flow.dispatch @offset_ex_0::@offset_entry_0(%{{.+}}, %[[C2]]) : (tensor<8x4xf32>, index) -> tensor<1x4xf32>
//       CHECK:   %[[C6:.+]] = arith.constant 6 : index
//       CHECK:   flow.dispatch @offset_ex_1::@offset_entry_1(%{{.+}}, %[[C6]]) : (tensor<8x4xf32>, index) -> tensor<1x4xf32>

// DEDUP-LABEL: flow.executable private @offset_ex_0
//   DEDUP-NOT: flow.executable private @offset_ex_1
//       DEDUP: flow.dispatch @offset_ex_0::@offset_entry_0
//       DEDUP: flow.dispatch @offset_ex_0::@offset_entry_0

// -----

// Two executables that differ in an f32 constant (e.g. a per-layer scale): the
// constant becomes a dispatch operand.

flow.executable private @scale_ex_0 {
  flow.executable.export public @scale_entry_0 workgroups() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    flow.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @scale_entry_0(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>) {
      %cst = arith.constant 5.000000e-01 : f32
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
      %1 = tensor.empty() : tensor<4xf32>
      %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<4xf32>) outs(%1 : tensor<4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %3 = arith.mulf %in, %cst : f32
        linalg.yield %3 : f32
      } -> tensor<4xf32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
      return
    }
  }
}
flow.executable private @scale_ex_1 {
  flow.executable.export public @scale_entry_1 workgroups() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    flow.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @scale_entry_1(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>) {
      %cst = arith.constant 2.500000e-01 : f32
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
      %1 = tensor.empty() : tensor<4xf32>
      %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<4xf32>) outs(%1 : tensor<4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %3 = arith.mulf %in, %cst : f32
        linalg.yield %3 : f32
      } -> tensor<4xf32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
      return
    }
  }
}
util.func public @scale(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = flow.dispatch @scale_ex_0::@scale_entry_0(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = flow.dispatch @scale_ex_1::@scale_entry_1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  util.return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: flow.executable private @scale_ex_0
//       CHECK:   func.func @scale_entry_0(%{{.+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>, %[[S:.+]]: f32,
//   CHECK-NOT:     arith.constant
//       CHECK:     arith.mulf %{{.+}}, %[[S]] : f32
// CHECK-LABEL: util.func public @scale
//       CHECK:   %[[A:.+]] = arith.constant 5.000000e-01 : f32
//       CHECK:   flow.dispatch @scale_ex_0::@scale_entry_0(%arg0, %[[A]]) : (tensor<4xf32>, f32) -> tensor<4xf32>
//       CHECK:   %[[B:.+]] = arith.constant 2.500000e-01 : f32
//       CHECK:   flow.dispatch @scale_ex_1::@scale_entry_1(%arg0, %[[B]]) : (tensor<4xf32>, f32) -> tensor<4xf32>

// DEDUP-LABEL: flow.executable private @scale_ex_0
//   DEDUP-NOT: flow.executable private @scale_ex_1

// -----

// An offset into a tensor that is not a global's contents is left alone: its
// binding offset may itself be dynamic, so the group is not generalized.

flow.executable private @transient_ex_0 {
  flow.executable.export public @transient_entry_0 workgroups() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    flow.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @transient_entry_0(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [2, 0], sizes = [1, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>> -> tensor<1x4xf32>
      iree_tensor_ext.dispatch.tensor.store %0, %arg1, offsets = [0, 0], sizes = [1, 4], strides = [1, 1] : tensor<1x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4xf32>>
      return
    }
  }
}
flow.executable private @transient_ex_1 {
  flow.executable.export public @transient_entry_1 workgroups() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    flow.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @transient_entry_1(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [6, 0], sizes = [1, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>> -> tensor<1x4xf32>
      iree_tensor_ext.dispatch.tensor.store %0, %arg1, offsets = [0, 0], sizes = [1, 4], strides = [1, 1] : tensor<1x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4xf32>>
      return
    }
  }
}
util.func public @transient(%arg0: tensor<8x4xf32>) -> (tensor<1x4xf32>, tensor<1x4xf32>) {
  %0 = flow.dispatch @transient_ex_0::@transient_entry_0(%arg0) : (tensor<8x4xf32>) -> tensor<1x4xf32>
  %1 = flow.dispatch @transient_ex_1::@transient_entry_1(%arg0) : (tensor<8x4xf32>) -> tensor<1x4xf32>
  util.return %0, %1 : tensor<1x4xf32>, tensor<1x4xf32>
}

// CHECK-LABEL: flow.executable private @transient_ex_0
//       CHECK:   func.func @transient_entry_0(%{{.+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4xf32>>, %{{.+}}: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4xf32>>)
//       CHECK:     offsets = [2, 0]
// CHECK-LABEL: util.func public @transient
//       CHECK:   flow.dispatch @transient_ex_0::@transient_entry_0(%arg0) : (tensor<8x4xf32>) -> tensor<1x4xf32>

// DEDUP-LABEL: flow.executable private @transient_ex_0
//       DEDUP: flow.executable private @transient_ex_1

// -----

// Integer constants other than index values compute values a backend may need
// to see as constants: executables differing in one are not generalized.

flow.executable private @int_ex_0 {
  flow.executable.export public @int_entry_0 workgroups() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    flow.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @int_entry_0(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi32>>) {
      %c3 = arith.constant 3 : i32
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi32>> -> tensor<4xi32>
      %1 = tensor.empty() : tensor<4xi32>
      %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<4xi32>) outs(%1 : tensor<4xi32>) {
      ^bb0(%in: i32, %out: i32):
        %3 = arith.addi %in, %c3 : i32
        linalg.yield %3 : i32
      } -> tensor<4xi32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi32>>
      return
    }
  }
}
flow.executable private @int_ex_1 {
  flow.executable.export public @int_entry_1 workgroups() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    flow.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @int_entry_1(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi32>>) {
      %c5 = arith.constant 5 : i32
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xi32>> -> tensor<4xi32>
      %1 = tensor.empty() : tensor<4xi32>
      %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<4xi32>) outs(%1 : tensor<4xi32>) {
      ^bb0(%in: i32, %out: i32):
        %3 = arith.addi %in, %c5 : i32
        linalg.yield %3 : i32
      } -> tensor<4xi32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [4], strides = [1] : tensor<4xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xi32>>
      return
    }
  }
}
util.func public @int(%arg0: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
  %0 = flow.dispatch @int_ex_0::@int_entry_0(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  %1 = flow.dispatch @int_ex_1::@int_entry_1(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  util.return %0, %1 : tensor<4xi32>, tensor<4xi32>
}

// CHECK-LABEL: flow.executable private @int_ex_0
//       CHECK:   arith.constant 3 : i32
// CHECK-LABEL: util.func public @int
//       CHECK:   flow.dispatch @int_ex_0::@int_entry_0(%arg0) : (tensor<4xi32>) -> tensor<4xi32>

// DEDUP-LABEL: flow.executable private @int_ex_0
//       DEDUP: flow.executable private @int_ex_1
