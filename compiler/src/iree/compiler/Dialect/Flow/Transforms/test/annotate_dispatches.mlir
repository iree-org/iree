// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-flow-annotate-dispatches %s | FileCheck %s

// Dispatches containing some ops get a heuristics-driven summary in their name.
// This also tests symbol reference renaming.

flow.executable private @ex0 {
  // CHECK: flow.executable.export public @dispatch0_fill_4x8_f32
  flow.executable.export public @dispatch0
  builtin.module {
    // CHECK: func.func @dispatch0_fill_4x8_f32
    func.func @dispatch0(%arg0: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>) {
      %cst = arith.constant 1.000000e+02 : f32
      %0 = tensor.empty() : tensor<4x8xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x8xf32>) -> tensor<4x8xf32>
      flow.dispatch.tensor.store %1, %arg0, offsets = [0, 0], sizes = [4, 8], strides = [1, 1] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
      return
    }
  }
}
flow.executable private @ex1 {
  // CHECK: flow.executable.export public @dispatch1_fill_8x4_f32
  flow.executable.export public @dispatch1
  builtin.module {
    // CHECK: func.func @dispatch1_fill_8x4_f32
    func.func @dispatch1(%arg0: !flow.dispatch.tensor<writeonly:tensor<8x4xf32>>) {
      %cst = arith.constant 2.000000e+02 : f32
      %0 = tensor.empty() : tensor<8x4xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x4xf32>) -> tensor<8x4xf32>
      flow.dispatch.tensor.store %1, %arg0, offsets = [0, 0], sizes = [8, 4], strides = [1, 1] : tensor<8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x4xf32>>
      return
    }
  }
}
func.func @main() -> (tensor<4x8xf32>, tensor<8x4xf32>) {
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  // CHECK: flow.dispatch @ex0::@dispatch0_fill_4x8_f32
  %0 = flow.dispatch @ex0::@dispatch0[%c100, %c50]() : () -> tensor<4x8xf32>
  // CHECK: flow.dispatch @ex1::@dispatch1_fill_8x4_f32
  %1 = flow.dispatch @ex1::@dispatch1[%c100, %c50]() : () -> tensor<8x4xf32>
  return %0, %1 : tensor<4x8xf32>, tensor<8x4xf32>
}

// -----

// A cost model picks the "most expensive" op to include in the summary.

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_fill_40_f32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !flow.dispatch.tensor<writeonly:tensor<10xf32>>) {
      %cst = arith.constant 1.000000e+02 : f32
      %0 = tensor.empty() : tensor<10xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>
      %2 = tensor.empty() : tensor<40xf32>
      %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<40xf32>) -> tensor<40xf32>
      %4 = tensor.empty() : tensor<20xf32>
      %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<20xf32>) -> tensor<20xf32>
      flow.dispatch.tensor.store %1, %arg0, offsets = [0], sizes = [10], strides = [1] : tensor<10xf32> -> !flow.dispatch.tensor<writeonly:tensor<10xf32>>
      return
    }
  }
}

// -----

// Dynamic dimensions are considered the most expensive.

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_fill_DxDxD_f32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: index, %arg1: !flow.dispatch.tensor<writeonly:tensor<10xf32>>) {
      %cst = arith.constant 1.000000e+02 : f32
      %0 = tensor.empty() : tensor<10xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>
      %2 = tensor.empty(%arg0, %arg0, %arg0) : tensor<?x?x?xf32>
      %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      flow.dispatch.tensor.store %1, %arg1, offsets = [0], sizes = [10], strides = [1] : tensor<10xf32> -> !flow.dispatch.tensor<writeonly:tensor<10xf32>>
      return
    }
  }
}

// -----

// Dispatch key op with multiple datatypes should be reflected in summary.

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_generic_4x8_i32xf32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>) {
      %0 = tensor.empty() : tensor<4x8xi32>
      %1 = tensor.empty() : tensor<4x8xf32>
      %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<4x8xi32>) outs(%1 : tensor<4x8xf32>) {
      ^bb0(%in: i32, %out: f32):
        %3 = arith.index_cast %in : i32 to index
        %extracted = tensor.extract %1[%3, %3] : tensor<4x8xf32>
        linalg.yield %extracted : f32
      } -> tensor<4x8xf32>
      flow.dispatch.tensor.store %2, %arg0, offsets = [0, 0], sizes = [4, 8], strides = [1, 1] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
      return
    }
  }
}

// -----

// Dispatches set_encoding and unset_encoding ops get a heuristics-driven
// summary in their name.

flow.executable private @ex0 {
  // CHECK: flow.executable.export public @dispatch0_map_DxD_f32
  flow.executable.export public @dispatch0
  builtin.module {
    func.func @dispatch0(%arg0: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>>) {
      %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
      %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
      %2 = flow.dispatch.workload.ordinal %arg4, 2 : index
      %3 = flow.dispatch.workload.ordinal %arg5, 3 : index
      %4 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
      %5 = flow.dispatch.tie_shape %arg1 : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
      %6 = flow.dispatch.tie_shape %arg6 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>>{%2, %3}
      %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
      %8 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
      %mapped = linalg.map { math.absf } ins(%7 : tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>)
      %9 = iree_linalg_ext.set_encoding %mapped : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>
      flow.dispatch.tensor.store %9, %6, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>>{%arg4, %arg5}
      return
    }
  }
}
flow.executable private @ex1 {
  // CHECK: flow.executable.export public @dispatch1_unset_encoding_MATMUL_LHS_DxD
  flow.executable.export public @dispatch1
  builtin.module {
    func.func @dispatch1(%arg0: !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>>, %arg1: index, %arg2: index, %arg3: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
      %0 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>>{%arg1, %arg2}
      %1 = flow.dispatch.tie_shape %arg3 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg1, %arg2}
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%arg1, %arg2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>>{%arg1, %arg2} -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>
      %3 = iree_linalg_ext.unset_encoding %2 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>> -> tensor<?x?xf32>
      flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [%arg1, %arg2], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg1, %arg2}
      return
    }
  }
}

// -----

// Named root linalg ops get represented in the dispatch name.

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_softmax_7xf32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !flow.dispatch.tensor<readonly:tensor<7xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<7xf32>>) {
      %0 = flow.dispatch.tensor.load %arg0, offsets = [0], sizes = [7], strides = [1] : !flow.dispatch.tensor<readonly:tensor<7xf32>> -> tensor<7xf32>
      %1 = tensor.empty() : tensor<7xf32>
      %2 = linalg.softmax dimension(0) ins(%0 : tensor<7xf32>) outs(%1 : tensor<7xf32>) -> tensor<7xf32>
      flow.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [7], strides = [1] : tensor<7xf32> -> !flow.dispatch.tensor<writeonly:tensor<7xf32>>
      return
    }
  }
}

// -----

// Executables with no contents are ignored.

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch
  flow.executable.export public @dispatch
}
