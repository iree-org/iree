// RUN: iree-opt --split-input-file --iree-flow-annotate-dispatches %s | FileCheck %s

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
util.func public @main() -> (tensor<4x8xf32>, tensor<8x4xf32>) {
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  // CHECK: flow.dispatch @ex0::@dispatch0_fill_4x8_f32
  %0 = flow.dispatch @ex0::@dispatch0[%c100, %c50]() : () -> tensor<4x8xf32>
  // CHECK: flow.dispatch @ex1::@dispatch1_fill_8x4_f32
  %1 = flow.dispatch @ex1::@dispatch1[%c100, %c50]() : () -> tensor<8x4xf32>
  util.return %0, %1 : tensor<4x8xf32>, tensor<8x4xf32>
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
  // CHECK: flow.executable.export public @dispatch_elementwise_4x8_i32xf32
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
    func.func @dispatch0(%arg0: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>>) {
      %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
      %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
      %2 = flow.dispatch.workload.ordinal %arg4, 2 : index
      %3 = flow.dispatch.workload.ordinal %arg5, 3 : index
      %4 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
      %5 = flow.dispatch.tie_shape %arg1 : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
      %6 = flow.dispatch.tie_shape %arg6 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>>{%2, %3}
      %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
      %8 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
      %mapped = linalg.map { math.absf } ins(%7 : tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>)
      %9 = iree_encoding.set_encoding %mapped : tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>
      flow.dispatch.tensor.store %9, %6, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>>{%arg4, %arg5}
      return
    }
  }
}
flow.executable private @ex1 {
  // CHECK: flow.executable.export public @dispatch1_unset_encoding_LHS_DxD
  flow.executable.export public @dispatch1
  builtin.module {
    func.func @dispatch1(%arg0: !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>>, %arg1: index, %arg2: index, %arg3: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
      %0 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>>{%arg1, %arg2}
      %1 = flow.dispatch.tie_shape %arg3 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg1, %arg2}
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%arg1, %arg2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>>{%arg1, %arg2} -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>
      %3 = iree_encoding.unset_encoding %2 : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
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

// -----

flow.executable private @ex {
  // CHECK: flow.executable.export public @ex_pack_f32
  flow.executable.export public @ex
  builtin.module {
    func.func @ex(%arg0: !flow.dispatch.tensor<readonly:tensor<384x512xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>) {
      %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
      %1 = tensor.empty() : tensor<24x512x16x1xf32>
      %pack = tensor.pack %0 inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %1 : tensor<384x512xf32> -> tensor<24x512x16x1xf32>
      flow.dispatch.tensor.store %pack, %arg1, offsets = [0, 0, 0, 0], sizes = [24, 512, 16, 1], strides = [1, 1, 1, 1] : tensor<24x512x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>
      return
    }
  }
}

// -----

flow.executable private @ex {
  // CHECK: flow.executable.export public @ex_unpack_f32
  flow.executable.export public @ex
  builtin.module {
    func.func @ex(%arg0: !flow.dispatch.tensor<readonly:tensor<24x32x16x16xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>) {
      %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [24, 32, 16, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<24x32x16x16xf32>> -> tensor<24x32x16x16xf32>
      %1 = tensor.empty() : tensor<384x512xf32>
      %unpack = tensor.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %1 : tensor<24x32x16x16xf32> -> tensor<384x512xf32>
      flow.dispatch.tensor.store %unpack, %arg1, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : tensor<384x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<384x512xf32>>
      return
    }
  }
}

// -----

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
flow.executable private @ex {
  // CHECK: flow.executable.export public @ex_unpack_broadcast_384x512_f32_pack
  flow.executable.export public @ex
  builtin.module {
    func.func @ex(%arg0: !flow.dispatch.tensor<readonly:tensor<24x32x16x16xf32>>, %arg1: !flow.dispatch.tensor<readonly:tensor<512xf32>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>) {
      %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [24, 32, 16, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<24x32x16x16xf32>> -> tensor<24x32x16x16xf32>
      %1 = flow.dispatch.tensor.load %arg1, offsets = [0], sizes = [512], strides = [1] : !flow.dispatch.tensor<readonly:tensor<512xf32>> -> tensor<512xf32>
      %2 = tensor.empty() : tensor<24x512x16x1xf32>
      %3 = tensor.empty() : tensor<384x512xf32>
      %unpack = tensor.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %3 : tensor<24x32x16x16xf32> -> tensor<384x512xf32>
      %4 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %unpack : tensor<512xf32>, tensor<384x512xf32>) outs(%3 : tensor<384x512xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %5 = arith.addf %in, %in_0 : f32
        linalg.yield %5 : f32
      } -> tensor<384x512xf32>
      %pack = tensor.pack %4 inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %2 : tensor<384x512xf32> -> tensor<24x512x16x1xf32>
      flow.dispatch.tensor.store %pack, %arg2, offsets = [0, 0, 0, 0], sizes = [24, 512, 16, 1], strides = [1, 1, 1, 1] : tensor<24x512x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>
      return
    }
  }
}

// -----

flow.executable private @ex {
  // CHECK: flow.executable.export public @ex_slow_memcpy
  flow.executable.export public @ex
  builtin.module {
    func.func @ex(%arg0: !flow.dispatch.tensor<readonly:tensor<2x3xi32>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<3x9xi32>>) {
      %0 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [2, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x3xi32>> -> tensor<2x3xi32>
      flow.dispatch.tensor.store %0, %arg1, offsets = [0, 1], sizes = [2, 3], strides = [1, 1] : tensor<2x3xi32> -> !flow.dispatch.tensor<readwrite:tensor<3x9xi32>>
      return
    }
  }
}

// -----

// Dispatch with only a yield and having indexing_maps only as permutations are transposes.

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_transpose_8x4_f32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !flow.dispatch.tensor<writeonly:tensor<8x4xf32>>) {
      %0 = tensor.empty() : tensor<4x8xf32>
      %1 = tensor.empty() : tensor<8x4xf32>
      %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<4x8xf32>) outs(%1 : tensor<8x4xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<8x4xf32>
      flow.dispatch.tensor.store %2, %arg0, offsets = [0, 0], sizes = [8, 4], strides = [1, 1] : tensor<8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x4xf32>>
      return
    }
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_matvec_like_1x32x8_f32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !flow.dispatch.tensor<readwrite:tensor<1x32xf32>>) {
      %0 = tensor.empty() : tensor<1x8xf32>
      %1 = tensor.empty() : tensor<8x32xf32>
      %init = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [16, 32], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<1x32xf32>> -> tensor<1x32xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
              ins(%0, %1 : tensor<1x8xf32>, tensor<8x32xf32>) outs(%init : tensor<1x32xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %3 = arith.mulf %in, %in_0 : f32
        %4 = arith.addf %out, %3 : f32
        linalg.yield %4 : f32
      } -> tensor<1x32xf32>
      flow.dispatch.tensor.store %2, %arg0, offsets = [0, 0], sizes = [1, 32], strides = [1, 1] : tensor<1x32xf32> -> !flow.dispatch.tensor<readwrite:tensor<1x32xf32>>
      return
    }
  }
}

// -----

#map =  affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// Batch matvec-like generic
flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_matvec_like_32x1x512x64_f32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !flow.dispatch.tensor<readwrite:tensor<32x1x512xf32>>) {
      %0 = tensor.empty() : tensor<1x32x64xf32>
      %1 = tensor.empty() : tensor<32x64x512xf32>
      %init = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 1, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<32x1x512xf32>> -> tensor<32x1x512xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
              ins(%0, %1 : tensor<1x32x64xf32>, tensor<32x64x512xf32>) outs(%init : tensor<32x1x512xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %8 = arith.mulf %in, %in_0 : f32
        %9 = arith.addf %out, %8 : f32
        linalg.yield %9 : f32
      } -> tensor<32x1x512xf32>
      flow.dispatch.tensor.store %2, %arg0, offsets = [0, 0, 0], sizes = [32, 1, 512], strides = [1, 1, 1] : tensor<32x1x512xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x1x512xf32>>
      return
    }
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_matmul_like_16x32x8_f32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !flow.dispatch.tensor<readwrite:tensor<16x32xf32>>) {
      %0 = tensor.empty() : tensor<16x8xf32>
      %1 = tensor.empty() : tensor<8x32xf32>
      %init = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [16, 32], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<16x32xf32>> -> tensor<16x32xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
              ins(%0, %1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%init : tensor<16x32xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %3 = arith.mulf %in, %in_0 : f32
        %4 = arith.addf %out, %3 : f32
        linalg.yield %4 : f32
      } -> tensor<16x32xf32>
      flow.dispatch.tensor.store %2, %arg0, offsets = [0, 0], sizes = [16, 32], strides = [1, 1] : tensor<16x32xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x32xf32>>
      return
    }
  }
}

// -----

#map =  affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// batch matmul-like generic
flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_matmul_like_32x8x512x64_f32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !flow.dispatch.tensor<readwrite:tensor<32x8x512xf32>>) {
      %0 = tensor.empty() : tensor<8x32x64xf32>
      %1 = tensor.empty() : tensor<32x64x512xf32>
      %init = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 8, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<32x8x512xf32>> -> tensor<32x8x512xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
              ins(%0, %1 : tensor<8x32x64xf32>, tensor<32x64x512xf32>) outs(%init : tensor<32x8x512xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %8 = arith.mulf %in, %in_0 : f32
        %9 = arith.addf %out, %8 : f32
        linalg.yield %9 : f32
      } -> tensor<32x8x512xf32>
      flow.dispatch.tensor.store %2, %arg0, offsets = [0, 0, 0], sizes = [32, 8, 512], strides = [1, 1, 1] : tensor<32x8x512xf32> -> !flow.dispatch.tensor<readwrite:tensor<32x8x512xf32>>
      return
    }
  }
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// Multi-reduction matmul-like generic
flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_matmul_like_512x11008x32x128_f32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !flow.dispatch.tensor<readwrite:tensor<512x11008xf32>>) {
      %0 = tensor.empty() : tensor<512x32x128xf32>
      %1 = tensor.empty() : tensor<11008x32x128xf32>
      %init = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [512, 11008], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<512x11008xf32>> -> tensor<512x11008xf32>
      %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
              ins(%0, %1 : tensor<512x32x128xf32>, tensor<11008x32x128xf32>) outs(%init : tensor<512x11008xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %24 = arith.mulf %in, %in_0 : f32
        %25 = arith.addf %24, %out : f32
        linalg.yield %25 : f32
      } -> tensor<512x11008xf32>
      flow.dispatch.tensor.store %2, %arg0, offsets = [0, 0], sizes = [512, 11008], strides = [1, 1] : tensor<512x11008xf32> -> !flow.dispatch.tensor<readwrite:tensor<512x11008xf32>>
      return
    }
  }
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// Multi-parallel matmul-like generic
flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_matmul_like_2x128x128x320x960_f16
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !flow.dispatch.tensor<readwrite:tensor<2x128x128x320xf16>>) {
      %0 = tensor.empty() : tensor<320x960xf16>
      %1 = tensor.empty() : tensor<960x2x128x128xf16>
      %init = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<2x128x128x320xf16>> -> tensor<2x128x128x320xf16>
      %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
                                            iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
              ins(%0, %1 : tensor<320x960xf16>, tensor<960x2x128x128xf16>) outs(%init : tensor<2x128x128x320xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %8 = arith.mulf %in_0, %in : f16
        %9 = arith.addf %out, %8 : f16
        linalg.yield %9 : f16
      } -> tensor<2x128x128x320xf16>
      flow.dispatch.tensor.store %2, %arg0, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : tensor<2x128x128x320xf16> -> !flow.dispatch.tensor<readwrite:tensor<2x128x128x320xf16>>
      return
    }
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_conv_2x3x4x2x3x2x2_f32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !flow.dispatch.tensor<readwrite:tensor<2x3x4x2x3xf32>>) {
      %0 = tensor.empty() : tensor<2x4x5x2xf32>
      %1 = tensor.empty() : tensor<2x2x2x3xf32>
      %init = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0, 0], sizes = [2, 3, 4, 2, 3], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<2x3x4x2x3xf32>> -> tensor<2x3x4x2x3xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
              ins(%0, %1 : tensor<2x4x5x2xf32>, tensor<2x2x2x3xf32>) outs(%init : tensor<2x3x4x2x3xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %3 = arith.mulf %in, %in_0 : f32
        %4 = arith.addf %out, %3 : f32
        linalg.yield %4 : f32
      } -> tensor<2x3x4x2x3xf32>
      flow.dispatch.tensor.store %2, %arg0, offsets = [0, 0, 0, 0, 0], sizes = [2, 3, 4, 2, 3], strides = [1, 1, 1, 1, 1] : tensor<2x3x4x2x3xf32> -> !flow.dispatch.tensor<readwrite:tensor<2x3x4x2x3xf32>>
      return
    }
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_elementwise_8x16x32_f32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !flow.dispatch.tensor<readwrite:tensor<8x16x32xf32>>) {
      %0 = tensor.empty() : tensor<8x16x32xf32>
      %init = flow.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [8, 16, 32], strides = [1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<8x16x32xf32>> -> tensor<8x16x32xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%0 : tensor<8x16x32xf32>) outs(%init : tensor<8x16x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        %3 = arith.maximumf %in, %out : f32
        linalg.yield %3 : f32
      } -> tensor<8x16x32xf32>
      flow.dispatch.tensor.store %2, %arg0, offsets = [0, 0, 0], sizes = [8, 16, 32], strides = [1, 1, 1] : tensor<8x16x32xf32> -> !flow.dispatch.tensor<readwrite:tensor<8x16x32xf32>>
      return
    }
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d4, d3, d5)>
flow.executable private @ex {
  // CHECK: flow.executable.export private @dispatch_winograd_input_transform_1xDxDx1xf16_generic
  flow.executable.export private @dispatch
    builtin.module {
      func.func @dispatch(%arg0: !flow.dispatch.tensor<readonly:tensor<1x?x?x1xf16>>, %arg1: !flow.dispatch.tensor<readwrite:tensor<8x8x1x1x1x1xf16>>, %arg2: index, %arg3: index) {
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<1x?x?x1xf16>>{%0, %1}
        %3 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [1, %0, %1, 1], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x?x?x1xf16>>{%0, %1} -> tensor<1x?x?x1xf16>
        %4 = flow.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 8, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<8x8x1x1x1x1xf16>> -> tensor<8x8x1x1x1x1xf16>
        %5 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%3 : tensor<1x?x?x1xf16>) outs(%4 : tensor<8x8x1x1x1x1xf16>) -> tensor<8x8x1x1x1x1xf16>
        %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
          ins(%5 : tensor<8x8x1x1x1x1xf16>) outs(%4 : tensor<8x8x1x1x1x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %7 = arith.negf %in : f16
              linalg.yield %7 : f16
          } -> tensor<8x8x1x1x1x1xf16>
        flow.dispatch.tensor.store %6, %arg1, offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 8, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1, 1] : tensor<8x8x1x1x1x1xf16> -> !flow.dispatch.tensor<readwrite:tensor<8x8x1x1x1x1xf16>>
        return
      }
    }
}
