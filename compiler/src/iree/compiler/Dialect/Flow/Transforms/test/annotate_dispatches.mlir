// RUN: iree-opt --split-input-file --iree-flow-annotate-dispatches %s | FileCheck %s

// Dispatches containing some ops get a heuristics-driven summary in their name.
// This also tests symbol reference renaming.

flow.executable private @ex0 {
  // CHECK: flow.executable.export public @dispatch0_fill_4x8_f32
  flow.executable.export public @dispatch0
  builtin.module {
    // CHECK: func.func @dispatch0_fill_4x8_f32
    func.func @dispatch0(%arg0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x8xf32>>) {
      %cst = arith.constant 1.000000e+02 : f32
      %0 = tensor.empty() : tensor<4x8xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x8xf32>) -> tensor<4x8xf32>
      iree_tensor_ext.dispatch.tensor.store %1, %arg0, offsets = [0, 0], sizes = [4, 8], strides = [1, 1] : tensor<4x8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x8xf32>>
      return
    }
  }
}
flow.executable private @ex1 {
  // CHECK: flow.executable.export public @dispatch1_fill_8x4_f32
  flow.executable.export public @dispatch1
  builtin.module {
    // CHECK: func.func @dispatch1_fill_8x4_f32
    func.func @dispatch1(%arg0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x4xf32>>) {
      %cst = arith.constant 2.000000e+02 : f32
      %0 = tensor.empty() : tensor<8x4xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x4xf32>) -> tensor<8x4xf32>
      iree_tensor_ext.dispatch.tensor.store %1, %arg0, offsets = [0, 0], sizes = [8, 4], strides = [1, 1] : tensor<8x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x4xf32>>
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
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<10xf32>>) {
      %cst = arith.constant 1.000000e+02 : f32
      %0 = tensor.empty() : tensor<10xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>
      %2 = tensor.empty() : tensor<40xf32>
      %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<40xf32>) -> tensor<40xf32>
      %4 = tensor.empty() : tensor<20xf32>
      %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<20xf32>) -> tensor<20xf32>
      iree_tensor_ext.dispatch.tensor.store %1, %arg0, offsets = [0], sizes = [10], strides = [1] : tensor<10xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<10xf32>>
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
    func.func @dispatch(%arg0: index, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<10xf32>>) {
      %cst = arith.constant 1.000000e+02 : f32
      %0 = tensor.empty() : tensor<10xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>
      %2 = tensor.empty(%arg0, %arg0, %arg0) : tensor<?x?x?xf32>
      %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
      iree_tensor_ext.dispatch.tensor.store %1, %arg1, offsets = [0], sizes = [10], strides = [1] : tensor<10xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<10xf32>>
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
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x8xf32>>) {
      %0 = tensor.empty() : tensor<4x8xi32>
      %1 = tensor.empty() : tensor<4x8xf32>
      %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<4x8xi32>) outs(%1 : tensor<4x8xf32>) {
      ^bb0(%in: i32, %out: f32):
        %3 = arith.index_cast %in : i32 to index
        %extracted = tensor.extract %1[%3, %3] : tensor<4x8xf32>
        linalg.yield %extracted : f32
      } -> tensor<4x8xf32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg0, offsets = [0, 0], sizes = [4, 8], strides = [1, 1] : tensor<4x8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x8xf32>>
      return
    }
  }
}

// -----

// Dispatches set_encoding and unset_encoding ops get a heuristics-driven
// summary in their name.

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>
flow.executable private @ex0 {
  // CHECK: flow.executable.export public @dispatch0_map_DxD_f32
  flow.executable.export public @dispatch0
  builtin.module {
    func.func @dispatch0(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>) {
      %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
      %1 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 1 : index
      %2 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 2 : index
      %3 = iree_tensor_ext.dispatch.workload.ordinal %arg5, 3 : index
      %4 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
      %5 = flow.dispatch.tie_shape %arg1 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3}
      %6 = flow.dispatch.tie_shape %arg6 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%2, %3}
      %7 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
      %8 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %3} -> tensor<?x?xf32>
      %mapped = linalg.map { math.absf } ins(%7 : tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>)
      %9 = iree_encoding.set_encoding %mapped : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
      iree_tensor_ext.dispatch.tensor.store %9, %6, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%arg4, %arg5}
      return
    }
  }
}
flow.executable private @ex1 {
  // CHECK: flow.executable.export public @dispatch1_unset_encoding_LHS_DxD
  flow.executable.export public @dispatch1
  builtin.module {
    func.func @dispatch1(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>, %arg1: index, %arg2: index, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
      %0 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%arg1, %arg2}
      %1 = flow.dispatch.tie_shape %arg3 : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg1, %arg2}
      %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%arg1, %arg2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%arg1, %arg2} -> tensor<?x?xf32, #encoding>
      %3 = iree_encoding.unset_encoding %2 : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%arg1, %arg2}
      iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [%arg1, %arg2], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg1, %arg2}
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
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<7xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [7], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7xf32>> -> tensor<7xf32>
      %1 = tensor.empty() : tensor<7xf32>
      %2 = linalg.softmax dimension(0) ins(%0 : tensor<7xf32>) outs(%1 : tensor<7xf32>) -> tensor<7xf32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [7], strides = [1] : tensor<7xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7xf32>>
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
    func.func @ex(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
      %1 = tensor.empty() : tensor<24x512x16x1xf32>
      %pack = linalg.pack %0 inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %1 : tensor<384x512xf32> -> tensor<24x512x16x1xf32>
      iree_tensor_ext.dispatch.tensor.store %pack, %arg1, offsets = [0, 0, 0, 0], sizes = [24, 512, 16, 1], strides = [1, 1, 1, 1] : tensor<24x512x16x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>
      return
    }
  }
}

// -----

flow.executable private @ex {
  // CHECK: flow.executable.export public @ex_unpack_f32
  flow.executable.export public @ex
  builtin.module {
    func.func @ex(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x32x16x16xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x512xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [24, 32, 16, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x32x16x16xf32>> -> tensor<24x32x16x16xf32>
      %1 = tensor.empty() : tensor<384x512xf32>
      %unpack = linalg.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %1 : tensor<24x32x16x16xf32> -> tensor<384x512xf32>
      iree_tensor_ext.dispatch.tensor.store %unpack, %arg1, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : tensor<384x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<384x512xf32>>
      return
    }
  }
}

// -----

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
flow.executable private @ex {
  // CHECK: flow.executable.export public @ex_unpack_elementwise_384x512_f32_pack
  flow.executable.export public @ex
  builtin.module {
    func.func @ex(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x32x16x16xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<512xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [24, 32, 16, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x32x16x16xf32>> -> tensor<24x32x16x16xf32>
      %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [512], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512xf32>> -> tensor<512xf32>
      %2 = tensor.empty() : tensor<24x512x16x1xf32>
      %3 = tensor.empty() : tensor<384x512xf32>
      %unpack = linalg.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %3 : tensor<24x32x16x16xf32> -> tensor<384x512xf32>
      %4 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %unpack : tensor<512xf32>, tensor<384x512xf32>) outs(%3 : tensor<384x512xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %5 = arith.addf %in, %in_0 : f32
        linalg.yield %5 : f32
      } -> tensor<384x512xf32>
      %pack = linalg.pack %4 inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %2 : tensor<384x512xf32> -> tensor<24x512x16x1xf32>
      iree_tensor_ext.dispatch.tensor.store %pack, %arg2, offsets = [0, 0, 0, 0], sizes = [24, 512, 16, 1], strides = [1, 1, 1, 1] : tensor<24x512x16x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>
      return
    }
  }
}

// -----

flow.executable private @ex {
  // CHECK: flow.executable.export public @ex_slow_memcpy
  flow.executable.export public @ex
  builtin.module {
    func.func @ex(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xi32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3x9xi32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [2, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xi32>> -> tensor<2x3xi32>
      iree_tensor_ext.dispatch.tensor.store %0, %arg1, offsets = [0, 1], sizes = [2, 3], strides = [1, 1] : tensor<2x3xi32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3x9xi32>>
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
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x4xf32>>) {
      %0 = tensor.empty() : tensor<4x8xf32>
      %1 = tensor.empty() : tensor<8x4xf32>
      %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<4x8xf32>) outs(%1 : tensor<8x4xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<8x4xf32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg0, offsets = [0, 0], sizes = [8, 4], strides = [1, 1] : tensor<8x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x4xf32>>
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
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x32xf32>>) {
      %0 = tensor.empty() : tensor<1x8xf32>
      %1 = tensor.empty() : tensor<8x32xf32>
      %init = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [16, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x32xf32>> -> tensor<1x32xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
              ins(%0, %1 : tensor<1x8xf32>, tensor<8x32xf32>) outs(%init : tensor<1x32xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %3 = arith.mulf %in, %in_0 : f32
        %4 = arith.addf %out, %3 : f32
        linalg.yield %4 : f32
      } -> tensor<1x32xf32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg0, offsets = [0, 0], sizes = [1, 32], strides = [1, 1] : tensor<1x32xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x32xf32>>
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
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x1x512xf32>>) {
      %0 = tensor.empty() : tensor<1x32x64xf32>
      %1 = tensor.empty() : tensor<32x64x512xf32>
      %init = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 1, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x1x512xf32>> -> tensor<32x1x512xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
              ins(%0, %1 : tensor<1x32x64xf32>, tensor<32x64x512xf32>) outs(%init : tensor<32x1x512xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %8 = arith.mulf %in, %in_0 : f32
        %9 = arith.addf %out, %8 : f32
        linalg.yield %9 : f32
      } -> tensor<32x1x512xf32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg0, offsets = [0, 0, 0], sizes = [32, 1, 512], strides = [1, 1, 1] : tensor<32x1x512xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x1x512xf32>>
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
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x32xf32>>) {
      %0 = tensor.empty() : tensor<16x8xf32>
      %1 = tensor.empty() : tensor<8x32xf32>
      %init = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [16, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x32xf32>> -> tensor<16x32xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
              ins(%0, %1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%init : tensor<16x32xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %3 = arith.mulf %in, %in_0 : f32
        %4 = arith.addf %out, %3 : f32
        linalg.yield %4 : f32
      } -> tensor<16x32xf32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg0, offsets = [0, 0], sizes = [16, 32], strides = [1, 1] : tensor<16x32xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x32xf32>>
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
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x8x512xf32>>) {
      %0 = tensor.empty() : tensor<8x32x64xf32>
      %1 = tensor.empty() : tensor<32x64x512xf32>
      %init = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 8, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x8x512xf32>> -> tensor<32x8x512xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
              ins(%0, %1 : tensor<8x32x64xf32>, tensor<32x64x512xf32>) outs(%init : tensor<32x8x512xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %8 = arith.mulf %in, %in_0 : f32
        %9 = arith.addf %out, %8 : f32
        linalg.yield %9 : f32
      } -> tensor<32x8x512xf32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg0, offsets = [0, 0, 0], sizes = [32, 8, 512], strides = [1, 1, 1] : tensor<32x8x512xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x8x512xf32>>
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
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<512x11008xf32>>) {
      %0 = tensor.empty() : tensor<512x32x128xf32>
      %1 = tensor.empty() : tensor<11008x32x128xf32>
      %init = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [512, 11008], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<512x11008xf32>> -> tensor<512x11008xf32>
      %2 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
              ins(%0, %1 : tensor<512x32x128xf32>, tensor<11008x32x128xf32>) outs(%init : tensor<512x11008xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %24 = arith.mulf %in, %in_0 : f32
        %25 = arith.addf %24, %out : f32
        linalg.yield %25 : f32
      } -> tensor<512x11008xf32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg0, offsets = [0, 0], sizes = [512, 11008], strides = [1, 1] : tensor<512x11008xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<512x11008xf32>>
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
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x128x128x320xf16>>) {
      %0 = tensor.empty() : tensor<320x960xf16>
      %1 = tensor.empty() : tensor<960x2x128x128xf16>
      %init = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x128x128x320xf16>> -> tensor<2x128x128x320xf16>
      %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
                                            iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
              ins(%0, %1 : tensor<320x960xf16>, tensor<960x2x128x128xf16>) outs(%init : tensor<2x128x128x320xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %8 = arith.mulf %in_0, %in : f16
        %9 = arith.addf %out, %8 : f16
        linalg.yield %9 : f16
      } -> tensor<2x128x128x320xf16>
      iree_tensor_ext.dispatch.tensor.store %2, %arg0, offsets = [0, 0, 0, 0], sizes = [2, 128, 128, 320], strides = [1, 1, 1, 1] : tensor<2x128x128x320xf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x128x128x320xf16>>
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
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x3x4x2x3xf32>>) {
      %0 = tensor.empty() : tensor<2x4x5x2xf32>
      %1 = tensor.empty() : tensor<2x2x2x3xf32>
      %init = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0, 0], sizes = [2, 3, 4, 2, 3], strides = [1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x3x4x2x3xf32>> -> tensor<2x3x4x2x3xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
              ins(%0, %1 : tensor<2x4x5x2xf32>, tensor<2x2x2x3xf32>) outs(%init : tensor<2x3x4x2x3xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %3 = arith.mulf %in, %in_0 : f32
        %4 = arith.addf %out, %3 : f32
        linalg.yield %4 : f32
      } -> tensor<2x3x4x2x3xf32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg0, offsets = [0, 0, 0, 0, 0], sizes = [2, 3, 4, 2, 3], strides = [1, 1, 1, 1, 1] : tensor<2x3x4x2x3xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x3x4x2x3xf32>>
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
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8x16x32xf32>>) {
      %0 = tensor.empty() : tensor<8x16x32xf32>
      %init = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [8, 16, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8x16x32xf32>> -> tensor<8x16x32xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%0 : tensor<8x16x32xf32>) outs(%init : tensor<8x16x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        %3 = arith.maximumf %in, %out : f32
        linalg.yield %3 : f32
      } -> tensor<8x16x32xf32>
      iree_tensor_ext.dispatch.tensor.store %2, %arg0, offsets = [0, 0, 0], sizes = [8, 16, 32], strides = [1, 1, 1] : tensor<8x16x32xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8x16x32xf32>>
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
      func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x1xf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8x8x1x1x1x1xf16>>, %arg2: index, %arg3: index) {
        %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
        %1 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x1xf16>>{%0, %1}
        %3 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [1, %0, %1, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x?x1xf16>>{%0, %1} -> tensor<1x?x?x1xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 8, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8x8x1x1x1x1xf16>> -> tensor<8x8x1x1x1x1xf16>
        %5 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%3 : tensor<1x?x?x1xf16>) outs(%4 : tensor<8x8x1x1x1x1xf16>) -> tensor<8x8x1x1x1x1xf16>
        %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
          ins(%5 : tensor<8x8x1x1x1x1xf16>) outs(%4 : tensor<8x8x1x1x1x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %7 = arith.negf %in : f16
              linalg.yield %7 : f16
          } -> tensor<8x8x1x1x1x1xf16>
        iree_tensor_ext.dispatch.tensor.store %6, %arg1, offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 8, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1, 1] : tensor<8x8x1x1x1x1xf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<8x8x1x1x1x1xf16>>
        return
      }
    }
}

// -----

// Test transposing elementwise operation.

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
flow.executable private @ex {
  // CHECK: flow.executable.export public @ex_elementwise_transpose_7x5_f32
  flow.executable.export public @ex
  builtin.module {
    func.func @ex(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x7xf32>>,
                  %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<7xf32>>,
                  %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x5xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [5, 7], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x7xf32>> -> tensor<5x7xf32>
      %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [7], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7xf32>> -> tensor<7xf32>
      %2 = tensor.empty() : tensor<7x5xf32>
      %3 = linalg.generic {
        indexing_maps = [#map, #map1, #map2],
        iterator_types = ["parallel", "parallel"]
      } ins(%1, %0 : tensor<7xf32>, tensor<5x7xf32>) outs(%2 : tensor<7x5xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %5 = arith.addf %in, %in_0 : f32
        linalg.yield %5 : f32
      } -> tensor<7x5xf32>
      iree_tensor_ext.dispatch.tensor.store %3, %arg2, offsets = [0, 0], sizes = [7, 5], strides = [1, 1] : tensor<7x5xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x5xf32>>
      return
    }
  }
}

// -----

// Same as the above, but with the transpose map represented on the output.

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>
flow.executable private @ex {
  // CHECK: flow.executable.export public @ex_elementwise_transpose_5x7_f32
  flow.executable.export public @ex
  builtin.module {
    func.func @ex(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x7xf32>>,
                  %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<7xf32>>,
                  %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x5xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [5, 7], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x7xf32>> -> tensor<5x7xf32>
      %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [7], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7xf32>> -> tensor<7xf32>
      %2 = tensor.empty() : tensor<7x5xf32>
      %3 = linalg.generic {
        indexing_maps = [#map, #map1, #map2],
        iterator_types = ["parallel", "parallel"]
      } ins(%1, %0 : tensor<7xf32>, tensor<5x7xf32>) outs(%2 : tensor<7x5xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %5 = arith.addf %in, %in_0 : f32
        linalg.yield %5 : f32
      } -> tensor<7x5xf32>
      iree_tensor_ext.dispatch.tensor.store %3, %arg2, offsets = [0, 0], sizes = [7, 5], strides = [1, 1] : tensor<7x5xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x5xf32>>
      return
    }
  }
}

// -----

// Test marking a strictly broadcasting elementwise operation as a broadcast.

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
flow.executable private @ex {
  // CHECK: flow.executable.export public @ex_elementwise_broadcast_7x5_f32
  flow.executable.export public @ex
  builtin.module {
    func.func @ex(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<5xf32>>,
                  %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<5xf32>>,
                  %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x5xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [7], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5xf32>> -> tensor<5xf32>
      %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [7], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5xf32>> -> tensor<5xf32>
      %2 = tensor.empty() : tensor<7x5xf32>
      %3 = linalg.generic {
        indexing_maps = [#map, #map, #map1],
        iterator_types = ["parallel", "parallel"]
      } ins(%1, %0 : tensor<5xf32>, tensor<5xf32>) outs(%2 : tensor<7x5xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %5 = arith.addf %in, %in_0 : f32
        linalg.yield %5 : f32
      } -> tensor<7x5xf32>
      iree_tensor_ext.dispatch.tensor.store %3, %arg2, offsets = [0, 0], sizes = [7, 5], strides = [1, 1] : tensor<7x5xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x5xf32>>
      return
    }
  }
}

// -----

// Test a pure elementwise operation with a broadcasted operand.

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
flow.executable private @ex {
  // CHECK: flow.executable.export public @ex_elementwise_7x5_f32
  flow.executable.export public @ex
  builtin.module {
    func.func @ex(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x5xf32>>,
                  %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<7xf32>>,
                  %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x5xf32>>,
                  %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x5xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [7, 5], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x5xf32>> -> tensor<7x5xf32>
      %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [7], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7xf32>> -> tensor<7xf32>
      %2 = tensor.empty() : tensor<7x5xf32>
      %3:2 = linalg.generic {
        indexing_maps = [#map, #map1, #map1, #map1],
        iterator_types = ["parallel", "parallel"]
      } ins(%1, %0 : tensor<7xf32>, tensor<7x5xf32>) outs(%2, %2 : tensor<7x5xf32>, tensor<7x5xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32, %out_0: f32):
        %4 = arith.mulf %in, %in_0 : f32
        %5 = arith.addf %in, %in_0 : f32
        linalg.yield %4, %5 : f32, f32
      } -> (tensor<7x5xf32>, tensor<7x5xf32>)
      iree_tensor_ext.dispatch.tensor.store %3#0, %arg2, offsets = [0, 0], sizes = [7, 5], strides = [1, 1] : tensor<7x5xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x5xf32>>
      iree_tensor_ext.dispatch.tensor.store %3#1, %arg3, offsets = [0, 0], sizes = [7, 5], strides = [1, 1] : tensor<7x5xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x5xf32>>
      return
    }
  }
}

// -----

// Test a multi-result elementwise operation where one result is transposed.

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>
flow.executable private @ex {
  // CHECK: flow.executable.export public @ex_elementwise_transpose_7x5_f32
  flow.executable.export public @ex
  builtin.module {
    func.func @ex(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x5xf32>>,
                  %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<7xf32>>,
                  %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x5xf32>>,
                  %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x7xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [7, 5], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x5xf32>> -> tensor<7x5xf32>
      %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0], sizes = [7], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7xf32>> -> tensor<7xf32>
      %2 = tensor.empty() : tensor<7x5xf32>
      %3 = tensor.empty() : tensor<5x7xf32>
      %4:2 = linalg.generic {
        indexing_maps = [#map, #map1, #map1, #map2],
        iterator_types = ["parallel", "parallel"]
      } ins(%1, %0 : tensor<7xf32>, tensor<7x5xf32>) outs(%2, %3 : tensor<7x5xf32>, tensor<5x7xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32, %out_0: f32):
        %5 = arith.addf %in, %in_0 : f32
        linalg.yield %5, %5 : f32, f32
      } -> (tensor<7x5xf32>, tensor<5x7xf32>)
      iree_tensor_ext.dispatch.tensor.store %4#0, %arg2, offsets = [0, 0], sizes = [7, 5], strides = [1, 1] : tensor<7x5xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x5xf32>>
      iree_tensor_ext.dispatch.tensor.store %4#1, %arg3, offsets = [0, 0], sizes = [5, 7], strides = [1, 1] : tensor<5x7xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x7xf32>>
      return
    }
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_matmul_like_16xDx8_f32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x?xf32>>, %arg1: index) {
      %0 = tensor.empty() : tensor<16x8xf32>
      %1 = tensor.empty(%arg1) : tensor<8x?xf32>
      %init = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [16, %arg1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x?xf32>>{%arg1} -> tensor<16x?xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
              ins(%0, %1 : tensor<16x8xf32>, tensor<8x?xf32>) outs(%init : tensor<16x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %3 = arith.mulf %in, %in_0 : f32
        %4 = arith.addf %out, %3 : f32
        linalg.yield %4 : f32
      } -> tensor<16x?xf32>
      %3 = linalg.generic {
        indexing_maps = [#map3, #map3],
        iterator_types = ["parallel", "parallel"]
      } ins(%2 : tensor<16x?xf32>) outs(%2 : tensor<16x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %4 = math.rsqrt %in : f32
        linalg.yield %4 : f32
      } -> tensor<16x?xf32>
      iree_tensor_ext.dispatch.tensor.store %3, %arg0, offsets = [0, 0], sizes = [16, %arg1], strides = [1, 1] : tensor<16x?xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x?xf32>>{%arg1}
      return
    }
  }
}

// -----

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch_horizontal_multi_contract_28x10x4096x64x640_i8xi8xi8xi8xi32xi32xi32
  flow.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<28x4096x640xi8>>,
                        %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x64x640xi8>>,
                        %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x64x640xi8>>,
                        %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x64x640xi8>>,
                        %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<28x10x4096x64xi32>>,
                        %arg5: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<28x10x4096x64xi32>>,
                        %arg6: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<28x10x64x4096xi32>>) {
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %49 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [28, 4096, 640], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<28x4096x640xi8>> -> tensor<28x4096x640xi8>
      %50 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0], sizes = [10, 64, 640], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x64x640xi8>> -> tensor<10x64x640xi8>
      %51 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0, 0], sizes = [10, 64, 640], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x64x640xi8>> -> tensor<10x64x640xi8>
      %52 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0, 0, 0], sizes = [10, 64, 640], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<10x64x640xi8>> -> tensor<10x64x640xi8>
      %64 = tensor.empty() : tensor<28x10x64x4096xi32>
      %65 = tensor.empty() : tensor<28x10x4096x64xi32>
      %66 = linalg.fill ins(%c0_i32 : i32) outs(%65 : tensor<28x10x4096x64xi32>) -> tensor<28x10x4096x64xi32>
      %67 = linalg.fill ins(%c0_i32 : i32) outs(%64 : tensor<28x10x64x4096xi32>) -> tensor<28x10x64x4096xi32>
      %68:3 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                           affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>],
          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
          ins(%49, %50, %51, %52 : tensor<28x4096x640xi8>, tensor<10x64x640xi8>, tensor<10x64x640xi8>, tensor<10x64x640xi8>)
          outs(%66, %66, %67 : tensor<28x10x4096x64xi32>, tensor<28x10x4096x64xi32>, tensor<28x10x64x4096xi32>) {
      ^bb0(%in: i8, %in_1: i8, %in_2: i8, %in_3: i8, %out: i32, %out_4: i32, %out_5: i32):
        %72 = arith.extsi %in : i8 to i32
        %73 = arith.extsi %in_1 : i8 to i32
        %74 = arith.muli %72, %73 : i32
        %75 = arith.addi %out, %74 : i32
        %76 = arith.extsi %in_2 : i8 to i32
        %77 = arith.muli %72, %76 : i32
        %78 = arith.addi %out_4, %77 : i32
        %79 = arith.extsi %in_3 : i8 to i32
        %80 = arith.muli %72, %79 : i32
        %81 = arith.addi %out_5, %80 : i32
        linalg.yield %75, %78, %81 : i32, i32, i32
      } -> (tensor<28x10x4096x64xi32>, tensor<28x10x4096x64xi32>, tensor<28x10x64x4096xi32>)
      iree_tensor_ext.dispatch.tensor.store %68#0, %arg4, offsets = [0, 0, 0, 0], sizes = [28, 10, 4096, 64], strides = [1, 1, 1, 1] : tensor<28x10x4096x64xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<28x10x4096x64xi32>>
      iree_tensor_ext.dispatch.tensor.store %68#1, %arg5, offsets = [0, 0, 0, 0], sizes = [28, 10, 4096, 64], strides = [1, 1, 1, 1] : tensor<28x10x4096x64xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<28x10x4096x64xi32>>
      iree_tensor_ext.dispatch.tensor.store %68#2, %arg6, offsets = [0, 0, 0, 0], sizes = [28, 10, 64, 4096], strides = [1, 1, 1, 1] : tensor<28x10x64x4096xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<28x10x64x4096xi32>>
      return
    }
  }
}

// -----

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch0_reduction_4x4096_f32
  flow.executable.export public @dispatch0
  builtin.module {
    // CHECK: func.func @dispatch0_reduction_4x4096_f32
    func.func @dispatch0(
                 %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4096xf32>>,
                 %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4096xf32>> -> tensor<4x4096xf32>
      %c2_i64 = arith.constant 2 : i64
      %cst = arith.constant 1.000000e+02 : f32
      %empty = tensor.empty() : tensor<4xf32>
      %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
      %reduction = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%0 : tensor<4x4096xf32>) outs(%fill : tensor<4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %20 = math.fpowi %in, %c2_i64 : f32, i64
        %21 = arith.addf %20, %out : f32
        linalg.yield %21 : f32
      } -> tensor<4xf32>
      iree_tensor_ext.dispatch.tensor.store %reduction, %arg1, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
      return
    }
  }
}

// -----

flow.executable private @ex {
  // CHECK: flow.executable.export public @dispatch0_reduction_4x4096_f32
  flow.executable.export public @dispatch0
  builtin.module {
    // CHECK: func.func @dispatch0_reduction_4x4096_f32
    func.func @dispatch0(
             %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4096xf32>>,
             %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4096xf32>>) {
      %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4096xf32>> -> tensor<4x4096xf32>
      %c2_i64 = arith.constant 2 : i64
      %cst = arith.constant 1.000000e+02 : f32
      %empty = tensor.empty() : tensor<4xf32>
      %empty0 = tensor.empty() : tensor<4x4096xf32>
      %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
      %reduction = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%0 : tensor<4x4096xf32>) outs(%fill : tensor<4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %20 = math.fpowi %in, %c2_i64 : f32, i64
        %21 = arith.addf %20, %out : f32
        linalg.yield %21 : f32
      } -> tensor<4xf32>
      %elem = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %reduction : tensor<4x4096xf32>, tensor<4xf32>) outs(%empty0 : tensor<4x4096xf32>) {
      ^bb0(%in: f32, %in0 : f32, %out: f32):
        %21 = arith.addf %in, %in0 : f32
        linalg.yield %21 : f32
      } -> tensor<4x4096xf32>
      iree_tensor_ext.dispatch.tensor.store %elem, %arg1, offsets = [0, 0], sizes = [4, 4096], strides = [1, 1] : tensor<4x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4096xf32>>
      return
    }
  }
}
