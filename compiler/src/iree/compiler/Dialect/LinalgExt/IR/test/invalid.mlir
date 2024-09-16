// RUN: iree-opt --split-input-file --verify-diagnostics %s

func.func @sort_invalid_dimension(%arg0: tensor<128xi32>) -> tensor<128xi32> {
  // expected-error @+1 {{dimension must be within (0, 1]}}
  %0 = iree_linalg_ext.sort dimension(1)
    outs(%arg0 : tensor<128xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
    %1 = arith.cmpi sgt, %arg1, %arg2 : i32
    iree_linalg_ext.yield %1 : i1
  } -> tensor<128xi32>
  return %0 : tensor<128xi32>
}

// -----

func.func @sort_mismatch_rank(%arg0: tensor<?x?xi32>, %arg1: tensor<?xf32>)
    -> (tensor<?x?xi32>, tensor<?xf32>) {
  // expected-error @+1 {{expected operand 1 to be rank 2, same as other operands}}
  %0:2 = iree_linalg_ext.sort dimension(0)
      outs(%arg0, %arg1 : tensor<?x?xi32>, tensor<?xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %1 = arith.cmpf ogt, %arg4, %arg5 : f32
        iree_linalg_ext.yield %1 : i1
      } -> tensor<?x?xi32>, tensor<?xf32>
  return %0#0, %0#1 : tensor<?x?xi32>, tensor<?xf32>
}

// -----

func.func @sort_mismatch_shape(%arg0: tensor<?xi32>, %arg1: tensor<42xf32>)
    -> (tensor<?xi32>, tensor<42xf32>) {
  // expected-error @+1 {{expected operand 1 to have same shape as other operands}}
  %0:2 = iree_linalg_ext.sort dimension(0)
      outs(%arg0, %arg1 : tensor<?xi32>, tensor<42xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %1 = arith.cmpf ogt, %arg4, %arg5 : f32
        iree_linalg_ext.yield %1 : i1
      } -> tensor<?xi32>, tensor<42xf32>
  return %0#0, %0#1 : tensor<?xi32>, tensor<42xf32>
}

// -----

func.func @scatter_extra_outputs(
    %update : tensor<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // expected-error @+1 {{expected the number of tensor results (2) to be equal to the number of output tensors (1)}}
  %0, %1 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
      ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        iree_linalg_ext.yield %1 : f32
      } -> tensor<?x?xf32>, tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

func.func @scatter_mistmatch_dim_map_entries(
    %update : tensor<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{invalid number of dimension map entries}}
  %0 = iree_linalg_ext.scatter dimension_map = [0, 1] unique_indices(true)
      ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        iree_linalg_ext.yield %1 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @scatter_duplicate_dim_map_entries(
    %update : tensor<?x?xf32>, %indices : tensor<?x2xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{dimension map is invalid}}
  %0 = iree_linalg_ext.scatter dimension_map = [1, 1] unique_indices(true)
      ins(%update, %indices : tensor<?x?xf32>, tensor<?x2xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        iree_linalg_ext.yield %1 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @scatter_invalid_dim_map_entries(
    %update : tensor<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{dimension map is invalid}}
  %0 = iree_linalg_ext.scatter dimension_map = [2] unique_indices(true)
      ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        iree_linalg_ext.yield %1 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @scatter_output_type_mismatch(
    %update : tensor<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<4x?xf32> {
  // expected-error @+1 {{expected type of operand #2 ('tensor<?x?xf32>') to match type of corresponding result ('tensor<4x?xf32>')}}
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
      ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = arith.addf %arg1, %arg2 : f32
        iree_linalg_ext.yield %1 : f32
      } -> tensor<4x?xf32>
  return %0 : tensor<4x?xf32>
}

// -----

func.func @scatter_dim_mismatch(
    %update : tensor<?x?xf32>, %indices : tensor<48x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{mismatch in shape of indices and update value at dim#0}}
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%update, %indices : tensor<?x?xf32>, tensor<48x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @scatter_dim_mismatch(
    %update : tensor<64x?xf32>, %indices : tensor<48x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{mismatch in shape of indices and update value at dim#0}}
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%update, %indices : tensor<64x?xf32>, tensor<48x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @scatter_dim_mismatch(
    %update : tensor<?x?x?x?xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{op update value rank exceeds the rank of the original value}}
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%update, %indices : tensor<?x?x?x?xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @scatter_dim_mismatch(
    %update : tensor<?x4xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x3xf32>) -> tensor<?x3xf32> {
  // expected-error @+1 {{op shape of update value dim#1 exceeds original value at dim#1}}
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%update, %indices : tensor<?x4xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      iree_linalg_ext.yield %1 : f32
    } -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// -----

func.func @scatter_region_type_mismatch(
    %update : tensor<?x?xi32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{expected region to have scalar argument of integer or float types}}
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%update, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi32>) {
    ^bb0(%arg1: index, %arg2: index):
      %1 = arith.addi %arg1, %arg2 : index
      %2 = arith.index_cast %1 : index to i32
      iree_linalg_ext.yield %2 : i32
    } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

func.func @scatter_region_type_mismatch(
    %update : tensor<?x?xi32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{mismatch in argument 0 of region 'i64' and element type of update value 'i32'}}
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%update, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi32>) {
    ^bb0(%arg1: i64, %arg2: i32):
      %1 = arith.trunci %arg1 : i64 to i32
      %2 = arith.addi %1, %arg2 : i32
      iree_linalg_ext.yield %2 : i32
    } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

func.func @scatter_region_type_mismatch(
    %update : tensor<?x?xi32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{mismatch in argument 1 of region 'i64' and element type of original value 'i32'}}
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%update, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi32>) {
    ^bb0(%arg1: i32, %arg2: i64):
      %1 = arith.trunci %arg2 : i64 to i32
      %2 = arith.addi %1, %arg1 : i32
      iree_linalg_ext.yield %2 : i32
    } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

func.func @scatter_region_type_mismatch(
    %update : tensor<?x?xi32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{mismatch in region argument types 'i32' and 'i64'}}
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%update, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i32, %arg2: i64):
      %1 = arith.extsi %arg1 : i32 to i64
      %2 = arith.addi %1, %arg2 : i64
      iree_linalg_ext.yield %2 : i64
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----

func.func @scatter_region_type_mismatch(
    %update : tensor<?x?xi64>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{expected region to have two arguments}}
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%update, %indices : tensor<?x?xi64>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64, %arg3 : i64):
      %1 = arith.addi %arg1, %arg2 : i64
      iree_linalg_ext.yield %1 : i64
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}


// -----

func.func @scatter_yield_mismatch(
    %update : tensor<?x?xi64>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%update, %indices : tensor<?x?xi64>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      // expected-error @+1 {{mismatch in type of yielded value 'i32' and argument of the region 'i64'}}
      iree_linalg_ext.yield %2 : i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----

func.func @scatter_yield_mismatch(
    %update : tensor<?x?xi64>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%update, %indices : tensor<?x?xi64>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      // expected-error @+1 {{expected region to yield a single value}}
      iree_linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----

func.func @scatter_index_depth_dynamic(
    %update : tensor<?x?xi64>, %indices : tensor<?x?xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{expected index depth is static}}
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%update, %indices : tensor<?x?xi64>, tensor<?x?xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      iree_linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----

func.func @scatter_original_rank_mismatch(
    %update : tensor<?xi64>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{op index depth and update value does not cover rank of original value}}
  %0 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
    ins(%update, %indices : tensor<?xi64>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = arith.addi %arg1, %arg2 : i64
      %2 = arith.trunci %1 : i64 to i32
      iree_linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----

func.func @topk_invalid(%input_values: tensor<2x10xf32>, %input_indices: tensor<2x10xi32>, %out_values : tensor<2x3xf32>, %out_indices: tensor<2x3xi32>) -> (tensor<2x3xf32>, tensor<2x3xi32>) {
  // expected-error@+1 {{expected one or two input operands}}
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_indices, %input_indices, %input_indices : tensor<2x10xi32>, tensor<2x10xi32>, tensor<2x10xi32>)
        outs(%out_values, %out_indices : tensor<2x3xf32>, tensor<2x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<2x3xf32>, tensor<2x3xi32>
  return %0#0, %0#1 : tensor<2x3xf32>, tensor<2x3xi32>
}

// -----

func.func @topk_invalid(%input_values: tensor<2x10xi32>, %input_indices: tensor<2x10xi32>, %out_values : tensor<2x3xf32>, %out_indices: tensor<2x3xi32>) -> (tensor<2x3xf32>, tensor<2x3xi32>) {
   // expected-error@+1 {{expected input/output value types to be identical}}
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : tensor<2x10xi32> , tensor<2x10xi32>)
        outs(%out_values, %out_indices : tensor<2x3xf32>, tensor<2x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<2x3xf32>, tensor<2x3xi32>
  return %0#0, %0#1 : tensor<2x3xf32>, tensor<2x3xi32>
}

// -----

func.func @topk_invalid(%input_values: tensor<2x10xf32>, %input_indices: tensor<2x10xf32>, %out_values : tensor<2x3xf32>, %out_indices: tensor<2x3xi32>) -> (tensor<2x3xf32>, tensor<2x3xi32>) {
   // expected-error@+1 {{expected input/output indices types to be int}}
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : tensor<2x10xf32> , tensor<2x10xf32>)
        outs(%out_values, %out_indices : tensor<2x3xf32>, tensor<2x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<2x3xf32>, tensor<2x3xi32>
  return %0#0, %0#1 : tensor<2x3xf32>, tensor<2x3xi32>
}

// -----

func.func @topk_invalid(%input_values: tensor<10x2x10xf32>, %input_indices: tensor<10x2x10xi32>, %out_values : tensor<2x3xf32>, %out_indices: tensor<2x3xi32>) -> (tensor<2x3xf32>, tensor<2x3xi32>) {
   // expected-error@+1 {{expected input/output to have the same rank}}
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : tensor<10x2x10xf32> , tensor<10x2x10xi32>)
        outs(%out_values, %out_indices : tensor<2x3xf32>, tensor<2x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<2x3xf32>, tensor<2x3xi32>
  return %0#0, %0#1 : tensor<2x3xf32>, tensor<2x3xi32>
}

// -----

func.func @topk_invalid(%input_values: tensor<3x10xf32>, %input_indices: tensor<2x10xi32>, %out_values : tensor<2x3xf32>, %out_indices: tensor<2x3xi32>) -> (tensor<2x3xf32>, tensor<2x3xi32>) {
   // expected-error@+1 {{input indices/values shape must match}}
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : tensor<3x10xf32> , tensor<2x10xi32>)
        outs(%out_values, %out_indices : tensor<2x3xf32>, tensor<2x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<2x3xf32>, tensor<2x3xi32>
  return %0#0, %0#1 : tensor<2x3xf32>, tensor<2x3xi32>
}

// -----

func.func @topk_invalid(%input_values: tensor<2x10xf32>, %input_indices: tensor<2x10xi32>, %out_values : tensor<3x3xf32>, %out_indices: tensor<2x3xi32>) -> (tensor<3x3xf32>, tensor<2x3xi32>) {
   // expected-error@+1 {{output indices/values shape must match}}
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : tensor<2x10xf32> , tensor<2x10xi32>)
        outs(%out_values, %out_indices : tensor<3x3xf32>, tensor<2x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<3x3xf32>, tensor<2x3xi32>
  return %0#0, %0#1 : tensor<3x3xf32>, tensor<2x3xi32>
}

// -----

func.func @topk_invalid(%input_values: tensor<3x10xf32>, %input_indices: tensor<3x10xi32>, %out_values : tensor<2x3xf32>, %out_indices: tensor<2x3xi32>) -> (tensor<2x3xf32>, tensor<2x3xi32>) {
   // expected-error@+1 {{incompatible input/output shapes}}
  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices  : tensor<3x10xf32> , tensor<3x10xi32>)
        outs(%out_values, %out_indices : tensor<2x3xf32>, tensor<2x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          iree_linalg_ext.yield %0 : i1
        } -> tensor<2x3xf32>, tensor<2x3xi32>
  return %0#0, %0#1 : tensor<2x3xf32>, tensor<2x3xi32>
}

// -----

func.func @pack_invalid(%input: tensor<256x128xf32>, %output: tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32> {
  // expected-error@+1 {{the shape of output is not large enough to hold the packed data. Expected at least 'tensor<8x8x16x32xf32>', got 'tensor<8x8x32x16xf32>'}}
  %0 = iree_linalg_ext.pack %input inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %output : (tensor<256x128xf32> tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32>
  return %0 : tensor<8x8x32x16xf32>
}

// -----

func.func @pack_invalid(%input: tensor<256x128xf32>, %output: tensor<8x8x16x33xf32>) -> tensor<8x8x16x33xf32> {
  // expected-error@+1 {{invalid tile factor provided. Only full tiles are supported when padding_value is not set}}
  %0 = iree_linalg_ext.pack %input inner_dims_pos = [1, 0] inner_tiles = [16, 33] into %output : (tensor<256x128xf32> tensor<8x8x16x33xf32>) -> tensor<8x8x16x33xf32>
  return %0 : tensor<8x8x16x33xf32>
}

// -----

func.func @pad_and_pack_invalid_type(%input: tensor<13x15xf32>, %output: tensor<2x8x8x2xf32>, %pad: i32) -> tensor<2x8x8x2xf32> {
  // expected-error@+1 {{expected padding_value has 'f32' but got: 'i32'}}
  %0 = iree_linalg_ext.pack %input padding_value(%pad: i32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : (tensor<13x15xf32> tensor<2x8x8x2xf32>) -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}

// -----

func.func @pack_invalid(%input: tensor<256x128xf32>, %output: tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32> {
  // expected-error@+1 {{invalid inner_dims_pos vector}}
  %0 = iree_linalg_ext.pack %input inner_dims_pos = [2, 0] inner_tiles = [2, 2] into %output : (tensor<256x128xf32> tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32>
  return %0 : tensor<8x8x32x16xf32>
}

// -----

func.func @pack_invalid(%input: tensor<256x128xf32>, %output: tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32> {
  // expected-error@+1 {{invalid tile factor}}
  %0 = iree_linalg_ext.pack %input inner_dims_pos = [1, 0] inner_tiles = [0, 2] into %output : (tensor<256x128xf32> tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32>
  return %0 : tensor<8x8x32x16xf32>
}

// -----

// duplicate element in `inner_dims_pos`, fail.
func.func @pack_invalid(%input: tensor<256x128xf32>, %output: tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32> {
  // expected-error@+1 {{invalid inner_dims_pos vector}}
  %0 = iree_linalg_ext.pack %input inner_dims_pos = [1, 1] inner_tiles = [2, 2] into %output : (tensor<256x128xf32> tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32>
  return %0 : tensor<8x8x32x16xf32>
}

// -----

func.func @unpack_invalid(%output: tensor<256x128xf32>, %input: tensor<8x8x32x16xf32>) -> tensor<256x128xf32> {
  // expected-error@+1 {{the shape of output is not large enough to hold the packed data. Expected at least 'tensor<8x32x4x32xf32>', got 'tensor<8x8x32x16xf32>'}}
  %0 = iree_linalg_ext.unpack %input inner_dims_pos = [1, 0] inner_tiles = [4, 32] into %output : (tensor<8x8x32x16xf32> tensor<256x128xf32>) -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}

// -----

// duplicate element in `outer_dims_perm`, fail.
func.func @pack_invalid(%input: tensor<256x128xf32>, %output: tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32> {
  // expected-error@+1 {{invalid outer_dims_perm vector}}
  %0 = iree_linalg_ext.pack %input outer_dims_perm = [1, 1] inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %output : (tensor<256x128xf32> tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32>
  return %0 : tensor<8x8x32x16xf32>
}

// -----

// duplicate element in `outer_dims_perm`, fail.
func.func @pack_invalid(%input: tensor<256x128xf32>, %output: tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32> {
  // expected-error@+1 {{invalid outer_dims_perm vector}}
  %0 = iree_linalg_ext.unpack %output outer_dims_perm = [1, 1] inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %input : (tensor<8x8x32x16xf32> tensor<256x128xf32>) -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}

// -----

// `outer_dims_perm` is out of bound.
func.func @pack_invalid(%input: tensor<256x128xf32>, %output: tensor<8x8x32x16xf32>) -> tensor<8x8x32x16xf32> {
  // expected-error@+1 {{invalid outer_dims_perm vector}}
  %0 = iree_linalg_ext.unpack %output outer_dims_perm = [2, 1] inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %input : (tensor<8x8x32x16xf32> tensor<256x128xf32>) -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}

// -----
func.func @pack_mismatch_inner_tile_size_and_output_shape(
  %input : tensor<?x?xf32>, %output : tensor<?x?x8x8xf32>) -> tensor<?x?x8x8xf32> {
  // expected-error@+1 {{mismatch in inner tile sizes specified and shaped of tiled dimension in the packed type}}
  %0 = iree_linalg_ext.pack %input inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %output
      : (tensor<?x?xf32> tensor<?x?x8x8xf32>) -> tensor<?x?x8x8xf32>
  return %0 : tensor<?x?x8x8xf32>
}

// -----

func.func @unpack_mismatch_inner_tile_size_and_output_shape(
  %input : tensor<?x?x8x8xf32>, %output : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{mismatch in inner tile sizes specified and shaped of tiled dimension in the packed type}}
  %0 = iree_linalg_ext.unpack %input inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %output
      : (tensor<?x?x8x8xf32> tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @illegal_im2col_strides(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1024x5760xf32> {
  %0 = tensor.empty() : tensor<2x1024x5760xf32>
  // expected-error @+1 {{expected strides rank to be equal to the kernel rank}}
  %1 = iree_linalg_ext.im2col strides = [1] dilations = [1, 1] kernel_size = [3, 3]
           m_offset = [0] k_offset = [0] batch_pos = [0] m_pos = [1, 2] k_pos = [3]
           ins(%arg0 : tensor<2x34x34x640xf32>)
           outs(%0 : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  return %1 : tensor<2x1024x5760xf32>
}

// -----

func.func @illegal_im2col_dilations(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1024x5760xf32> {
  %0 = tensor.empty() : tensor<2x1024x5760xf32>
  // expected-error @+1 {{expected dilations rank to be equal to the kernel rank}}
  %1 = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1, 1] kernel_size = [3, 3]
           m_offset = [0] k_offset = [0] batch_pos = [0] m_pos = [1, 2] k_pos = [3]
           ins(%arg0 : tensor<2x34x34x640xf32>)
           outs(%0 : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  return %1 : tensor<2x1024x5760xf32>
}

// -----

func.func @illegal_im2col_kernel_size(%arg0: tensor<2x34x34x640xf32>) -> tensor<2x1024x5760xf32> {
  %0 = tensor.empty() : tensor<2x1024x5760xf32>
  // expected-error @+1 {{expected kernel rank to be equal to the m_pos rank}}
  %1 = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3]
           m_offset = [0] k_offset = [0] batch_pos = [0] m_pos = [1, 2] k_pos = [3]
           ins(%arg0 : tensor<2x34x34x640xf32>)
           outs(%0 : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  return %1 : tensor<2x1024x5760xf32>
}

// -----

func.func @illegal_im2col_input_rank(%arg0: tensor<1x2x34x34x640xf32>) -> tensor<2x1024x5760xf32> {
  %0 = tensor.empty() : tensor<2x1024x5760xf32>
  // expected-error @+1 {{expected input rank to be the sum of batch, m, and k ranks}}
  %1 = iree_linalg_ext.im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]
           m_offset = [0] k_offset = [0] batch_pos = [0] m_pos = [1, 2] k_pos = [3]
           ins(%arg0 : tensor<1x2x34x34x640xf32>)
           outs(%0 : tensor<2x1024x5760xf32>) -> tensor<2x1024x5760xf32>
  return %1 : tensor<2x1024x5760xf32>
}

// -----

func.func @illegal_winograd_input_shape(%arg0: tensor<1x10x10x32xf32>) -> tensor<8x8x1x6x6x32xf32> {
  %0 = tensor.empty() : tensor<8x8x1x6x6x32xf32>
  // expected-error @+1 {{incompatible output shape}}
  %1 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : tensor<1x10x10x32xf32>) outs(%0 : tensor<8x8x1x6x6x32xf32>) -> tensor<8x8x1x6x6x32xf32>
  return %1 : tensor<8x8x1x6x6x32xf32>
}

// -----

func.func @illegal_winograd_input_rank(%arg0: tensor<1x10x10x32xf32>) -> tensor<8x8x1x6xf32> {
  %0 = tensor.empty() : tensor<8x8x1x6xf32>
  // expected-error @+1 {{expected output rank to be equal to input rank + 2}}
  %1 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
    ins(%arg0 : tensor<1x10x10x32xf32>) outs(%0 : tensor<8x8x1x6xf32>) -> tensor<8x8x1x6xf32>
  return %1 : tensor<8x8x1x6xf32>
}

// -----

func.func @illegal_winograd_output_shape(%arg0: tensor<8x8x1x2x2x32xf32>) -> tensor<1x8x8x32xf32> {
  %0 = tensor.empty() : tensor<1x8x8x32xf32>
  // expected-error @+1 {{incompatible output shape}}
  %1 = iree_linalg_ext.winograd.output_transform output_tile_size(6)
        kernel_size(3) image_dimensions([1, 2])
        ins(%arg0 : tensor<8x8x1x2x2x32xf32>) outs(%0 : tensor<1x8x8x32xf32>) -> tensor<1x8x8x32xf32>
  return %1 : tensor<1x8x8x32xf32>
}

// -----

func.func @illegal_winograd_input_shape_nchw(%arg0: tensor<1x32x10x10xf32>) -> tensor<8x8x1x32x6x6xf32> {
  %0 = tensor.empty() : tensor<8x8x1x32x6x6xf32>
  // expected-error @+1 {{incompatible output shape}}
  %1 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
    ins(%arg0 : tensor<1x32x10x10xf32>) outs(%0 : tensor<8x8x1x32x6x6xf32>) -> tensor<8x8x1x32x6x6xf32>
  return %1 : tensor<8x8x1x32x6x6xf32>
}

// -----

func.func @illegal_winograd_input_image_dimensions(%arg0: tensor<1x1280x10x10xf32>) -> tensor<8x8x1x2x2x1280xf32> {
  %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
  // expected-error @+1 {{expect image dimensions to be either [1, 2] or [2, 3]}}
  %1 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([0, 3])
    ins(%arg0 : tensor<1x1280x10x10xf32>) outs(%0 : tensor<8x8x1x2x2x1280xf32>) -> tensor<8x8x1x2x2x1280xf32>
  return %1 : tensor<8x8x1x2x2x1280xf32>
}

// -----

func.func @illegal_winograd_output_image_dimensions(%arg0: tensor<8x8x1x2x2x32xf32>) -> tensor<1x32x12x12xf32> {
  %0 = tensor.empty() : tensor<1x32x12x12xf32>
  // expected-error @+1 {{expect image dimensions to be either [1, 2] or [2, 3]}}
  %1 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([0, 3])
        ins(%arg0 : tensor<8x8x1x2x2x32xf32>) outs(%0 : tensor<1x32x12x12xf32>) -> tensor<1x32x12x12xf32>
  return %1 : tensor<1x32x12x12xf32>
}

// -----

func.func @illegal_winograd_filter_kernel_shape(%arg0: tensor<3x2x64x128xf32>) -> tensor<8x8x64x128xf32> {
  %0 = tensor.empty() : tensor<8x8x64x128xf32>
  // expected-error @+1 {{expect all kernel dimensions to have the kernel size}}
  %1 = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1])
    ins(%arg0 : tensor<3x2x64x128xf32>) outs(%0 : tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
  return %1 : tensor<8x8x64x128xf32>
}

// -----

func.func @illegal_winograd_filter_kernel_shape_fchw(%arg0: tensor<128x64x3x2xf32>) -> tensor<8x8x64x128xf32> {
  %0 = tensor.empty() : tensor<8x8x64x128xf32>
  // expected-error @+1 {{expect all kernel dimensions to have the kernel size}}
  %1 = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([2, 3])
    ins(%arg0 : tensor<128x64x3x2xf32>) outs(%0 : tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
  return %1 : tensor<8x8x64x128xf32>
}

// -----

func.func @illegal_winograd_filter_result_shape(%arg0: tensor<3x3x64x128xf32>) -> tensor<8x8x128x64xf32> {
  %0 = tensor.empty() : tensor<8x8x128x64xf32>
  // expected-error @+1 {{incompatible output shape}}
  %1 = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1])
    ins(%arg0 : tensor<3x3x64x128xf32>) outs(%0 : tensor<8x8x128x64xf32>) -> tensor<8x8x128x64xf32>
  return %1 : tensor<8x8x128x64xf32>
}

// -----

func.func @illegal_winograd_filter_result_shape_fchw(%arg0: tensor<128x64x3x3xf32>) -> tensor<8x8x128x64xf32> {
  %0 = tensor.empty() : tensor<8x8x128x64xf32>
  // expected-error @+1 {{incompatible output shape}}
  %1 = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([2, 3])
    ins(%arg0 : tensor<128x64x3x3xf32>) outs(%0 : tensor<8x8x128x64xf32>) -> tensor<8x8x128x64xf32>
  return %1 : tensor<8x8x128x64xf32>
}

// -----

func.func @illegal_winograd_filter_kernel_dimensions(%arg0: tensor<3x3x64x128xf32>) -> tensor<8x8x64x128xf32> {
  %0 = tensor.empty() : tensor<8x8x64x128xf32>
  // expected-error @+1 {{expect kernel dimensions to be either [0, 1] or [2, 3]}}
  %1 = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([0, 3])
    ins(%arg0 : tensor<3x3x64x128xf32>) outs(%0 : tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32>
  return %1 : tensor<8x8x64x128xf32>
}

// -----

func.func @illegal_attention_inputs(%query: tensor<6x12x20x8xf32>, %key: tensor<6x12x20x8xf32>, %value: tensor<6x12x20x8xf32>) -> tensor<6x12x20x8xf32> {
  %0 = tensor.empty() : tensor<6x12x20x8xf32>
  %scale = arith.constant 1.0 : f32
  // expected-error @+1 {{Rank Mismatch for Query. Expected: 3 Got: 4}}
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>,
                    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>,
                    affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
                    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]}
                    ins(%query, %key, %value, %scale : tensor<6x12x20x8xf32>, tensor<6x12x20x8xf32>, tensor<6x12x20x8xf32>, f32) outs(%0 : tensor<6x12x20x8xf32>) -> tensor<6x12x20x8xf32>
  return %1 : tensor<6x12x20x8xf32>
}

// -----

func.func @illegal_flash_attention_inputs(%query: tensor<20xf32>, %key: tensor<20x8xf32>, %value: tensor<20x8xf32>) -> (tensor<20x8xf32>, tensor<8xf32>, tensor<8xf32>) {
  %result = tensor.empty() : tensor<20x8xf32>
  %max = tensor.empty() : tensor<8xf32>
  %sum = tensor.empty() : tensor<8xf32>
  %scale = arith.constant 1.0 : f32
  // expected-error @+1 {{Rank Mismatch for Query. Expected: 2 Got: 1}}
  %1:3 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d2, d1)>,
                       affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d3)>,
                       affine_map<(d0, d1, d2, d3) -> ()>,
                       affine_map<(d0, d1, d2, d3) -> (d0)>,
                       affine_map<(d0, d1, d2, d3) -> (d0)>,
                       affine_map<(d0, d1, d2, d3) -> (d0)>]}
                       ins(%query, %key, %value, %scale : tensor<20xf32>, tensor<20x8xf32>, tensor<20x8xf32>, f32) outs(%result, %max, %sum : tensor<20x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<20x8xf32>, tensor<8xf32>, tensor<8xf32>
  return %1#0, %1#1, %1#2 : tensor<20x8xf32>, tensor<8xf32>, tensor<8xf32>
}

// -----

func.func @illegal_attention_inputs(%query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>, %value: f32) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %scale = arith.constant 1.0 : f32
  // expected-error @+6 {{custom op 'iree_linalg_ext.attention' invalid kind of type specified}}
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>,
                    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>,
                    affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
                    affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]}
                    ins(%query, %key, %value, %scale : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, f32, f32) outs(%0 : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}

// -----

func.func @attention_missing_affine_map(%query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>, %value: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %scale = arith.constant 1.0 : f32
  // expected-error @below {{'iree_linalg_ext.attention' op expected an indexing map for each operand}}
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, f32) outs(%0 : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}

// -----

func.func @attention_affine_map_domain_mismatch(%query: tensor<192x1024x64xf32>, %key: tensor<192x1024x64xf32>, %value: tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32> {
  %0 = tensor.empty() : tensor<192x1024x64xf32>
  %scale = arith.constant 1.0 : f32
  // expected-error @below {{Mismatched map domain for Scale. Expected: 5 Got: 4}}
  %1 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                     affine_map<(d0, d1, d2, d3) -> ()>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]}
                     ins(%query, %key, %value, %scale : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, f32) outs(%0 : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
  return %1 : tensor<192x1024x64xf32>
}

// -----

func.func @custom_op_memref_operand(%arg0 : memref<?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{operand #0 must be variadic of ranked tensor of signless integer or index or floating-point values or signless integer or index or floating-point, but got 'memref<?xf32>'}}
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0, %arg1 : memref<?xf32>, tensor<?xf32>) outs(%arg2 : tensor<?xf32>) {
    ^bb0(%b0 : memref<?xf32>, %b1 : tensor<?xf32>, %b2 : tensor<?xf32>):
      iree_linalg_ext.yield %b1 : tensor<?xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @custom_op_scalar_outs_argument(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>, %arg2 : f32) -> f32 {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{operand #2 must be variadic of ranked tensor of any type values, but got 'f32'}}
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%arg2 : f32) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?xf32>, %b2 : f32):
      %1 = tensor.extract %b1[%c0] : tensor<?xf32>
      iree_linalg_ext.yield %1 : f32
  } -> f32
  return %0 : f32
}

// -----

func.func @custom_op_missing_indexing_maps(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+2 {{expected 'indexing_maps'}}
  %0 = iree_linalg_ext.custom_op {
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<?xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @custom_op_missing_iterator_types(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+3 {{expected 'iterator_types'}}
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<?xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @custom_op_outs_result_mismatch(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> tensor<10xf32> {
  // expected-error @+1 {{expected type of operand #1 ('tensor<?xf32>') to match type of corresponding result ('tensor<10xf32>')}}
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<?xf32>
  } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

func.func @custom_op_missing_indexing_map(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected number of indexing maps (2) to be same as the number of input/output operands (3)}}
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%arg2 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?xf32>, %b2 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<?xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @custom_op_indexing_map_inconsistent_num_symbol(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{inconsistent number of symbol dimensions in indexing_map #1, expected 1 instead of 0}}
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0)[s0] -> (d0 + s0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<?xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @custom_op_indexing_map_domain_mismatch(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected indexing_map #0 to have 1 dim(s) to match the number of loops or be zero}}
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<?xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @custom_op_indexing_map_range_mismatch(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected operand rank(1) to match the result rank of indexing map #0}}
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<?xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @custom_op_missing_bb_arg(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected as many basic block arguments (1) as the number of operands (2)}}
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<?xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @custom_op_scalar_operand_bb_arg_mismatch(%arg0 : tensor<?xf32>, %arg1 : f32, %arg2 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{for (scalar) operand #1 expected corresponding basic block argument to be of the same type}}
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0, %arg1 : tensor<?xf32>, f32) outs(%arg2 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<f32>, %b2 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<?xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @custom_op_vector_operand_bb_arg_mismatch(%arg0 : tensor<10xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected basic block argument corresponding to (tensor) operand #0 to be 'tensor<?xf32>' instead of 'tensor<10xf32>'}}
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0 : tensor<10xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<10xf32>, %b1 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<10xf32>
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @custom_op_number_of_yields_mismatch(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  // expected-error @+1 {{expected as many yields as the numbers of `outs` operand}}
  %0:2 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0 : tensor<?xf32>) outs(%arg1, %arg1 : tensor<?xf32>, tensor<?xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?xf32>, %b2 : tensor<?xf32>):
      iree_linalg_ext.yield %b0 : tensor<?xf32>
  } -> tensor<?xf32>, tensor<?xf32>
  return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

func.func @custom_op_yield_type_mismatch(%arg0 : tensor<?xf32>, %arg1 : tensor<10xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %1 = tensor.cast %arg1 : tensor<10xf32> to tensor<?xf32>
  // expected-error @+1 {{expected type of 1-th operand of yield to match the corresponding output basic block argument}}
  %0:2 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}
      ins(%arg0 : tensor<?xf32>) outs(%1, %arg1 : tensor<?xf32>, tensor<10xf32>) {
    ^bb0(%b0 : tensor<?xf32>, %b1 : tensor<?xf32>, %b2 : tensor<?xf32>):
      iree_linalg_ext.yield %b0, %arg1 : tensor<?xf32>, tensor<10xf32>
  } -> tensor<?xf32>, tensor<?xf32>
  return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}
