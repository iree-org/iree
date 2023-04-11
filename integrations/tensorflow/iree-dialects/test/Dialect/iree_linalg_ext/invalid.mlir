// RUN: iree-dialects-opt --split-input-file --verify-diagnostics %s

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
  // expected-error @+1 {{expected number of outputs to be same as the number of results}}
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
  // expected-error @+1 {{expected type of `outs` operand #0 'tensor<?x?xf32>' to be same as result type 'tensor<4x?xf32>'}}
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

func.func @reverse_diff_element_type(%arg0: tensor<3x5xi32>) -> tensor<3x5xf32> {
  %init = tensor.empty() : tensor<3x5xf32>
  // expected-error @+1 {{expected input/output element types to be identical}}
  %0 = iree_linalg_ext.reverse
         dimensions(dense<0> : tensor<1xi64>)
         ins(%arg0 : tensor<3x5xi32>)
         outs(%init : tensor<3x5xf32>) : tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// -----

func.func @reverse_diff_shape(%arg0: tensor<3x5xi32>) -> tensor<3x6xi32> {
  %init = tensor.empty() : tensor<3x6xi32>
  // expected-error @+1 {{incompatible input/output shapes}}
  %0 = iree_linalg_ext.reverse
         dimensions(dense<0> : tensor<1xi64>)
         ins(%arg0 : tensor<3x5xi32>)
         outs(%init : tensor<3x6xi32>) : tensor<3x6xi32>
  return %0 : tensor<3x6xi32>
}

// -----

func.func @reverse_dup_dims(%arg0: tensor<3x5xi32>) -> tensor<3x5xi32> {
  %init = tensor.empty() : tensor<3x5xi32>
  // expected-error @+1 {{expected dimensions numbers are all unique}}
  %0 = iree_linalg_ext.reverse
         dimensions(dense<[0, 0]> : tensor<2xi64>)
         ins(%arg0 : tensor<3x5xi32>)
         outs(%init : tensor<3x5xi32>) : tensor<3x5xi32>
  return %0 : tensor<3x5xi32>
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

func.func @illegal_set_encoding_op_with_no_result_encoding(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{result of set_encoding op expected to have a valid tensor encoding}}
  %0 = iree_linalg_ext.set_encoding %arg0: tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @illegal_set_encoding_op_with_source_encoding(%arg0 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>) -> tensor<?x?xf32> {
  // expected-error @+1 {{source of set_encoding op cannot have a tensor encoding}}
  %0 = iree_linalg_ext.set_encoding %arg0: tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @illegal_set_encoding_op_with_unknown_encoding(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32, "gemm_lhs"> {
  // expected-error @+1 {{result of set_encoding op expected to have a valid tensor encoding}}
  %0 = iree_linalg_ext.set_encoding %arg0: tensor<?x?xf32> -> tensor<?x?xf32, "gemm_lhs">
  return %0 : tensor<?x?xf32, "gemm_lhs">
}

// -----

func.func @illegal_set_encoding_op_with_rank_change(%arg0 : tensor<?x?xf32>) -> tensor<?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> {
  // expected-error @+1 {{cannot change the rank of the tensor}}
  %0 = iree_linalg_ext.set_encoding %arg0: tensor<?x?xf32> -> tensor<?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  return %0 : tensor<?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
}

// -----

func.func @illegal_set_encoding_op_with_shape_change(%arg0 : tensor<10x20xf32>) -> tensor<20x30xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> {
  // expected-error @+1 {{expected to preserve the logical shape of the tensor}}
  %0 = iree_linalg_ext.set_encoding %arg0: tensor<10x20xf32> -> tensor<20x30xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  return %0 : tensor<20x30xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
}

// -----

func.func @illegal_unset_encoding_op_with_no_source_encoding(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{source of unset_encoding op expected to have a valid tensor encoding}}
  %0 = iree_linalg_ext.unset_encoding %arg0: tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @illegal_unset_encoding_op_with_result_encoding(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> {
  // expected-error @+1 {{result of unset_encoding op cannot have a tensor encoding}}
  %0 = iree_linalg_ext.unset_encoding %arg0: tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  return %0 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
}

// -----

func.func @illegal_unset_encoding_op_with_unknown_encoding(%arg0 : tensor<?x?xf32, "gemm_lhs">) -> tensor<?x?xf32> {
  // expected-error @+1 {{source of unset_encoding op expected to have a valid tensor encoding}}
  %0 = iree_linalg_ext.unset_encoding %arg0: tensor<?x?xf32, "gemm_lhs"> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @illegal_unset_encoding_op_with_rank_change(%arg0 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>) -> tensor<?xf32> {
  // expected-error @+1 {{cannot change the rank of the tensor}}
  %0 = iree_linalg_ext.unset_encoding %arg0: tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @illegal_unset_encoding_op_with_shape_change(%arg0 : tensor<20x30xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>) -> tensor<10x20xf32> {
  // expected-error @+1 {{expected to preserve the logical shape of the tensor}}
  %0 = iree_linalg_ext.unset_encoding %arg0: tensor<20x30xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> -> tensor<10x20xf32>
  return %0 : tensor<10x20xf32>
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

func.func @illegal_softmax_output_shape(%arg0: tensor<2x16x32xf32>) -> tensor<2x16xf32> {
  %0 = tensor.empty() : tensor<2x16xf32>
  // expected-error @+1 {{incompatible output shape}}
  %1 = iree_linalg_ext.softmax dimension(2) ins(%arg0 : tensor<2x16x32xf32>) outs(%0: tensor<2x16xf32>) -> tensor<2x16xf32>
  return %1 : tensor<2x16xf32>
}

// -----

func.func @illegal_attention_inputs(%query: tensor<6x12x20x8xf32>, %key: tensor<6x12x20x8xf32>, %value: tensor<6x12x20x8xf32>) {
  %0 = tensor.empty() : tensor<6x12x20x8xf32>
  // expected-error @+1 {{failed to verify that query has shaped type of rank 3}}
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<6x12x20x8xf32>, tensor<6x12x20x8xf32>, tensor<6x12x20x8xf32>) outs(%0 : tensor<6x12x20x8xf32>) -> tensor<6x12x20x8xf32>
  return %1 : tensor<6x12x20x8xf32>
}

// -----
