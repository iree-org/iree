// RUN: iree-opt -split-input-file -verify-diagnostics %s

func @sort_invalid_dimension(%arg0: tensor<128xi32>) -> tensor<128xi32> {
  // expected-error @+1 {{dimension must be within (0, 1]}}
  %0 = linalg_ext.sort dimension(1)
    outs(%arg0 : tensor<128xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
    %1 = cmpi sgt, %arg1, %arg2 : i32
    linalg_ext.yield %1 : i1
  } -> tensor<128xi32>
  return %0 : tensor<128xi32>
}

// -----

func @sort_without_dimension(%arg0: tensor<3x4xi32>) -> tensor<3x4xi32> {
  // expected-error @+1 {{dimension must be specified if rank > 1}}
  %0 = linalg_ext.sort
    outs(%arg0 : tensor<3x4xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
    %1 = cmpi sgt, %arg1, %arg2 : i32
    linalg_ext.yield %1 : i1
  } -> tensor<3x4xi32>
  return %0 : tensor<3x4xi32>
}

// -----

func @sort_mismatch_rank(%arg0: tensor<?x?xi32>, %arg1: tensor<?xf32>)
    -> (tensor<?x?xi32>, tensor<?xf32>) {
  // expected-error @+1 {{expected operand 1 to be rank 2, same as other operands}}
  %0:2 = linalg_ext.sort dimension(0)
      outs(%arg0, %arg1 : tensor<?x?xi32>, tensor<?xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %1 = cmpf ogt, %arg4, %arg5 : f32
        linalg_ext.yield %1 : i1
      } -> tensor<?x?xi32>, tensor<?xf32>
  return %0#0, %0#1 : tensor<?x?xi32>, tensor<?xf32>
}

// -----

func @sort_mismatch_shape(%arg0: tensor<?xi32>, %arg1: tensor<42xf32>)
    -> (tensor<?xi32>, tensor<42xf32>) {
  // expected-error @+1 {{expected operand 1 to have same shape as other operands}}
  %0:2 = linalg_ext.sort dimension(0)
      outs(%arg0, %arg1 : tensor<?xi32>, tensor<42xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):  // no predecessors
        %1 = cmpf ogt, %arg4, %arg5 : f32
        linalg_ext.yield %1 : i1
      } -> tensor<?xi32>, tensor<42xf32>
  return %0#0, %0#1 : tensor<?xi32>, tensor<42xf32>
}

// -----

func @scatter_mixed_tensor_memref(
    %update : memref<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{expected inputs and outputs to be RankedTensorType or scalar}}
  %0 = linalg_ext.scatter
      ins(%update, %indices : memref<?x?xf32>, tensor<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @scatter_mixed_tensor_memref(
    %update : tensor<?x?xf32>, %indices : memref<?x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{expected inputs and outputs to be RankedTensorType or scalar}}
  %0 = linalg_ext.scatter
      ins(%update, %indices : tensor<?x?xf32>, memref<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @scatter_extra_outputs(
    %update : tensor<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // expected-error @+1 {{expected number of outputs to be same as the number of results}}
  %0, %1 = linalg_ext.scatter
      ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<?x?xf32>, tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

func @scatter_mixed_tensor_memref(
    %update : tensor<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : memref<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{expected inputs and outputs to be RankedTensorType or scalar}}
  %0 = linalg_ext.scatter
      ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%original : memref<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @scatter_mixed_tensor_memref(
    %update : tensor<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> memref<?x?xf32> {
  // expected-error @+1 {{expected type of `outs` operand #0 'tensor<?x?xf32>' to be same as result type 'memref<?x?xf32>'}}
  %0 = linalg_ext.scatter
      ins(%update, %indices : tensor<?x?xf32>, tensor<?x1xi32>)
      outs(%original : tensor<?x?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %1 = addf %arg1, %arg2 : f32
        linalg_ext.yield %1 : f32
      } -> memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// -----

func @scatter_mixed_tensor_memref(
    %update : memref<?x?xf32>, %indices : tensor<?x1xi32>,
    %original : memref<?x?xf32>) {
  // expected-error @+1 {{expected inputs and outputs to be MemRefType or scalar}}
  linalg_ext.scatter
    ins(%update, %indices : memref<?x?xf32>, tensor<?x1xi32>)
    outs(%original : memref<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    }
  return
}

// -----

func @scatter_mixed_tensor_memref(
    %update : memref<?x?xf32>, %indices : memref<?x1xi32>,
    %original : tensor<?x?xf32>) {
  // expected-error @+1 {{expected inputs and outputs to be MemRefType or scalar}}
  linalg_ext.scatter
    ins(%update, %indices : memref<?x?xf32>, memref<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    }
  return
}

// -----

func @scatter_dim_mismatch(
    %update : tensor<?x?xf32>, %indices : tensor<48x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{mismatch in shape of indices and update value at dim#0}}
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<?x?xf32>, tensor<48x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @scatter_dim_mismatch(
    %update : tensor<64x?xf32>, %indices : tensor<48x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{mismatch in shape of indices and update value at dim#0}}
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<64x?xf32>, tensor<48x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @scatter_dim_mismatch(
    %update : tensor<?x?x?xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{mismatch in rank of update value, index depth and original value}}
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<?x?x?xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @scatter_dim_mismatch(
    %update : tensor<?x4xf32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{mismatch in shape of update value dim#1 and original value at dim#1}}
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<?x4xf32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = addf %arg1, %arg2 : f32
      linalg_ext.yield %1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @scatter_region_type_mismatch(
    %update : tensor<?x?xi32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{expected region to have scalar argument of integer or float types}}
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi32>) {
    ^bb0(%arg1: index, %arg2: index):
      %1 = addi %arg1, %arg2 : index
      %2 = index_cast %1 : index to i32
      linalg_ext.yield %2 : i32
    } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

func @scatter_region_type_mismatch(
    %update : tensor<?x?xi32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{mismatch in argument 0 of region 'i64' and element type of update value 'i32'}}
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi32>) {
    ^bb0(%arg1: i64, %arg2: i32):
      %1 = trunci %arg1 : i64 to i32
      %2 = addi %1, %arg2 : i32
      linalg_ext.yield %2 : i32
    } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

func @scatter_region_type_mismatch(
    %update : tensor<?x?xi32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{mismatch in argument 1 of region 'i64' and element type of original value 'i32'}}
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi32>) {
    ^bb0(%arg1: i32, %arg2: i64):
      %1 = trunci %arg2 : i64 to i32
      %2 = addi %1, %arg1 : i32
      linalg_ext.yield %2 : i32
    } -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

func @scatter_region_type_mismatch(
    %update : tensor<?x?xi32>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{mismatch in region argument types 'i32' and 'i64'}}
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<?x?xi32>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i32, %arg2: i64):
      %1 = sexti %arg1 : i32 to i64
      %2 = addi %1, %arg2 : i64
      linalg_ext.yield %2 : i64
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----

func @scatter_region_type_mismatch(
    %update : tensor<?x?xi64>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{expected region to have two arguments}}
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<?x?xi64>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64, %arg3 : i64):
      %1 = addi %arg1, %arg2 : i64
      linalg_ext.yield %1 : i64
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}


// -----

func @scatter_yield_mismatch(
    %update : tensor<?x?xi64>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<?x?xi64>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = addi %arg1, %arg2 : i64
      %2 = trunci %1 : i64 to i32
      // expected-error @+1 {{mismatch in type of yielded value 'i32' and argument of the region 'i64'}}
      linalg_ext.yield %2 : i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----

func @scatter_yield_mismatch(
    %update : tensor<?x?xi64>, %indices : tensor<?x1xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<?x?xi64>, tensor<?x1xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = addi %arg1, %arg2 : i64
      %2 = trunci %1 : i64 to i32
      // expected-error @+1 {{expected region to yield a single value}}
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----

func @scatter_index_depth_dynamic(
    %update : tensor<?x?xi64>, %indices : tensor<?x?xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{expected index depth is static}}
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<?x?xi64>, tensor<?x?xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = addi %arg1, %arg2 : i64
      %2 = trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}

// -----

func @scatter_original_rank_mismatch(
    %update : tensor<?x?xi64>, %indices : tensor<?x2xi32>,
    %original : tensor<?x?xi64>) -> tensor<?x?xi64> {
  // expected-error @+1 {{mismatch in rank of update value, index depth and original value}}
  %0 = linalg_ext.scatter
    ins(%update, %indices : tensor<?x?xi64>, tensor<?x2xi32>)
    outs(%original : tensor<?x?xi64>) {
    ^bb0(%arg1: i64, %arg2: i64):
      %1 = addi %arg1, %arg2 : i64
      %2 = trunci %1 : i64 to i32
      linalg_ext.yield %1, %2 : i64, i32
    } -> tensor<?x?xi64>
  return %0 : tensor<?x?xi64>
}
