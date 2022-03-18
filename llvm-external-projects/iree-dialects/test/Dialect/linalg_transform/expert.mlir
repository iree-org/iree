// RUN: iree-dialects-opt -linalg-transform-expert-expansion -split-input-file %s | FileCheck %s --check-prefix=EXPAND
// RUN: iree-dialects-opt -linalg-transform-expert-expansion -linalg-interp-transforms -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @matmul_tensors
// CHECK-NOT: linalg
// CHECK: llvm
func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32> { linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  return %0 : tensor<128x128xf32>
}

pdl.pattern @pdl_target : benefit(1) {
  %args = operands
  %results = types
  %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@matmul_tensors](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  // This should match the strategy below.
  // EXPAND-NOT: expert apply
  // EXPAND: %[[OP:.*]] = match @pdl_target
  // EXPAND: %[[HANDLE:.*]] = tile %[[OP]] {sizes = [4, 4, 4]}
  // EXPAND: %[[HANDLE2:.*]] = vectorize %[[HANDLE]] {vectorize_padding = true}
  // EXPAND: bufferize
  // EXPAND: lower_vectors {multireduction_lowering = "innerreduce"}
  // EXPAND: lower_to_llvm
  %0 = match @pdl_target
  expert apply "single_tiling" to %0
  {
    tile_sizes = [4, 4, 4],
    vectorize_padding = true,
    multireduction_lowering = "innerreduce"
  }
}

// CHECK-NOT: @strategies
// EXPAND-NOT: @strategies
module @strategies {
  pdl.pattern @single_tiling_matcher : benefit(1) {
    %tile_sizes = attribute
    %vectorize_padding = attribute
    %multireduction_lowering = attribute
    %name = attribute : "single_tiling"
    %type = type : !pdl.operation
    %target = operand : %type
    %transformed = type
    %root = operation "iree_linalg_transform.expert"(%target : !pdl.value) {
      "expertName" = %name,
      "tile_sizes" = %tile_sizes,
      "vectorize_padding" = %vectorize_padding,
      "multireduction_lowering" = %multireduction_lowering
    } -> (%transformed : !pdl.type)

    rewrite %root {
      %tile = operation "iree_linalg_transform.tile"(%target : !pdl.value) {
        "sizes" = %tile_sizes
      } -> (%transformed : !pdl.type)
      %handle = result 0 of %tile

      %vectorize = operation "iree_linalg_transform.vectorize"(%handle : !pdl.value) {
        "vectorize_padding" = %vectorize_padding
      } -> (%transformed : !pdl.type)
      %handle2 = result 0 of %vectorize

      %bufferize = operation "iree_linalg_transform.bufferize"
      %lower_vectors = operation "iree_linalg_transform.lower_vectors" {
        "multireduction_lowering" = %multireduction_lowering
      }
      %lower_to_llvm = operation "iree_linalg_transform.lower_to_llvm"

      replace %root with (%handle2 : !pdl.value)
    }
  }
}

// -----

// CHECK-LABEL: func @matmul_tensors2
// CHECK-NOT: linalg
// CHECK: llvm
func @matmul_tensors2(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32> { linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  return %0 : tensor<128x128xf32>
}

pdl.pattern @pdl_target2 : benefit(1) {
  %args = pdl.operands
  %results = pdl.types
  %0 = pdl.operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  pdl.apply_native_constraint "nestedInFunc"[@matmul_tensors2](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  pdl.rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  // This should match the strategy below.
  // EXPAND-NOT: expert apply
  // EXPAND: %[[OP:.*]] = match @pdl_target2
  // EXPAND: %[[HANDLE:.*]] = tile %[[OP]] {sizes = [32, 8, 8]}
  // EXPAND: %[[HANDLE2:.*]] = tile %[[HANDLE]] {sizes = [4, 4, 4]}
  // EXPAND: %[[HANDLE3:.*]] = vectorize %[[HANDLE2]] {vectorize_padding = false}
  // EXPAND: bufferize
  // EXPAND: lower_vectors {multireduction_lowering = "innerparallel"}
  // EXPAND: lower_to_llvm
  %0 = match @pdl_target2
  %1 = tile %0 {sizes = [32, 8, 8]}
  expert apply "single_tiling" to %1
  {
    tile_sizes = [4, 4, 4],
    vectorize_padding = false,
    multireduction_lowering = "innerparallel"
  }
}

module @strategies {
  pdl.pattern @single_tiling_operand : benefit(1) {
    %tile_sizes = attribute
    %vectorize_padding = attribute
    %multireduction_lowering = attribute
    %name = attribute : "single_tiling"
    %type = type : !pdl.operation
    %target = operand : %type
    %transformed = type
    %root = operation "iree_linalg_transform.expert"(%target : !pdl.value) {
      "expertName" = %name,
      "tile_sizes" = %tile_sizes,
      "vectorize_padding" = %vectorize_padding,
      "multireduction_lowering" = %multireduction_lowering
    } -> (%transformed : !pdl.type)

    rewrite %root {
      %tile = operation "iree_linalg_transform.tile"(%target : !pdl.value)  {
        "sizes" = %tile_sizes
      } -> (%transformed : !pdl.type)
      %handle = result 0 of %tile

      %vectorize = operation "iree_linalg_transform.vectorize"(%handle : !pdl.value) {
        "vectorize_padding" = %vectorize_padding
      } -> (%transformed : !pdl.type)
      %handle2 = result 0 of %vectorize

      %bufferize = operation "iree_linalg_transform.bufferize"
      %lower_vectors = operation "iree_linalg_transform.lower_vectors" {
        "multireduction_lowering" = %multireduction_lowering
      }
      %lower_to_llvm = operation "iree_linalg_transform.lower_to_llvm"

      replace %root with (%handle2 : !pdl.value)
    }
  }
}
