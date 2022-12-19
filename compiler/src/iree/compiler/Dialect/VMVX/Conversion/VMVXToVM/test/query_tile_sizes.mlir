// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @query_tile_sizes_2d
func.func @query_tile_sizes_2d(
    // Encoding
    %arg0 : i64,
    // Sizes
    %arg1 : index, %arg2 : index
    ) -> (index, index) {
  //      CHECK: vm.call @vmvx.query_tile_sizes.2d(
  // CHECK-SAME: %arg0, %arg1, %arg2
  // CHECK-SAME: ) : (i64, i64, i64) -> (i64, i64)
  %tile_size0, %tile_size1 = vmvx.query_tile_sizes encoding(%arg0) sizes (%arg1, %arg2) -> index, index
  return %tile_size0, %tile_size1 : index, index
}
