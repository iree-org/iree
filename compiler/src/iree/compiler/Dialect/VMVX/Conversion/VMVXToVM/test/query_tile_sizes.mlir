// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @foo
func.func @foo(
    // Sizes
    %arg0 : index, %arg1 : index
    ) -> (index, index) {
  //      CHECK: %[[FLAGS:.+]] = vm.const.i32 1048576
  //      CHECK: vm.call @vmvx.query_tile_sizes.2d(%arg0, %arg1, %[[FLAGS]]) : (i64, i64, i32) -> (i64, i64)
  %tile_sizes:2 = vmvx.query_tile_sizes sizes (%arg0, %arg1) flags (1048576) -> index, index
  return %tile_sizes#0, %tile_sizes#1 : index, index
}
