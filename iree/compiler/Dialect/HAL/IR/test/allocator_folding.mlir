// Tests folding and canonicalization of HAL allocator ops.

// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @simplify_allocate_shaped
func @simplify_allocate_shapedy() -> !iree.ref<!hal.buffer> {
  // CHECK-DAG: [[AL:%.+]] = "test_hal.allocator"
  %0 = "test_hal.allocator"() : () -> !iree.ref<!hal.allocator>
  // CHECK-DAG: [[SH:%.+]]:2 = "test_hal.shape"
  %1:2 = "test_hal.shape"() : () -> (i32, i32)
  // CHECK-NEXT: %buffer = hal.allocator.allocate.shaped [[AL]], "HostVisible|HostCoherent", "Transfer", shape=[
  // CHECK-SAME:     [[SH]]#0, [[SH]]#1
  // CHECK-SAME: ], element_size=4 : !iree.ref<!hal.buffer>
  %sz = hal.allocator.compute_size %0, "HostVisible|HostCoherent", "Transfer", shape=[%1#0, %1#1], element_size=4
  %buffer = hal.allocator.allocate %0, "HostVisible|HostCoherent", "Transfer", %sz : !iree.ref<!hal.buffer>
  // CHECK-NEXT: return %buffer
  return %buffer : !iree.ref<!hal.buffer>
}
