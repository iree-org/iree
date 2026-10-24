// RUN: iree-opt --iree-llvmcpu-propagate-dispatch-status --iree-convert-to-llvm --split-input-file %s | FileCheck %s

// Status failures are returned through the entry-point ABI.
module {
  func.func public @entry_point() {
    %status = arith.constant 9 : i32
    util.status.check_ok %status
    return
  }
  iree_codegen.dispatch_config @entry_point workgroup_size = [1, 1, 1] {
  ^bb0(%device: !hal.device):
    %one = arith.constant 1 : index
    iree_codegen.yield %one, %one, %one : index, index, index
  }
}
// CHECK-LABEL: llvm.func @entry_point(
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !llvm.ptr
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !llvm.ptr
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !llvm.ptr
// CHECK-SAME: ) -> i32 attributes {{.*}}passthrough = ["iree-returns-status"]{{.*}} {
// CHECK: %[[FAIL_STATUS:[a-zA-Z0-9_]+]] = llvm.mlir.constant(9 : i32) : i32
// CHECK: %[[OK_STATUS:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.cond_br %{{.*}}, ^[[STATUS_BLOCK:bb[0-9]+]](%[[FAIL_STATUS]] : i32), ^[[STATUS_BLOCK]](%[[OK_STATUS]] : i32)
// CHECK: ^[[STATUS_BLOCK]](%[[RETURN_STATUS:[a-zA-Z0-9_]+]]: i32)
// CHECK: llvm.return %[[RETURN_STATUS]] : i32

// A private helper returns its value and status; the caller propagates a
// nonzero status before returning the helper's value.
// -----
module {
  func.func private @helper(%status: i32) -> i32 {
    util.status.check_ok %status
    %result = arith.constant 42 : i32
    return %result : i32
  }

  func.func private @consume(i32)

  func.func public @entry_point() {
    %status = arith.constant 9 : i32
    %result = func.call @helper(%status) : (i32) -> i32
    func.call @consume(%result) : (i32) -> ()
    return
  }
}
// CHECK-LABEL: llvm.func @helper(%{{.*}}: i32) -> !llvm.struct<(i32, i32)> attributes {{.*}} {
// CHECK: %[[HELPER_RETURN:[a-zA-Z0-9_]+]] = llvm.insertvalue {{.*}}[1] : !llvm.struct<(i32, i32)>
// CHECK: llvm.return %[[HELPER_RETURN]] : !llvm.struct<(i32, i32)>
// CHECK-LABEL: llvm.func @entry_point
// CHECK: %[[CALL_RESULT:[a-zA-Z0-9_]+]] = llvm.call @helper
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[CALL_RESULT]][0]
// CHECK: %[[STATUS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[CALL_RESULT]][1]
// CHECK: %[[FAILED:[a-zA-Z0-9_]+]] = llvm.icmp "ne" %[[STATUS]], {{.*}} : i32
// CHECK: llvm.cond_br %[[FAILED]], ^[[RETURN_BLOCK:bb[0-9]+]](%[[STATUS]] : i32), ^[[SUCCESS_BLOCK:bb[0-9]+]]
// CHECK: ^[[SUCCESS_BLOCK]]:
// CHECK: %[[PACKED_VALUE:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[VALUE]], {{.*}}[0] : !llvm.struct<(i32)>
// CHECK: llvm.store %[[PACKED_VALUE]]
// CHECK: llvm.call %{{.*}}(
// CHECK: ^[[RETURN_BLOCK]](%[[RETURN_STATUS:[a-zA-Z0-9_]+]]: i32)
// CHECK: llvm.return %[[RETURN_STATUS]] : i32

// Nested control flow preserves the original successful return value.
// -----
module {
  func.func private @nested_control_flow(%condition: i1, %status: i32) -> i32 {
    %result = arith.constant 42 : i32
    scf.if %condition {
      util.status.check_ok %status
    }
    return %result : i32
  }

  func.func private @consume(i32)

  func.func public @entry_point() {
    %condition = arith.constant true
    %status = arith.constant 9 : i32
    %result = func.call @nested_control_flow(%condition, %status) : (i1, i32) -> i32
    func.call @consume(%result) : (i32) -> ()
    return
  }
}
// CHECK-LABEL: llvm.func @nested_control_flow
// CHECK-NOT: util.status.check_ok
// CHECK: llvm.cond_br
// CHECK: %[[FAILED:[a-zA-Z0-9_]+]] = llvm.icmp "ne" %arg1, %{{.*}} : i32
// CHECK: llvm.cond_br %[[FAILED]], ^[[MERGE_BLOCK:bb[0-9]+]]({{.*}}, %arg1 : i32, i32), ^bb{{[0-9]+}}
// CHECK: llvm.br ^[[MERGE_BLOCK]]({{.*}}, {{.*}} : i32, i32)
// CHECK: ^[[MERGE_BLOCK]](%[[MERGED_VALUE:[a-zA-Z0-9_]+]]: i32, %[[MERGED_STATUS:[a-zA-Z0-9_]+]]: i32)
// CHECK: %[[MERGED_RESULT_VALUE:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[MERGED_VALUE]], {{.*}}[0] : !llvm.struct<(i32, i32)>
// CHECK: %[[MERGED_RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[MERGED_STATUS]], %[[MERGED_RESULT_VALUE]][1] : !llvm.struct<(i32, i32)>
// CHECK: llvm.return %[[MERGED_RESULT]] : !llvm.struct<(i32, i32)>
