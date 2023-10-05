// RUN: iree-opt --split-input-file --torch-iree-set-strict-symbolic-shapes %s | FileCheck %s

module {
  // CHECK: func @forward() {{.*}} attributes {torch.assume_strict_symbolic_shapes}
  func.func @forward() -> !torch.int {
    %int0 = torch.constant.int 0
    return %int0 : !torch.int
  }
  // CHECK: func @other_forward() {{.*}} attributes {torch.assume_strict_symbolic_shapes}
  func.func @other_forward() -> !torch.int {
    %int1 = torch.constant.int 1
    return %int1 : !torch.int
  }
}
