// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Tests folding and canonicalization of arithmetic ops.

// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(canonicalize)' %s | IreeFileCheck %s

// CHECK-LABEL: @add_i32_folds
vm.module @add_i32_folds {
  // CHECK-LABEL: @add_i32_0_y
  vm.func @add_i32_0_y(%arg0 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.add.i32 %zero, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @add_i32_x_0
  vm.func @add_i32_x_0(%arg0 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.add.i32 %arg0, %zero : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @add_i32_const
  vm.func @add_i32_const() -> i32 {
    // CHECK: %c5 = vm.const.i32 5 : i32
    // CHECK-NEXT: vm.return %c5 : i32
    %c1 = vm.const.i32 1 : i32
    %c4 = vm.const.i32 4 : i32
    %0 = vm.add.i32 %c1, %c4 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @sub_i32_folds
vm.module @sub_i32_folds {
  // CHECK-LABEL: @sub_i32_x_0
  vm.func @sub_i32_x_0(%arg0 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.sub.i32 %arg0, %zero : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @sub_i32_const
  vm.func @sub_i32_const() -> i32 {
    // CHECK: %c3 = vm.const.i32 3 : i32
    // CHECK-NEXT: vm.return %c3 : i32
    %c1 = vm.const.i32 1 : i32
    %c4 = vm.const.i32 4 : i32
    %0 = vm.sub.i32 %c4, %c1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @mul_i32_folds
vm.module @mul_i32_folds {
  // CHECK-LABEL: @mul_i32_by_0
  vm.func @mul_i32_by_0(%arg0 : i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.mul.i32 %arg0, %zero : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @mul_i32_1_y
  vm.func @mul_i32_1_y(%arg0 : i32) -> i32 {
    // CHECK-NEXT: vm.return %arg0 : i32
    %c1 = vm.const.i32 1 : i32
    %0 = vm.mul.i32 %c1, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @mul_i32_x_1
  vm.func @mul_i32_x_1(%arg0 : i32) -> i32 {
    // CHECK-NEXT: vm.return %arg0 : i32
    %c1 = vm.const.i32 1 : i32
    %0 = vm.mul.i32 %arg0, %c1 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @mul_i32_const
  vm.func @mul_i32_const() -> i32 {
    // CHECK: %c8 = vm.const.i32 8 : i32
    // CHECK-NEXT: vm.return %c8 : i32
    %c2 = vm.const.i32 2 : i32
    %c4 = vm.const.i32 4 : i32
    %0 = vm.mul.i32 %c2, %c4 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @div_i32_folds
vm.module @div_i32_folds {
  // CHECK-LABEL: @div_i32_0_y
  vm.func @div_i32_0_y(%arg0 : i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.div.i32.s %zero, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @div_i32_x_1
  vm.func @div_i32_x_1(%arg0 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %c1 = vm.const.i32 1 : i32
    %0 = vm.div.i32.s %arg0, %c1 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @div_i32_const
  vm.func @div_i32_const() -> i32 {
    // CHECK: %c3 = vm.const.i32 3 : i32
    // CHECK-NEXT: vm.return %c3 : i32
    %c15 = vm.const.i32 15 : i32
    %c5 = vm.const.i32 5 : i32
    %0 = vm.div.i32.s %c15, %c5 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @rem_i32_folds
vm.module @rem_i32_folds {
  // CHECK-LABEL: @rem_i32_x_1
  vm.func @rem_i32_x_1(%arg0 : i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %c1 = vm.const.i32 1 : i32
    %0 = vm.rem.i32.s %arg0, %c1 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @rem_i32_0_y
  vm.func @rem_i32_0_y(%arg0 : i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.rem.i32.s %zero, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @rem_i32_const
  vm.func @rem_i32_const() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %c3 = vm.const.i32 3 : i32
    %c4 = vm.const.i32 4 : i32
    %0 = vm.rem.i32.s %c4, %c3 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @not_i32_folds
vm.module @not_i32_folds {
  // CHECK-LABEL: @not_i32_const
  vm.func @not_i32_const() -> i32 {
    // CHECK: %c889262066 = vm.const.i32 889262066 : i32
    // CHECK-NEXT: vm.return %c889262066 : i32
    %c = vm.const.i32 0xCAFEF00D : i32
    %0 = vm.not.i32 %c : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @and_i32_folds
vm.module @and_i32_folds {
  // CHECK-LABEL: @and_i32_zero
  vm.func @and_i32_zero(%arg0 : i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.and.i32 %arg0, %zero : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @and_i32_eq
  vm.func @and_i32_eq(%arg0 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %0 = vm.and.i32 %arg0, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @and_i32_const
  vm.func @and_i32_const() -> i32 {
    // CHECK: %c1 = vm.const.i32 1 : i32
    // CHECK-NEXT: vm.return %c1 : i32
    %c1 = vm.const.i32 1 : i32
    %c3 = vm.const.i32 3 : i32
    %0 = vm.and.i32 %c1, %c3 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @or_i32_folds
vm.module @or_i32_folds {
  // CHECK-LABEL: @or_i32_0_y
  vm.func @or_i32_0_y(%arg0 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.or.i32 %zero, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @or_i32_x_0
  vm.func @or_i32_x_0(%arg0 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.or.i32 %arg0, %zero : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @or_i32_x_x
  vm.func @or_i32_x_x(%arg0 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %0 = vm.or.i32 %arg0, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @or_i32_const
  vm.func @or_i32_const() -> i32 {
    // CHECK: %c5 = vm.const.i32 5 : i32
    // CHECK-NEXT: vm.return %c5 : i32
    %c1 = vm.const.i32 1 : i32
    %c4 = vm.const.i32 4 : i32
    %0 = vm.or.i32 %c1, %c4 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @xor_i32_folds
vm.module @xor_i32_folds {
  // CHECK-LABEL: @xor_i32_0_y
  vm.func @xor_i32_0_y(%arg0 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.xor.i32 %zero, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @xor_i32_x_0
  vm.func @xor_i32_x_0(%arg0 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.xor.i32 %arg0, %zero : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @xor_i32_x_x
  vm.func @xor_i32_x_x(%arg0 : i32) -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %0 = vm.xor.i32 %arg0, %arg0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @xor_i32_const
  vm.func @xor_i32_const() -> i32 {
    // CHECK: %c2 = vm.const.i32 2 : i32
    // CHECK-NEXT: vm.return %c2 : i32
    %c1 = vm.const.i32 1 : i32
    %c3 = vm.const.i32 3 : i32
    %0 = vm.xor.i32 %c1, %c3 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shl_i32_folds
vm.module @shl_i32_folds {
  // CHECK-LABEL: @shl_i32_0_by_y
  vm.func @shl_i32_0_by_y() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.shl.i32 %zero, 4 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @shl_i32_x_by_0
  vm.func @shl_i32_x_by_0(%arg0 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %0 = vm.shl.i32 %arg0, 0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @shl_i32_const
  vm.func @shl_i32_const() -> i32 {
    // CHECK: %c16 = vm.const.i32 16 : i32
    // CHECK-NEXT: vm.return %c16 : i32
    %c1 = vm.const.i32 1 : i32
    %0 = vm.shl.i32 %c1, 4 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shr_i32_s_folds
vm.module @shr_i32_s_folds {
  // CHECK-LABEL: @shr_i32_s_0_by_y
  vm.func @shr_i32_s_0_by_y() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.shr.i32.s %zero, 4 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @shr_i32_s_x_by_0
  vm.func @shr_i32_s_x_by_0(%arg0 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %0 = vm.shr.i32.s %arg0, 0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @shr_i32_s_const
  vm.func @shr_i32_s_const() -> i32 {
    // CHECK: vm.const.i32 -134217728 : i32
    // CHECK-NEXT: vm.return %c
    %c = vm.const.i32 0x80000000 : i32
    %0 = vm.shr.i32.s %c, 4 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shr_i32_u_folds
vm.module @shr_i32_u_folds {
  // CHECK-LABEL: @shr_i32_u_0_by_y
  vm.func @shr_i32_u_0_by_y() -> i32 {
    // CHECK: %zero = vm.const.i32.zero : i32
    // CHECK-NEXT: vm.return %zero : i32
    %zero = vm.const.i32.zero : i32
    %0 = vm.shr.i32.u %zero, 4 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @shr_i32_u_x_by_0
  vm.func @shr_i32_u_x_by_0(%arg0 : i32) -> i32 {
    // CHECK: vm.return %arg0 : i32
    %0 = vm.shr.i32.u %arg0, 0 : i32
    vm.return %0 : i32
  }

  // CHECK-LABEL: @shr_i32_u_const
  vm.func @shr_i32_u_const() -> i32 {
    // CHECK: vm.const.i32 134217728 : i32
    // CHECK-NEXT: vm.return %c
    %c = vm.const.i32 0x80000000 : i32
    %0 = vm.shr.i32.u %c, 4 : i32
    vm.return %0 : i32
  }
}
