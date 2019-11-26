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

// Tests printing and parsing of arithmetic ops.

// RUN: iree-opt -split-input-file %s | FileCheck %s --enable-var-scope --dump-input=fail

// CHECK-LABEL: @add_i32
vm.module @my_module {
  vm.func @add_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.add.i32 %arg0, %arg1 : i32
    %0 = vm.add.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @sub_i32
vm.module @my_module {
  vm.func @sub_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.sub.i32 %arg0, %arg1 : i32
    %0 = vm.sub.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @mul_i32
vm.module @my_module {
  vm.func @mul_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.mul.i32 %arg0, %arg1 : i32
    %0 = vm.mul.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @div_i32_s
vm.module @my_module {
  vm.func @div_i32_s(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.div.i32.s %arg0, %arg1 : i32
    %0 = vm.div.i32.s %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @div_i32_u
vm.module @my_module {
  vm.func @div_i32_u(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.div.i32.u %arg0, %arg1 : i32
    %0 = vm.div.i32.u %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @rem_i32_s
vm.module @my_module {
  vm.func @rem_i32_s(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.rem.i32.s %arg0, %arg1 : i32
    %0 = vm.rem.i32.s %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @rem_i32_u
vm.module @my_module {
  vm.func @rem_i32_u(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.rem.i32.u %arg0, %arg1 : i32
    %0 = vm.rem.i32.u %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @not_i32
vm.module @my_module {
  vm.func @not_i32(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.not.i32 %arg0 : i32
    %0 = vm.not.i32 %arg0 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @and_i32
vm.module @my_module {
  vm.func @and_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.and.i32 %arg0, %arg1 : i32
    %0 = vm.and.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @or_i32
vm.module @my_module {
  vm.func @or_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.or.i32 %arg0, %arg1 : i32
    %0 = vm.or.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @xor_i32
vm.module @my_module {
  vm.func @xor_i32(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: %0 = vm.xor.i32 %arg0, %arg1 : i32
    %0 = vm.xor.i32 %arg0, %arg1 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shl_i32
vm.module @my_module {
  vm.func @shl_i32(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.shl.i32 %arg0, 2 : i32
    %0 = vm.shl.i32 %arg0, 2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shr_i32_s
vm.module @my_module {
  vm.func @shr_i32_s(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.shr.i32.s %arg0, 2 : i32
    %0 = vm.shr.i32.s %arg0, 2 : i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @shr_i32_u
vm.module @my_module {
  vm.func @shr_i32_u(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.shr.i32.u %arg0, 2 : i32
    %0 = vm.shr.i32.u %arg0, 2 : i32
    vm.return %0 : i32
  }
}
