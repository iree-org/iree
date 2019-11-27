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

// Tests printing and parsing of control flow ops.

// RUN: iree-opt -split-input-file %s | FileCheck %s --enable-var-scope --dump-input=fail

// CHECK-LABEL: @branch_empty
vm.module @my_module {
  vm.func @branch_empty() {
    // CHECK: vm.br ^bb1
    vm.br ^bb1
  ^bb1:
    vm.return
  }
}

// -----

// CHECK-LABEL: @branch_args
vm.module @my_module {
  vm.func @branch_args(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.br ^bb1(%arg0, %arg1 : i32, i32)
    vm.br ^bb1(%arg0, %arg1 : i32, i32)
  ^bb1(%0 : i32, %1 : i32):
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @cond_branch_empty
vm.module @my_module {
  vm.func @cond_branch_empty(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: vm.cond_br %arg0, ^bb1, ^bb2
    vm.cond_br %arg0, ^bb1, ^bb2
  ^bb1:
    vm.return %arg1 : i32
  ^bb2:
    vm.return %arg2 : i32
  }
}

// -----

// CHECK-LABEL: @cond_branch_args
vm.module @my_module {
  vm.func @cond_branch_args(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
    // CHECK: vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
    vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  ^bb2(%1 : i32):
    vm.return %1 : i32
  }
}

// -----

// CHECK-LABEL: @call_fn
vm.module @my_module {
  vm.import @import_fn(%arg0 : i32) -> i32
  vm.func @call_fn(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.call @import_fn(%arg0) : (i32) -> i32
    %0 = vm.call @import_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @call_variadic_but_not_really
vm.module @my_module {
  vm.import @import_fn(%arg0 : i32) -> i32
  vm.func @call_variadic_but_not_really(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.call.variadic @import_fn(%arg0) : (i32) -> i32
    %0 = vm.call.variadic @import_fn(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @call_variadic_empty
vm.module @my_module {
  vm.import @import_fn(%arg0 : i32, %arg1 : !ireex.ref<!hal.buffer>...) -> i32
  vm.func @call_variadic_empty(%arg0 : i32) -> i32 {
    // CHECK: %0 = vm.call.variadic @import_fn(%arg0, []) : (i32, !ireex.ref<!hal.buffer>...) -> i32
    %0 = vm.call.variadic @import_fn(%arg0, []) : (i32, !ireex.ref<!hal.buffer>...) -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @call_variadic
vm.module @my_module {
  vm.import @import_fn(%arg0 : i32, %arg1 : !ireex.ref<!hal.buffer>...) -> i32
  vm.func @call_variadic(%arg0 : i32, %arg1 : !ireex.ref<!hal.buffer>) -> i32 {
    // CHECK: %0 = vm.call.variadic @import_fn(%arg0, [%arg1, %arg1]) : (i32, !ireex.ref<!hal.buffer>...) -> i32
    %0 = vm.call.variadic @import_fn(%arg0, [%arg1, %arg1]) : (i32, !ireex.ref<!hal.buffer>...) -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @call_variadic_multiple
vm.module @my_module {
  vm.import @import_fn(%arg0 : i32, %arg1 : !ireex.ref<!hal.buffer>...) -> i32
  vm.func @call_variadic_multiple(%arg0 : i32, %arg1 : !ireex.ref<!hal.buffer>) -> i32 {
    // CHECK: %0 = vm.call.variadic @import_fn(%arg0, [%arg1, %arg1], [%arg1]) : (i32, !ireex.ref<!hal.buffer>..., !ireex.ref<!hal.buffer>...) -> i32
    %0 = vm.call.variadic @import_fn(%arg0, [%arg1, %arg1], [%arg1]) : (i32, !ireex.ref<!hal.buffer>..., !ireex.ref<!hal.buffer>...) -> i32
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @call_variadic_no_results
vm.module @my_module {
  vm.import @import_fn(%arg0 : i32, %arg1 : !ireex.ref<!hal.buffer>...)
  vm.func @call_variadic_no_results(%arg0 : i32, %arg1 : !ireex.ref<!hal.buffer>) {
    // CHECK: vm.call.variadic @import_fn(%arg0, [%arg1, %arg1], [%arg1]) : (i32, !ireex.ref<!hal.buffer>..., !ireex.ref<!hal.buffer>...)
    vm.call.variadic @import_fn(%arg0, [%arg1, %arg1], [%arg1]) : (i32, !ireex.ref<!hal.buffer>..., !ireex.ref<!hal.buffer>...)
    vm.return
  }
}

// -----

// CHECK-LABEL: @return_empty
vm.module @my_module {
  vm.func @return_empty() {
    // CHECK: vm.return
    vm.return
  }
}

// -----

// CHECK-LABEL: @return_args
vm.module @my_module {
  vm.func @return_args(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
    // CHECK: vm.return %arg0, %arg1 : i32, i32
    vm.return %arg0, %arg1 : i32, i32
  }
}

// -----

// CHECK-LABEL: @yield
vm.module @my_module {
  vm.func @yield() {
    // CHECK: vm.yield
    vm.yield
    vm.return
  }
}
