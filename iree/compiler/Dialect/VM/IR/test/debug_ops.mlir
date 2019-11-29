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

// Tests printing and parsing of debug ops.

// RUN: iree-opt -split-input-file %s | FileCheck %s --enable-var-scope --dump-input=fail

// CHECK-LABEL: @trace_args
vm.module @my_module {
  vm.func @trace_args(%arg0 : i32, %arg1 : i32) {
    // CHECK: vm.trace "event"(%arg0, %arg1) : i32, i32
    vm.trace "event"(%arg0, %arg1) : i32, i32
    vm.return
  }
}

// -----

// CHECK-LABEL: @print_args
vm.module @my_module {
  vm.func @print_args(%arg0 : i32, %arg1 : i32) {
    // CHECK: vm.print "message"(%arg0, %arg1) : i32, i32
    vm.print "message"(%arg0, %arg1) : i32, i32
    vm.return
  }
}

// -----

// CHECK-LABEL: @break_empty
vm.module @my_module {
  vm.func @break_empty() {
    // CHECK: vm.break ^bb1
    vm.break ^bb1
  ^bb1:
    vm.return
  }
}

// -----

// CHECK-LABEL: @break_args
vm.module @my_module {
  vm.func @break_args(%arg0 : i32) -> i32 {
    // CHECK: vm.break ^bb1(%arg0 : i32)
    vm.break ^bb1(%arg0 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  }
}

// -----

// CHECK-LABEL: @cond_break_empty
vm.module @my_module {
  vm.func @cond_break_empty(%arg0 : i32) {
    // CHECK: vm.cond_break %arg0, ^bb1
    vm.cond_break %arg0, ^bb1
  ^bb1:
    vm.return
  }
}

// -----

// CHECK-LABEL: @break_args
vm.module @my_module {
  vm.func @break_args(%arg0 : i32, %arg1 : i32) -> i32 {
    // CHECK: vm.cond_break %arg0, ^bb1(%arg1 : i32)
    vm.cond_break %arg0, ^bb1(%arg1 : i32)
  ^bb1(%0 : i32):
    vm.return %0 : i32
  }
}
