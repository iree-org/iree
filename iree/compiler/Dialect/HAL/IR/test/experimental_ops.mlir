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

// Tests printing and parsing of experimental ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @shared_device
func @shared_device() -> !ireex.ref<!hal.device> {
  // CHECK: %dev = hal.ex.shared_device : !ireex.ref<!hal.device>
  %dev = hal.ex.shared_device : !ireex.ref<!hal.device>
  return %dev : !ireex.ref<!hal.device>
}

// -----

// CHECK-LABEL: @cache_executable
hal.executable @foo {}
func @cache_executable() -> !ireex.ref<!hal.executable> {
  %0 = "test_hal.device"() : () -> !ireex.ref<!hal.device>
  // CHECK: %exe = hal.ex.cache_executable %0, @foo : !ireex.ref<!hal.executable>
  %exe = hal.ex.cache_executable %0, @foo : !ireex.ref<!hal.executable>
  return %exe : !ireex.ref<!hal.executable>
}

// -----

// CHECK-LABEL: @submit_and_wait
func @submit_and_wait() {
  %0 = "test_hal.device"() : () -> !ireex.ref<!hal.device>
  %1 = "test_hal.command_buffer"() : () -> !ireex.ref<!hal.command_buffer>
  // CHECK: hal.ex.submit_and_wait %0, %1
  hal.ex.submit_and_wait %0, %1
  return
}
