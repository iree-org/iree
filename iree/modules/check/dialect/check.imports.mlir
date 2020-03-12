// Copyright 2020 Google LLC
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

vm.module @check {

vm.import @expect_true(
  %operand : i32
)

vm.import @expect_false(
  %operand : i32
)

vm.import @expect_all_true(
  %operand : !vm.ref<!hal.buffer_view>,
)

vm.import @expect_eq(
  %lhs : !vm.ref<!hal.buffer_view>,
  %rhs : !vm.ref<!hal.buffer_view>
)

}  // vm.module
