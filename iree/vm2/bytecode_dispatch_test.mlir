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

// These test functions are called by the bytecode_dispatch_test.cc runner.
// The prefix of fail_ can be used to denote that the test is expected to fail
// (error returned from dispatch).
vm.module @bytecode_dispatch_test {
  // Tests that an empty function (0 args, 0 results, 0 ops) works.
  vm.export @empty
  vm.func @empty() {
    vm.return
  }

  // TODO(benvanik): more tests.
}
