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

func @print_example_func(%arg0 : i32) attributes { iree.module.export } {
  %0 = "strings.i32_to_string"(%arg0) : (i32) -> !strings.string
  "strings.print"(%0) : (!strings.string) -> ()
  return
}

func @string_tensor_to_string(%arg0 : !strings.string_tensor) -> !strings.string attributes { iree.module.export, iree.abi.none } {
  %0 = "strings.string_tensor_to_string"(%arg0) : (!strings.string_tensor) -> (!strings.string)
  return %0 : !strings.string
}

func @to_string_tensor(%arg0 : !hal.buffer_view) -> !strings.string_tensor attributes { iree.module.export, iree.abi.none } {
  %0 = "strings.to_string_tensor"(%arg0) : (!hal.buffer_view) -> !strings.string_tensor
  return %0 : !strings.string_tensor
}
