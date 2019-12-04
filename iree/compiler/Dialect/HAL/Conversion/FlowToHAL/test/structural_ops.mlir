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

// RUN: iree-opt -split-input-file -iree-convert-flow-to-hal %s | IreeFileCheck %s

// CHECK-LABEL: func @tensorIO(%arg0: !ireex.ref<!hal.buffer>) -> !ireex.ref<!hal.buffer>
func @tensorIO(%arg0 : tensor<1x1xi32>) -> tensor<1x1xi32> {
  // CHECK-NEXT: br ^bb1(%arg0 : !ireex.ref<!hal.buffer>)
  br ^bb1(%arg0 : tensor<1x1xi32>)
// CHECK-NEXT: ^bb1([[BB0:%.+]]: !ireex.ref<!hal.buffer>)
^bb1(%0 : tensor<1x1xi32>):
  // CHECK-NEXT: return [[BB0]] : !ireex.ref<!hal.buffer>
  return %0 : tensor<1x1xi32>
}
