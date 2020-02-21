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

#ifndef IREE_MODULES_CHECK_DIALECT_CHECK_OPS
#define IREE_MODULES_CHECK_DIALECT_CHECK_OPS

include "mlir/IR/OpBase.td"

def CHECK_Dialect : Dialect {
  let name = "check";
  let cppNamespace = "IREE::Check";
  let summary = [{
    A dialect implementing test assertions for IREE modules.
  }];
}

def CHECK_ExpectTrueOp : Op<CHECK_Dialect, "expect_true"> {
  let summary = [{Checks that the operand is true}];
  let description = [{
    Verifies that the operand represents a true value, which is represented by
    any non-zero integer.
  }];

  let arguments = (ins AnySignlessInteger:$operand);

  let assemblyFormat = "`(` $operand `)` attr-dict `:` type($operand)";

  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ print$cppClass(p, *this); }];
}

#endif  // IREE_MODULES_CHECK_DIALECT_CHECK_OPS
