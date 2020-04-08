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

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"
include "iree/compiler/Dialect/HAL/IR/HALBase.td"

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
    Verifies that the operand contains a true value, which is represented by
    any non-zero integer.

    Issues a non-fatal failure if the verification fails.
  }];

  let arguments = (ins AnySignlessInteger:$operand);

  let assemblyFormat = "`(` $operand `)` attr-dict `:` type($operand)";
}


def CHECK_ExpectFalseOp : Op<CHECK_Dialect, "expect_false"> {
  let summary = [{Checks that the operand is false}];
  let description = [{
    Verifies that the operand contains a false value, which is represented by
    zero.

    Issues a non-fatal failure if the verification fails.
  }];

  let arguments = (ins AnySignlessInteger:$operand);

  let assemblyFormat = "`(` $operand `)` attr-dict `:` type($operand)";
}

def CHECK_ExpectAllTrueOp : Op<CHECK_Dialect, "expect_all_true"> {
  let summary = [{Checks that the operand contains only values that are true}];
  let description = [{
    Verifies that the operand contains true values, which are represented by any
    non-zero integer.

    Issues a non-fatal failure if the verification fails.
  }];

  let arguments =
    (ins AnyTypeOf<[HAL_BufferView, TensorOf<[AnySignlessInteger]>]>:$operand);

  let assemblyFormat = "`(` $operand `)` attr-dict `:` type($operand)";
}

// TODO(b/146898896): Consider a cmp op instead.
def CHECK_ExpectEqOp : Op<CHECK_Dialect, "expect_eq", [SameTypeOperands]> {
  let summary = [{Checks that the tensor or buffer view operands are equal}];
  let description = [{
    Verifies that the operands are exactly equal.

    Issues a non-fatal failure if the verification fails.
  }];

  let arguments = (ins
      AnyTypeOf<[HAL_BufferView, AnyTensor]>:$lhs,
      AnyTypeOf<[HAL_BufferView, AnyTensor]>:$rhs
  );

  let assemblyFormat = "`(` $lhs `,` $rhs `)` attr-dict `:` type($lhs)";
}

def CHECK_ExpectEqConstOp :
    Op<CHECK_Dialect, "expect_eq_const", [AllTypesMatch<["lhs", "value"]>]> {
  let summary = [{Checks that the tensor operand is equal to some constant}];
  let description =  [{
    Verifies that the tensor operand is exactly equal to a constant attribute.

    Issues a non-fatal failure if the verification fails.

    This op is just a convenience wrapper around the expect_eq op.
  }];

  let arguments = (ins
    AnyTensor:$lhs,
    ElementsAttr:$value
  );

  let hasCanonicalizer = 1;

  let assemblyFormat = "`(` $lhs `,` $value `)` attr-dict `:` type($lhs)";
}

def CHECK_ExpectAlmostEqOp :
    Op<CHECK_Dialect, "expect_almost_eq", [SameTypeOperands]> {
  let summary = [{Checks that the operands are almost equal}];
  let description = [{
    Verifies that the buffer view or tensor operands with float elements are
    almost equal to within an implementation-defined "reasonable" tolerance.

    Issues a non-fatal failure if the verification fails.
  }];

  let arguments = (ins
      AnyTypeOf<[HAL_BufferView, TensorOf<[AnyFloat]>]>:$lhs,
      AnyTypeOf<[HAL_BufferView, TensorOf<[AnyFloat]>]>:$rhs
  );

  let assemblyFormat = "`(` $lhs `,` $rhs `)` attr-dict `:` type($lhs)";
}

def CHECK_ExpectAlmostEqConstOp :
    Op<CHECK_Dialect,
       "expect_almost_eq_const", [AllTypesMatch<["lhs", "value"]>]> {
  let summary = [{Checks that the tensor operand is almost equal to some constant}];
  let description =  [{
    Verifies that the tensor operand with float elements is almost equal to the
    constant attribute within an implementation-defined "reasonable" tolerance.

    Issues a non-fatal failure if the verification fails.

    This op is just a convenience wrapper around the expect_almost_eq op.
  }];

  let arguments = (ins
    TensorOf<[AnyFloat]>:$lhs,
    ElementsAttr:$value
  );

  let hasCanonicalizer = 1;

  let assemblyFormat = "`(` $lhs `,` $value `)` attr-dict `:` type($lhs)";
}

#endif  // IREE_MODULES_CHECK_DIALECT_CHECK_OPS
