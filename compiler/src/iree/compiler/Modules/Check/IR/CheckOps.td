// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_CHECK_DIALECT_CHECK_OPS
#define IREE_MODULES_CHECK_DIALECT_CHECK_OPS

include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/HAL/IR/HALBase.td"

def CHECK_Dialect : Dialect {
  let name = "check";
  let cppNamespace = "::mlir::iree_compiler::IREE::Check";
  let summary = [{
    A dialect implementing test assertions for IREE modules.
  }];
}

def CHECK_ExpectTrueOp : Op<CHECK_Dialect, "expect_true"> {
  let summary = [{Checks that the operand is true.}];
  let description = [{
    Verifies that the operand contains a true value, which is represented by
    any non-zero integer.

    Issues a non-fatal failure if the verification fails.

    ```mlir
    check.expect_true(%arg0) : i32
    ```
  }];

  let arguments = (ins AnySignlessInteger:$operand);

  let assemblyFormat = "`(` $operand `)` attr-dict `:` type($operand)";
}

def CHECK_ExpectFalseOp : Op<CHECK_Dialect, "expect_false"> {
  let summary = [{Checks that the operand is false.}];
  let description = [{
    Verifies that the operand contains a false value, which is represented by
    zero.

    Issues a non-fatal failure if the verification fails.

    ```mlir
    check.expect_false(%arg0) : i32
    ```
  }];

  let arguments = (ins AnySignlessInteger:$operand);

  let assemblyFormat = "`(` $operand `)` attr-dict `:` type($operand)";
}

def CHECK_ExpectAllTrueOp : Op<CHECK_Dialect, "expect_all_true"> {
  let summary = [{Checks that the operand contains only values that are true.}];
  let description = [{
    Verifies that the operand contains true values, which are represented by any
    non-zero integer.

    Issues a non-fatal failure if the verification fails.

    ```mlir
    check.expect_all_true<%device>(%arg0) : !hal.buffer_view
    check.expect_all_true(%arg1) : tensor<2x2xi32>
    ```
  }];

  let arguments = (ins
    Optional<HAL_Device>:$device,
    AnyTypeOf<[HAL_BufferView, TensorOf<[AnySignlessInteger]>]>:$operand
  );

  let assemblyFormat = [{
    (`` `<` $device^ `>`)?
    `` `(` $operand `)` attr-dict `:` type($operand)
  }];
}

def CHECK_ExpectEqOp :
    Op<CHECK_Dialect, "expect_eq", [AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = [{Checks that the tensor or buffer view operands are equal.}];
  let description = [{
    Verifies that the operands are exactly equal.

    Issues a non-fatal failure if the verification fails.

    ```mlir
    check.expect_eq(%arg0, %arg1) : tensor<5xi32>
    ```
  }];

  let arguments = (ins
    Optional<HAL_Device>:$device,
    AnyTypeOf<[HAL_BufferView, AnyTensor]>:$lhs,
    AnyTypeOf<[HAL_BufferView, AnyTensor]>:$rhs
  );

  let assemblyFormat = [{
    (`` `<` $device^ `>`)?
    `` `(` $lhs `,` $rhs `)` attr-dict `:` type($lhs)
  }];
}

def CHECK_ExpectEqConstOp :
    Op<CHECK_Dialect, "expect_eq_const", [AllTypesMatch<["lhs", "value"]>]> {
  let summary = [{Checks that the tensor operand is equal to some constant.}];
  let description =  [{
    Verifies that the tensor operand is exactly equal to a constant attribute.

    Issues a non-fatal failure if the verification fails.

    This op is just a convenience wrapper around the expect_eq op.

    ```mlir
    check.expect_eq_const(%arg0, dense<[1, 2]> : tensor<2xi32>) : tensor<2xi32>
    ```
  }];

  let arguments = (ins
    Optional<HAL_Device>:$device,
    AnyTensor:$lhs,
    ElementsAttr:$value
  );

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    (`` `<` $device^ `>`)?
    `` `(` $lhs `,` $value `)` attr-dict `:` type($lhs)
  }];
}

def CHECK_ExpectAlmostEqOp :
    Op<CHECK_Dialect, "expect_almost_eq", [AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = [{Checks that the operands are almost equal.}];
  let description = [{
    Verifies that the buffer view or tensor operands with float elements satisfy
    the Numpy-style fuzzy-comparision condition with parameters `atol`,
    `rtol`, defined exactly as in NumPy isclose():
    https://github.com/numpy/numpy/blob/7297f3117d84745bfade1e2f9aec3531e5917500/numpy/_core/numeric.py#L2447-L2449

    The condition being verified on each lhs and rhs value is:
      lhs == rhs || (isfinite(rhs) && abs(lhs - rhs) <= atol + rtol * abs(rhs)).
    Note that the `lhs == rhs` part is needed for the case (lhs=+inf, rhs+inf)
    to return true. Indeed, in that case, lhs-rhs is NaN.

    Issues a non-fatal failure if the verification fails.

    The `atol`, `rtol` parameters may be omitted, in which case some default
    value is used. The default `atol` is nonzero, while the default `rtol` is
    zero, which makes these comparision behave closer to exact comparisons as
    the values being compared get large.

    This default behavior is supported for legacy compatibility and to support
    some use cases that legitimately don't care, but the majority of use cases
    should care and so should provide explicit `atol`, `rtol` values.

    ```mlir
    check.expect_almost_eq(%arg0, %arg1, atol 1.0e-2, rtol 1.0e-3) : tensor<5xf32>
    ```
  }];

  let arguments = (ins
    Optional<HAL_Device>:$device,
    AnyTypeOf<[HAL_BufferView, TensorOf<[AnyFloat]>]>:$lhs,
    AnyTypeOf<[HAL_BufferView, TensorOf<[AnyFloat]>]>:$rhs,
    DefaultValuedAttr<F32Attr, "1.e-4f">:$atol,
    DefaultValuedAttr<F32Attr, "0.f">:$rtol
  );

  let assemblyFormat = [{
    (`` `<` $device^ `>`)?
    `` `(` $lhs `,` $rhs (`` `,` `atol` $atol^)? (`` `,` `rtol` $rtol^)? `)`
    attr-dict `:` type($lhs)
  }];
}

def CHECK_ExpectAlmostEqConstOp :
    Op<CHECK_Dialect,
       "expect_almost_eq_const", [AllTypesMatch<["lhs", "value"]>]> {
  let summary = [{Checks that the tensor operand is almost equal to some constant.}];
  let description =  [{
    This op is just a convenience wrapper around the expect_almost_eq op.

    Verifies that the buffer view or tensor operands with float elements satisfy
    the Numpy-style fuzzy-comparision condition with pararameters `atol`,
    `rtol`. More details in the description of `expect_almost_eq`.

    Issues a non-fatal failure if the verification fails.

    ```mlir
    check.expect_almost_eq_const(%const0, dense<[0.999999, 2.0]> : tensor<5xf32>, atol 1.0e-2, rtol 1.0e-3) : tensor<5xf32>
    ```
  }];

  let arguments = (ins
    Optional<HAL_Device>:$device,
    TensorOf<[AnyFloat]>:$lhs,
    ElementsAttr:$value,
    DefaultValuedAttr<F32Attr, "1.e-4f">:$atol,
    DefaultValuedAttr<F32Attr, "0.f">:$rtol
  );

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    (`` `<` $device^ `>`)?
    `` `(` $lhs `,` $value (`` `,` `atol` $atol^)? (`` `,` `rtol` $rtol^)? `)`
    attr-dict `:` type($lhs)
  }];
}

#endif  // IREE_MODULES_CHECK_DIALECT_CHECK_OPS
