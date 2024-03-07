// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This is the legalization pattern definition file for CHLO to StableHLO.
// These are included in the populateDecompositionPatterns factory
// and should only include canonical expansions which are not actually
// ambiguous/different for various backends. Avoid patterns that are actually
// lowering to non-canonical forms.

include "mlir/IR/OpBase.td"
include "stablehlo/dialect/ChloOps.td"
include "stablehlo/dialect/StablehloOps.td"

class StableHLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<StableHLO_ComparisonDirectionAttr,
               "::mlir::stablehlo::ComparisonDirection::" # enumStr>;

class ConstantLike<string value> : NativeCodeCall<
    "::mlir::iree_compiler::stablehlo::getConstantLike($_builder, $_loc, " # value # ", $0)">;

def ComplexElementType : Type<
  CPred<"isa<ComplexType>(cast<ShapedType>($_self).getElementType())">,
  "Complex element type">;

def NonComplexElementType : Type<
  CPred<"!isa<ComplexType>(cast<ShapedType>($_self).getElementType())">,
  "Non-complex element type">;

def ConstantLikeMaxFiniteValue : NativeCodeCall<
    "::mlir::iree_compiler::stablehlo::getConstantLikeMaxFiniteValue($_builder, $_loc, $0)">;

def ConstantLikePosInfValue : NativeCodeCall<
    "::mlir::iree_compiler::stablehlo::getConstantLikeInfValue($_builder, $_loc, $0, /*negative=*/false)">;

def ConstantLikeNegInfValue : NativeCodeCall<
    "::mlir::iree_compiler::stablehlo::getConstantLikeInfValue($_builder, $_loc, $0, /*negative=*/true)">;

//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

// Expand acos for non-complex arguments to MHLO dialect as follows:
//   acos(x) = 2 * atan2(sqrt(1 - x^2), (1 + x))  if x != -1
//           = pi                                 if x == -1
//
// TODO(b/237376133): Support operands with complex element types separately
// using the following formula.
//   acos(x) = -(i * log(x + i * sqrt((1 + x) * (1 - x))))
def : Pat<(CHLO_AcosOp NonComplexElementType:$input),
  (StableHLO_SelectOp
    (StableHLO_CompareOp
      $input,
      (ConstantLike<"-1"> $input),
      StableHLO_ComparisonDirectionValue<"NE">,
      (STABLEHLO_DEFAULT_COMPARISON_TYPE)
    ),
    (StableHLO_MulOp
      (ConstantLike<"2"> $input),
      (StableHLO_Atan2Op
        (StableHLO_SqrtOp
          (StableHLO_SubtractOp
            (ConstantLike<"1"> $input),
            (StableHLO_MulOp $input, $input)
          )
        ),
        (StableHLO_AddOp
          (ConstantLike<"1"> $input),
          $input
        )
      )
    ),
    (ConstantLike<"M_PI"> $input)
  )>;

// Expand acosh to MHLO dialect as follows:
//   acosh(x) = log(x + sqrt(x^2 - 1))      if x >= -1
//            = log(x + sqrt((x+1)*(x-1)))
//   acosh(x) = nan                         if x < -1
//
// If x^2 will overflow, we approximate sqrt(x^2 - 1) == x and compute as
// log(2*x) = log(2) + log(x).  (Note this works because negative x never
// overflows; x < -1 simply yields nan.
def : Pat<(CHLO_AcoshOp NonComplexElementType:$input),
  (StableHLO_SelectOp
    (StableHLO_CompareOp
      $input,
      (ConstantLike<"-1"> $input),
      StableHLO_ComparisonDirectionValue<"LT">,
      (STABLEHLO_DEFAULT_COMPARISON_TYPE)
    ),
    (ConstantLike<"NAN"> $input),
    (StableHLO_SelectOp
      (StableHLO_CompareOp
        $input,
        (StableHLO_SqrtOp
          (ConstantLikeMaxFiniteValue $input)
        ),
        StableHLO_ComparisonDirectionValue<"GE">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)
      ),
      (StableHLO_AddOp
        (StableHLO_LogOp $input),
        (StableHLO_LogOp
          (ConstantLike<"2"> $input)
        )
      ),
      (StableHLO_LogOp
        (StableHLO_AddOp
          $input,
          (StableHLO_SqrtOp
            (StableHLO_MulOp
              (StableHLO_AddOp
                (ConstantLike<"1"> $input),
                $input
              ),
              (StableHLO_AddOp
                (ConstantLike<"-1"> $input),
                $input
              )
            )
          )
        )
      )
    )
  )>;

// Expand acosh for complex arguments to MHLO dialect as
//   acosh(x) = log(x + sqrt((x+1)*(x-1)))
//
// Per tensorflow/compiler/xla/client/lib/math.cc at the time of writing:
// "For now, we ignore the question of overflow if x is a
// complex type, because we don't yet have exhaustive tests for complex trig
// functions".
def : Pat<(CHLO_AcoshOp ComplexElementType:$input),
  (StableHLO_LogOp
    (StableHLO_AddOp
      $input,
      (StableHLO_SqrtOp
        (StableHLO_MulOp
          (StableHLO_AddOp
            $input,
            (ConstantLike<"1"> $input)
          ),
          (StableHLO_SubtractOp
            $input,
            (ConstantLike<"1"> $input)
          )
        )
      )
    )
  )>;


// Expand asin to MHLO dialect as follows:
//   asin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))
def : Pat<(CHLO_AsinOp $input),
  (StableHLO_MulOp
    (ConstantLike<"2"> $input),
    (StableHLO_Atan2Op
      $input,
      (StableHLO_AddOp
        (ConstantLike<"1"> $input),
        (StableHLO_SqrtOp
          (StableHLO_SubtractOp
            (ConstantLike<"1"> $input),
            (StableHLO_MulOp $input, $input)
          )
        )
      )
    )
  )>;

// Expand asinh for non-complex arguments to MHLO dialect as
//   asinh(x) = log(x + sqrt(x^2 + 1))
//
// If x^2 will overflow and x is positive, we can approximate x + sqrt(x^2 + 1)
// as 2*x and return log(2) + log(x).
//
// For small x, sqrt(x^2 + 1) will evaluate to 1 due to floating point
// arithmetic. However, we would like to retain the low order term of this,
// which is around 0.5 * x^2 using a binomial expansion.
// Let z = sqrt(a^2 + 1)
// The following rewrite retains the lower order term.
// log(a + sqrt(a^2 + 1))
//   = log((a + sqrt(a^2 + 1)) * (1 + sqrt(a^2 + 1)) / (1 + sqrt(a^2 + 1)))
//   = log((a + a^2 + 1 + a * z + z) / (1 + z))
//   = log(1 + a + a^2 / (1 + z))
//   = log(1 + a + a^2 / (1 + sqrt(a^2 + 1)))
//
// If x is negative, the above would give us some trouble; we can't approximate
// the result as x + abs(x) = 0 but we are saved by the fact that asinh(-x) =
// -asinh(x).
def : Pat<(CHLO_AsinhOp NonComplexElementType:$input),
  (StableHLO_MulOp
    (StableHLO_SignOp $input),
    (StableHLO_SelectOp
      (StableHLO_CompareOp
        (StableHLO_AbsOp $input),
        (StableHLO_SqrtOp
          (ConstantLikeMaxFiniteValue $input)
        ),
        StableHLO_ComparisonDirectionValue<"GE">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)
      ),
      (StableHLO_AddOp
        (StableHLO_LogOp
          (StableHLO_AbsOp $input)
        ),
        (StableHLO_LogOp
          (ConstantLike<"2"> $input)
        )
      ),
      (StableHLO_SelectOp
        (StableHLO_CompareOp
          (StableHLO_AbsOp $input),
          (ConstantLike<"1"> $input),
          StableHLO_ComparisonDirectionValue<"LE">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE)
        ),
        (StableHLO_Log1pOp
          (StableHLO_AddOp
            (StableHLO_AbsOp $input),
            (StableHLO_MulOp
              (StableHLO_AbsOp $input),
              (StableHLO_DivOp
                (StableHLO_AbsOp $input),
                (StableHLO_AddOp
                  (ConstantLike<"1"> $input),
                  (StableHLO_SqrtOp
                    (StableHLO_AddOp
                      (StableHLO_MulOp
                        (StableHLO_AbsOp $input),
                        (StableHLO_AbsOp $input)
                      ),
                      (ConstantLike<"1"> $input)
                    )
                  )
                )
              )
            )
          )
        ),
        (StableHLO_LogOp
          (StableHLO_AddOp
            (StableHLO_AbsOp $input),
            (StableHLO_SqrtOp
              (StableHLO_AddOp
                (StableHLO_MulOp
                  (StableHLO_AbsOp $input),
                  (StableHLO_AbsOp $input)
                ),
                (ConstantLike<"1"> $input)
              )
            )
          )
        )
      )
    )
  )>;

// Expand asinh for complex arguments to MHLO dialect as
//   asinh(x) = log(x + sqrt(x^2 + 1))
//
// Per tensorflow/compiler/xla/client/lib/math.cc at the time of writing:
// "For now, we ignore the question of overflow if x is a
// complex type, because we don't yet have exhaustive tests for complex trig
// functions".
def : Pat<(CHLO_AsinhOp ComplexElementType:$input),
  (StableHLO_LogOp
    (StableHLO_AddOp
      $input,
      (StableHLO_SqrtOp
        (StableHLO_AddOp
          (StableHLO_MulOp $input, $input),
          (ConstantLike<"1"> $input)
        )
      )
    )
  )>;

// Express `atan` as
//   atan(x) = atan2(x, 1)
def : Pat<(CHLO_AtanOp $input),
  (StableHLO_Atan2Op
    $input,
    (ConstantLike<"1"> $input)
  )>;

// Express `atanh` for non-complex arguments as follows:
//   atanh(x) = 0.5 * log((1 + x) / (1 - x)) if abs(x) <= 1
//   atanh(x) = nan                          otherwise
def : Pat<(CHLO_AtanhOp NonComplexElementType:$input),
  (StableHLO_SelectOp
    (StableHLO_CompareOp
      (StableHLO_AbsOp $input),
      (ConstantLike<"1"> $input),
      StableHLO_ComparisonDirectionValue<"GT">,
      (STABLEHLO_DEFAULT_COMPARISON_TYPE)
    ),
    (ConstantLike<"NAN"> $input),
    (StableHLO_MulOp
      (StableHLO_SubtractOp
        (StableHLO_Log1pOp $input),
        (StableHLO_Log1pOp
          (StableHLO_NegOp $input)
        )
      ),
      (ConstantLike<"0.5"> $input)
    )
  )>;

// Express `atanh` for complex arguments as follows:
//   atanh(x) = (log(1 + x) - log(1 + (-x))) * 0.5
//
// Per tensorflow/compiler/xla/client/lib/math.cc at the time of writing:
// "For now, we ignore the nan edge case for complex inputs,
// because we don't yet have exhaustive tests for complex trig functions".
def : Pat<(CHLO_AtanhOp ComplexElementType:$input),
  (StableHLO_MulOp
    (StableHLO_SubtractOp
      (StableHLO_Log1pOp $input),
      (StableHLO_Log1pOp
        (StableHLO_NegOp $input)
      )
    ),
    (ConstantLike<"0.5"> $input)
  )>;

// Express `conj` as
//   conj(x) = (re(x), -im(x)).
def : Pat<(CHLO_ConjOp $v),
          (StableHLO_ComplexOp (StableHLO_RealOp $v), (StableHLO_NegOp (StableHLO_ImagOp $v)))>;

// Express `is_inf` as
//   is_inf(x) = is_pos_inf(|x|)
def : Pat<(CHLO_IsInfOp NonComplexElementType:$input),
  (CHLO_IsPosInfOp
    (StableHLO_AbsOp $input)
  )>;

// Express `is_pos_inf` as
//   is_pos_inf(x) = (x == +inf)
def : Pat<(CHLO_IsPosInfOp NonComplexElementType:$input),
  (StableHLO_CompareOp
    $input,
    (ConstantLikePosInfValue $input),
    StableHLO_ComparisonDirectionValue<"EQ">,
    (STABLEHLO_DEFAULT_COMPARISON_TYPE)
  )>;

// Express `is_neg_inf` as
//   is_neg_inf(x) = (x == -inf)
def : Pat<(CHLO_IsNegInfOp NonComplexElementType:$input),
  (StableHLO_CompareOp
    $input,
    (ConstantLikeNegInfValue $input),
    StableHLO_ComparisonDirectionValue<"EQ">,
    (STABLEHLO_DEFAULT_COMPARISON_TYPE)
  )>;

// Express `tan` as
//   sine(x) / cosine(x)
def : Pat<(CHLO_TanOp NonComplexElementType:$input),
  (StableHLO_DivOp
    (StableHLO_SineOp $input),
    (StableHLO_CosineOp $input)
  )>;


// Express `tan(a + bi)` as
//   (tan(a) + i tanh(b)) / (1 - i tan(a) * tanh(b))
def : Pat<(CHLO_TanOp ComplexElementType:$input),
  (StableHLO_DivOp
    (StableHLO_ComplexOp
      (CHLO_TanOp:$tan (StableHLO_RealOp $input)),
      (StableHLO_TanhOp:$tanh (StableHLO_ImagOp $input))),
    (StableHLO_ComplexOp
      (ConstantLike<"1.0"> $tan),
      (StableHLO_NegOp (StableHLO_MulOp $tan, $tanh)))
  )>;
