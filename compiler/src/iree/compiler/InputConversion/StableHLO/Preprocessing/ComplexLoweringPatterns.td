// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This is the legalization pattern that converts complex operations into
// equivalent real value operations.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "stablehlo/dialect/StablehloOps.td"

class ConstantSplat<string value> : NativeCodeCall<
    "::mlir::iree_compiler::stablehlo::getSplat(&$_builder, $0, " # value # ")">;

//===----------------------------------------------------------------------===//
// Folds
//===----------------------------------------------------------------------===//

// real(complex x, y) -> x
def : Pat<(StableHLO_RealOp (StableHLO_ComplexOp $lhs, $rhs)),
          (replaceWithValue $lhs)>;

// imag(complex x, y) -> y
def : Pat<(StableHLO_ImagOp (StableHLO_ComplexOp $lhs, $rhs)),
          (replaceWithValue $rhs)>;

//===----------------------------------------------------------------------===//
// Binary op patterns.
//===----------------------------------------------------------------------===//

// Add and subtraction are elementwise and can be distributed across the real
// and imaginary components.
foreach elementwiseOp = [StableHLO_AddOp, StableHLO_SubtractOp] in
  def : Pat<(elementwiseOp HLO_ComplexTensor:$lhs,
             HLO_ComplexTensor:$rhs),
            (StableHLO_ComplexOp
              (elementwiseOp (StableHLO_RealOp $lhs), (StableHLO_RealOp $rhs)),
              (elementwiseOp (StableHLO_ImagOp $lhs), (StableHLO_ImagOp $rhs)))>;

// Complex multiplication results in a cross product multiplication between the
// real and imaginary components such that:
//   result.real = lhs.real * rhs.real - lhs.imag * rhs.imag
//   result.imag = lhs.imag * rhs.real + lhs.real * rhs.imag
def : Pat<(StableHLO_MulOp HLO_ComplexTensor:$lhs,
           HLO_ComplexTensor:$rhs),
          (StableHLO_ComplexOp
           (StableHLO_SubtractOp
            (StableHLO_MulOp
             (StableHLO_RealOp:$lhs_real $lhs),
             (StableHLO_RealOp:$rhs_real $rhs)),
            (StableHLO_MulOp
             (StableHLO_ImagOp:$lhs_imag $lhs),
             (StableHLO_ImagOp:$rhs_imag $rhs))),
           (StableHLO_AddOp
            (StableHLO_MulOp $lhs_real, $rhs_imag),
            (StableHLO_MulOp $lhs_imag, $rhs_real)))>;


// Division is performed by normalizing the denominator by multiplying by the
// conjugate of the rhs.
//   numerator = lhs * conj(rhs)
//   denominator = rhs * conj(rhs)
def : Pat<(StableHLO_DivOp HLO_ComplexTensor:$lhs, HLO_ComplexTensor:$rhs),
          (StableHLO_ComplexOp
            (StableHLO_DivOp
             (StableHLO_RealOp (StableHLO_MulOp:$num $lhs,
                          (StableHLO_ComplexOp:$conj
                           (StableHLO_RealOp $rhs),
                           (StableHLO_NegOp (StableHLO_ImagOp $rhs))))),
              (StableHLO_AddOp:$den
               (StableHLO_MulOp (StableHLO_RealOp $rhs), (StableHLO_RealOp $rhs)),
               (StableHLO_MulOp (StableHLO_ImagOp $rhs), (StableHLO_ImagOp $rhs)))),
            (StableHLO_DivOp (StableHLO_ImagOp $num), $den))>;

// Absolute value is evaluated as:
//   result = sqrt(val.real * val.real + val.imag * val.imag)
def : Pat<(StableHLO_AbsOp HLO_ComplexTensor:$val),
           (StableHLO_SqrtOp
             (StableHLO_AddOp
              (StableHLO_MulOp (StableHLO_RealOp:$real $val), $real),
              (StableHLO_MulOp (StableHLO_ImagOp:$imag $val), $imag)))>;

// Can deconstruct sin(a + ib) as follows:
//   sin(a) * cosh(b) + icos(a) * sinh(b)
//   sinh(b) = (e^x - e^-x) / 2
//   cosh(b) = (e^x + e^-x) / 2
def : Pat<(StableHLO_SineOp HLO_ComplexTensor:$val),
            (StableHLO_ComplexOp
              (StableHLO_DivOp
                (StableHLO_MulOp
                  (StableHLO_SineOp (StableHLO_RealOp:$real $val)),
                  (StableHLO_AddOp
                    (StableHLO_ExpOp:$exp (StableHLO_ImagOp:$imag $val)),
                    (StableHLO_ExpOp:$nexp (StableHLO_NegOp $imag)))),
                 (StableHLO_ConstantOp : $two (ConstantSplat<"2.0"> $real))),
              (StableHLO_DivOp
                (StableHLO_MulOp
                  (StableHLO_CosineOp $real),
                  (StableHLO_SubtractOp $exp, $nexp)), $two))>;

// Can deconstruct cos(a + ib) as follows:
//   cos(a) * cosh(b) - isin(a) * sinh(b)
//   sinh(b) = (e^x - e^-x) / 2
//   cosh(b) = (e^x + e^-x) / 2
def : Pat<(StableHLO_CosineOp HLO_ComplexTensor:$val),
            (StableHLO_ComplexOp
              (StableHLO_DivOp
                (StableHLO_MulOp
                  (StableHLO_CosineOp (StableHLO_RealOp:$real $val)),
                  (StableHLO_AddOp
                    (StableHLO_ExpOp:$exp (StableHLO_ImagOp:$imag $val)),
                    (StableHLO_ExpOp:$nexp (StableHLO_NegOp $imag)))),
                 (StableHLO_ConstantOp : $two (ConstantSplat<"2.0"> $real))),
              (StableHLO_DivOp
                (StableHLO_MulOp
                  (StableHLO_SineOp $real),
                  (StableHLO_SubtractOp $nexp, $exp)), $two))>;

// Exponential can be lowered to an exponential on the real component and a
// sum of sinusoids of the imaginary component, which equates to a normal
// exponential operator multiplied by Euler's formula.
//
// Exp(a + ib) = Exp(a) * Exp(ib) = Exp(a) * Cos(b) + Exp(a) * iSin(b))
class StableHLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<StableHLO_ComparisonDirectionAttr, "::mlir::stablehlo::ComparisonDirection::" # enumStr>;

def : Pat<(StableHLO_ExpOp HLO_ComplexTensor:$val),
          (StableHLO_ComplexOp
           (StableHLO_MulOp
            (StableHLO_CosineOp (StableHLO_ImagOp:$imag $val)),
            (StableHLO_ExpOp:$exp (StableHLO_RealOp:$real $val))),
           (StableHLO_MulOp (StableHLO_SineOp $imag), $exp))>;

foreach pair = [[StableHLO_ComparisonDirectionValue<"NE">, StableHLO_OrOp],
                [StableHLO_ComparisonDirectionValue<"EQ">, StableHLO_AndOp]] in {
def : Pat<(StableHLO_CompareOp HLO_ComplexTensor:$lhs, HLO_ComplexTensor:$rhs, pair[0], $compare_type),
            (pair[1]
             (StableHLO_CompareOp (StableHLO_RealOp $lhs), (StableHLO_RealOp $rhs), pair[0], $compare_type),
             (StableHLO_CompareOp (StableHLO_ImagOp $lhs), (StableHLO_ImagOp $rhs), pair[0], $compare_type))>;
}
