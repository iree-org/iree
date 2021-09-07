# RUN: %PYTHON %s | iree-dialects-opt -split-input-file | FileCheck --enable-var-scope --dump-input-filter=all %s

from typing import List
from mlir.dialects.iree_pydm.importer import *
from mlir.dialects.iree_pydm.importer.test_util import *

from mlir.dialects import iree_pydm as d
from mlir import ir

################################################################################
# Pyfunc intrinsics
################################################################################


@def_pyfunc_intrinsic(symbol="__return_one")
def intrinsic_return_one() -> int:
  return 1


@def_pyfunc_intrinsic(symbol="__return_first_true")
def intrinsic_return_first_true(a: int, b: int) -> int:
  return a or b


# CHECK-LABEL: @test_intrinsic_function_no_args
# CHECK: dynamic_call @__return_one() : () -> (!iree_pydm.exception_result, !iree_pydm.object)
# CHECK: func private @__return_one()
@test_import_global
def test_intrinsic_function_no_args():
  value = intrinsic_return_one()
  return value


# CHECK-LABEL: @test_intrinsic_function_double_call
# No need to check anything: verifier will fail if double emitted.
@test_import_global
def test_intrinsic_function_double_call():
  value = intrinsic_return_one()
  value2 = intrinsic_return_one()
  return value


# CHECK-LABEL: @test_intrinsic_function_args
# CHECK: %[[ZERO:.*]] = constant 0 : i64 -> !iree_pydm.integer
# CHECK: %[[ONE:.*]] = constant 1 : i64 -> !iree_pydm.integer
# CHECK: dynamic_call @__return_first_true(%[[ZERO]], %[[ONE]]) : (!iree_pydm.integer, !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.object)
# CHECK: func private @__return_first_true
@test_import_global
def test_intrinsic_function_args():
  value = intrinsic_return_first_true(0, 1)
  return value


################################################################################
# IR macro intrinsics
################################################################################


@def_ir_macro_intrinsic
def macro_return_none(stage: ImportStage) -> ir.Value:
  return d.NoneOp(d.NoneType.get()).result


# Boxing isn't load bearing here: It is just something we can do/test.
@def_ir_macro_intrinsic
def macro_box_arg(stage: ImportStage, arg: ir.Value) -> ir.Value:
  return stage.ic.box(arg)


# CHECK-LABEL: @test_intrinsic_macro_no_args
# CHECK: %[[ONE:.*]] = constant 1
# CHECK: box %[[ONE]] : !iree_pydm.integer -> !iree_pydm.object<!iree_pydm.integer>
@test_import_global
def test_intrinsic_macro_no_args() -> int:
  return macro_box_arg(1)


################################################################################
# Test multi func intrinsic.
# There is nothing special about a logical not. It is just something we can
# test.
################################################################################
@def_pyfunc_intrinsic(symbol="__logical_not_bool")
def logical_not_bool(x: bool) -> bool:
  return not x


@def_pyfunc_intrinsic(symbol="__logical_not_generic")
def logical_not_generic(x):
  return not x


logical_not = def_pattern_call_intrinsic(match_generic=[logical_not_generic],
                                         match_specific=[logical_not_bool])


# CHECK-LABEL: @test_pattern_call
# CHECK: %[[TRUE:.*]] = constant true
# CHECK: pattern_match_call(%[[TRUE]]) : (!iree_pydm.bool) -> (!iree_pydm.exception_result, !iree_pydm.object)
# CHECK-SAME:   matching generic [@__logical_not_generic] specific [@__logical_not_bool]
# CHECK-DAG: func private @__logical_not_generic
# CHECK-DAG: func private @__logical_not_bool
@test_import_global
def test_pattern_call():
  return logical_not(True)
