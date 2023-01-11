# RUN: %PYTHON %s | FileCheck %s

import iree.compiler.ir as ir
import iree.compiler.dialects.transform.iree_structured as iree_structured_transform
import iree.compiler._mlir_libs._ireeDialects.transform


def constructAndPrintInModule(f):
  print("\nTEST:", f.__name__)
  with ir.Context() as ctx, ir.Location.unknown():
    iree.compiler._mlir_libs._ireeDialects.transform.register_dialect(ctx)
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
      f()
    print(module)
  return f


# CHECK-LABEL: TEST: testLowerVectorsOp
# CHECK: transform.lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1], transpose_avx2_lowering = false, transpose_lowering = "shuffle", unroll_vector_transfers = true}
@constructAndPrintInModule
def testLowerVectorsOp():
  op = iree_structured_transform.LowerVectorsOp(
      contraction_lowering="outerproduct",
      multireduction_lowering="innerparallel",
      split_transfers="linalg-copy",
      stages=[1],
      transpose_avx2_lowering=False,
      transpose_lowering="shuffle",
      unroll_vector_transfers=True)
