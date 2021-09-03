# RUN: %PYTHON %s

import mlir.ir
from mlir.dialects import iree
from mlir.dialects import iree_pydm

with mlir.ir.Context() as ctx:
  iree.register_dialect()
  iree_pydm.register_dialect()

  # iree_pydm types.
  bool_t = iree_pydm.BoolType.get()
  typed_object_t = iree_pydm.ObjectType.get_typed(bool_t)
  untyped_object_t = iree_pydm.ObjectType.get()
