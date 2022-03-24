# RUN: %PYTHON %s

import iree.compiler.ir
from iree.compiler.dialects import iree_input as iree_d
from iree.compiler.dialects import iree_linalg_ext
from iree.compiler.dialects import iree_linalg_transform
from iree.compiler.dialects import iree_pydm as pydm_d

with iree.compiler.ir.Context() as ctx:
  iree_d.register_dialect()
  iree_linalg_ext.register_dialect()
  iree_linalg_transform.register_dialect()
  pydm_d.register_dialect()

  # iree_pydm types.
  bool_t = pydm_d.BoolType.get()
  typed_object_t = pydm_d.ObjectType.get_typed(bool_t)
  untyped_object_t = pydm_d.ObjectType.get()
