# RUN: %PYTHON %s

import iree.compiler.ir
from iree.compiler.dialects import iree_input as iree_d
from iree.compiler.dialects import iree_linalg_ext

with iree.compiler.ir.Context() as ctx:
    iree_d.register_dialect()
    iree_linalg_ext.register_dialect()
