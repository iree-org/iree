# RUN: %PYTHON %s

import iree.compiler.ir
from iree.compiler.dialects import iree_input as iree_d

with iree.compiler.ir.Context() as ctx:
    iree_d.register_dialect()
