# RUN: %PYTHON %s

import mlir.ir
from mlir.dialects import iree_public

with mlir.ir.Context() as ctx:
  iree_public.register_iree_public_dialect(ctx)
