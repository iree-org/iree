# RUN: %PYTHON -m mlir.dialects.iree_pydm.rtl.rtl_builder | iree-dialects-opt -canonicalize
# This test is only verifying that the runtime library builds and validates
# by passing it through opt.
