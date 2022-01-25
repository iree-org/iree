# RUN: %PYTHON %s

from iree.compiler.dialects.iree_pydm import rtl

# Ensures that we can compile the standard library to a SourceModule.
print(rtl.get_std_rtl_source_bundle())
