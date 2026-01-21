iree_add_all_subdirs()

# The plugin mechanism is not yet supported on Windows.
if (NOT WIN32)

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "invalid_options.txt"
    "invalid_plugin.c"
    "valid_plugin.c"
  TOOLS
    FileCheck
    iree-compile
    iree-opt
)

endif()
