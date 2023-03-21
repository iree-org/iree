# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# See iree_compiler_plugin.cmake for the properties that this file handles.

get_property(_CONFIG_LINKED_PLUGIN_IDS GLOBAL PROPERTY IREE_COMPILER_LINKED_PLUGIN_IDS)
get_property(_CONFIG_LINKED_PLUGIN_LIBS GLOBAL PROPERTY IREE_COMPILER_LINKED_PLUGIN_LIBS)

# Generate an include file with one "HANDLE_PLUGIN_ID(plugin_id)" macro
# evaluation per linked plugin id.
set(_STATIC_LINKED_PLUGINS_CONTENTS)
foreach(_plugin_id ${_CONFIG_LINKED_PLUGIN_IDS})
  string(APPEND _STATIC_LINKED_PLUGINS_CONTENTS "HANDLE_PLUGIN_ID(${_plugin_id})\n")
endforeach()
file(GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/StaticLinkedPlugins.inc"
  CONTENT "${_STATIC_LINKED_PLUGINS_CONTENTS}"
)

# Uncomment for debugging.
# message(STATUS "*** STATIC LINKING PLUGIN LIBRARIES: ${_CONFIG_LINKED_PLUGIN_LIBS}")
iree_cc_library(
  NAME
    StaticLinkedPlugins
  HDRS
    StaticLinkedPlugins.inc
  DEPS
    ${_CONFIG_LINKED_PLUGIN_LIBS}
)
