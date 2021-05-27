# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Overlays a subdirectory into the main python bindings directory.
function(_add_overlay_subdirectory dir)
  # Overlay binary directories onto the main bindings directory.
  set(_MAIN_PYTHON_DIR "${CMAKE_BINARY_DIR}/bindings/python")
  add_subdirectory(${dir} "${_MAIN_PYTHON_DIR}/${dir}")
endfunction()

if(${IREE_BUILD_TENSORFLOW_COMPILER} OR ${IREE_BUILD_TENSORFLOW_ALL})
  _add_overlay_subdirectory(iree/tools/tf)
endif()

if(${IREE_BUILD_TFLITE_COMPILER} OR ${IREE_BUILD_TENSORFLOW_ALL})
  _add_overlay_subdirectory(iree/tools/tflite)
endif()

if(${IREE_BUILD_XLA_COMPILER} OR ${IREE_BUILD_TENSORFLOW_ALL})
  _add_overlay_subdirectory(iree/tools/xla)
endif()

# TODO: Find another place for the TF support library.
# Pure python so can just always be generated.
_add_overlay_subdirectory(iree/tf/support)
