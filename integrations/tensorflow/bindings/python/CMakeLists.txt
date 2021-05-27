# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
