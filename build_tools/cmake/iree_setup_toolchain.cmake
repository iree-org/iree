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

function(iree_append value)
  foreach(variable ${ARGN})
    set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
  endforeach(variable)
endfunction()

if(IREE_ENABLE_LLD)
  if (IREE_USE_LINKER)
    message(FATAL_ERROR "IREE_ENABLE_LLD and IREE_USE_LINKER can't be set at the same time")
  endif()
  set(IREE_USE_LINKER "lld")
endif()

if(IREE_USE_LINKER)
  set(IREE_LINKER_FLAG "-fuse-ld=${IREE_USE_LINKER}")

  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS "${IREE_LINKER_FLAG}")
  check_cxx_source_compiles("int main() { return 0; }" CXX_SUPPORTS_CUSTOM_LINKER)
  if(NOT CXX_SUPPORTS_CUSTOM_LINKER)
    message(FATAL_ERROR "Compiler does not support '-fuse-ld=${IREE_USE_LINKER}'")
  endif()

  iree_append("-fuse-ld=${IREE_USE_LINKER}"
    CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
endif()
