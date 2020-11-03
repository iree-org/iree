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

#-------------------------------------------------------------------------------
# Sanitizer configurations
#-------------------------------------------------------------------------------

# Note: we add these flags to the global CMake flags, not to IREE-specific
# variables such as IREE_DEFAULT_COPTS so that all symbols are consistently
# defined with the same sanitizer flags, including e.g. standard library
# symbols that might be used by both IREE and non-IREE (e.g. LLVM) code.

if(${IREE_ENABLE_ASAN})
  string(APPEND CMAKE_CXX_FLAGS " -fsanitize=address")
  string(APPEND CMAKE_C_FLAGS " -fsanitize=address")
endif()
if(${IREE_ENABLE_MSAN})
  string(APPEND CMAKE_CXX_FLAGS " -fsanitize=memory")
  string(APPEND CMAKE_C_FLAGS " -fsanitize=memory")
endif()
if(${IREE_ENABLE_TSAN})
  string(APPEND CMAKE_CXX_FLAGS " -fsanitize=thread")
  string(APPEND CMAKE_C_FLAGS " -fsanitize=thread")
endif()
if(ANDROID)
  # Work around https://github.com/android/ndk/issues/1088
  if(${IREE_ENABLE_ASAN} OR ${IREE_ENABLE_MSAN} OR ${IREE_ENABLE_TSAN})
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -fuse-ld=gold")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -fuse-ld=gold")
  endif()
endif()