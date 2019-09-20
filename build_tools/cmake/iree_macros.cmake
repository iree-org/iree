# Copyright 2019 Google LLC
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

# Sets ${PACKAGE_NS} to the IREE-root relative package name in C++ namespace
# format (::).
#
# Example when called from iree/base/CMakeLists.txt:
#   iree::base
function(iree_package_ns PACKAGE_NS)
    string(REPLACE ${IREE_ROOT_DIR} "" _PACKAGE ${CMAKE_CURRENT_LIST_DIR})
    string(SUBSTRING ${_PACKAGE} 1 -1 _PACKAGE)
    string(REPLACE "/" "::" _PACKAGE_NS ${_PACKAGE})
    set(${PACKAGE_NS} ${_PACKAGE_NS} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_NAME} to the IREE-root relative package name.
#
# Example when called from iree/base/CMakeLists.txt:
#   iree_base
function(iree_package_name PACKAGE_NAME)
    iree_package_ns(_PACKAGE_NS)
    string(REPLACE "::" "_" _PACKAGE_NAME ${_PACKAGE_NS})
    set(${PACKAGE_NAME} ${_PACKAGE_NAME} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_PATH} to the IREE-root relative package path.
#
# Example when called from iree/base/CMakeLists.txt:
#   iree/base
function(iree_package_path PACKAGE_PATH)
    iree_package_ns(_PACKAGE_NS)
    string(REPLACE "::" "/" PACKAGE_PATH ${_PACKAGE_NS})
    set(${PACKAGE_PATH} ${_PACKAGE_PATH} PARENT_SCOPE)
endfunction()
