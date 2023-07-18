# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_add_all_subidrs
#
# CMake macro to add all subdirectories of the current directory that contain
# a CMakeLists.txt file
#
# Takes no arguments.
macro(iree_add_all_subdirs)
  FILE(GLOB _CHILDREN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
  SET(_DIRLIST "")
  foreach(_CHILD ${_CHILDREN})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${_CHILD} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_CHILD}/CMakeLists.txt)
      LIST(APPEND _DIRLIST ${_CHILD})
    endif()
  endforeach()

  foreach(subdir ${_DIRLIST})
    add_subdirectory(${subdir})
  endforeach()
endmacro()
