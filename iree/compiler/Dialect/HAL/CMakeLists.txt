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

add_subdirectory(Conversion)
add_subdirectory(IR)
add_subdirectory(Target)
add_subdirectory(Transforms)
add_subdirectory(Utils)

add_custom_command(
  PRE_BUILD
  OUTPUT hal.imports.h hal.imports.cc
  COMMAND generate_cc_embed_data "${CMAKE_CURRENT_SOURCE_DIR}/hal.imports.mlir"
          --output_header=hal.imports.h
          --output_impl=hal.imports.cc
          --identifier=hal_imports
          --cpp_namespace=mlir::iree_compiler
          --flatten
  DEPENDS generate_cc_embed_data
)

add_custom_target(iree_compiler_Dialect_HAL_hal_imports DEPENDS hal.imports.h)
