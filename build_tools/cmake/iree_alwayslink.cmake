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

include(CMakeParseArguments)

# Additional libraries containing statically registered functions/flags, which
# should always be linked in to binaries.


# set_alwayslink_property()
#
# CMake function to set the ALWAYSLINK on external libraries
#
# Parameters:
# ALWAYSLINK_LIBS: List of libraries
# SKIP_NONEXISTING: When added, ALWAYSLINK is only set on existing libraries.

function(set_alwayslink_property)
  cmake_parse_arguments(
    _RULE
    "SKIP_NONEXISTING"
    ""
    "ALWAYSLINK_LIBS"
    ${ARGN}
  )

  foreach(_LIB ${_RULE_ALWAYSLINK_LIBS})
    # If SKIP_NONEXISTING is false: Always try to set the property.
    # If SKIP_NONEXISTING is true : Only set the property if the target exists.
    if(NOT TARGET ${_LIB} AND _RULE_SKIP_NONEXISTING)
      continue()
    endif()

    # Check if the target is an aliased target.
    # If so get the non aliased target.
    get_target_property(_ALIASED_TARGET ${_LIB} ALIASED_TARGET)
    if(_ALIASED_TARGET)
      set(_LIB ${_ALIASED_TARGET})
    endif()

    set_property(TARGET ${_LIB} PROPERTY ALWAYSLINK 1)
  endforeach()
endfunction()


function(set_alwayslink_mlir_libs)
  set(_ALWAYSLINK_LIBS_MLIR
    # Dep tagged ALWAYSLINK for mlir-translate
    MLIRSPIRVSerialization
    # Required IR targets
    MLIRIR
    # Required passes
    MLIRPass
    # Required transforms
    MLIRGPUtoCUDATransforms
    MLIRGPUtoNVVMTransforms
    MLIRGPUtoROCDLTransforms
    MLIRGPUtoVulkanTransforms
    MLIRQuantizerTransforms
    MLIRLinalgToLLVM # createConvertLinalgToLLVMPass()
    MLIRLinalgToSPIRVTransforms
    MLIRLoopOpsTransforms
    # TODO(marbre): Check the previously added libs
    MLIRAnalysis
    MLIREDSC
    MLIRLoopToStandard
    MLIRParser
    MLIRSPIRVTransforms
    MLIRStandardToLLVM
    MLIRTargetLLVMIR
    MLIRTransforms
    MLIRTranslation
    MLIRSupport
    MLIROptLib
  )

  set_alwayslink_property(
    ALWAYSLINK_LIBS
      ${_ALWAYSLINK_LIBS_MLIR}
  )
endfunction()


function(set_alwayslink_tensorflow_libs)
  set(_ALWAYSLINK_LIBS_TENSORFLOW
    tensorflow::mlir_xla
  )

  set_alwayslink_property(
    ALWAYSLINK_LIBS
      ${_ALWAYSLINK_LIBS_TENSORFLOW}
  )
endfunction()
