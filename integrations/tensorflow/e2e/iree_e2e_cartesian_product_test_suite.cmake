# Copyright 2021 Google LLC
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

# iree_e2e_cartesian_product_test_suite()
#
# Expands a testing matrix to create python test targets.
#
# Mirrors the bzl rule of the same name.
#
# This rule uses CMake in a way that it was never meant to be used. In
# particular it uses custom hackery to enable nested data structures. The
# original Starlark rule was written without the expectation that it would ever
# need to be ported to CMake and so has some complex matrix expansion logic.
# The author apologizes for what you are about to read.
#
# Parameters:
#   NAME: Base name of the test targets.
#   MATRIX_KEYS: Keys for the different dimensions of the matrix to be expanded.
#       One of the keys must be `src` and is expanded to the source files for
#       each test. One key must be `target_backends` and is expanded to the
#       target backends for the IREE integration test. It is also used to derive
#       a driver label to attach to the generated tests. All other keys are
#       assumed to correspond to flag names that should be passed to the
#       generated tests (with the corresponding values from MATRIX_VALUES).
#   MATRIX_VALUES: Lists of values for each key of the matrix. Each element of
#       this argument is interpreted as a list and the order of arguments must
#       match MATRIX_KEYS.
#   FAILING_CONFIGURATIONS: Configurations in the matrix expansion that are
#       expected to fail. Each element of this argument is a *comma*-separated
#       list of matrix values in the same order as MATRIX_KEYS (and having the
#       same number of elements. An empty element is interpreted to correspond
#       to all values for that key in the matrix.
#   LABELS: Additional labels to apply to the test. The package path and
#       "driver=${DRIVER}" based on the target_backends matrix key are added
#       automatically.
#
#
# Example:
#   iree_e2e_cartesian_product_test_suite(
#     NAME
#       my_tests
#     MATRIX_KEYS
#       "src"
#       "target_backend"
#       "reference_backend"
#       "magic_flag"
#     MATRIX_VALUES
#       "concat_test.py;range_test.py"
#       "tf;tflite;iree_vmla;iree_llvmaot;iree_vulkan"
#       "tf"
#       "true;false"
#    FAILING_CONFIGURATIONS
#      "concat_test.py,,,true"
#      "range_test.py,iree_vulkan,,"
#   )
#
# Would expand to the following tests:
#   DISABLED: python concat_test.py --target_backend=tf           --reference_backend=tf --magic_flag=true
#   DISABLED: python concat_test.py --target_backend=tflite       --reference_backend=tf --magic_flag=true
#   DISABLED: python concat_test.py --target_backend=iree_vmla    --reference_backend=tf --magic_flag=true
#   DISABLED: python concat_test.py --target_backend=iree_llvmaot --reference_backend=tf --magic_flag=true
#   DISABLED: python concat_test.py --target_backend=iree_vulkan  --reference_backend=tf --magic_flag=true
#             python range_test.py  --target_backend=tf           --reference_backend=tf --magic_flag=true
#             python range_test.py  --target_backend=tflite       --reference_backend=tf --magic_flag=true
#             python range_test.py  --target_backend=iree_vmla    --reference_backend=tf --magic_flag=true
#             python range_test.py  --target_backend=iree_llvmaot --reference_backend=tf --magic_flag=true
#   DISABLED: python range_test.py  --target_backend=iree_vulkan  --reference_backend=tf --magic_flag=true
#             python concat_test.py --target_backend=tf           --reference_backend=tf --magic_flag=false
#             python concat_test.py --target_backend=tflite       --reference_backend=tf --magic_flag=false
#             python concat_test.py --target_backend=iree_vmla    --reference_backend=tf --magic_flag=false
#             python concat_test.py --target_backend=iree_llvmaot --reference_backend=tf --magic_flag=false
#             python concat_test.py --target_backend=iree_vulkan  --reference_backend=tf --magic_flag=false
#             python range_test.py  --target_backend=tf           --reference_backend=tf --magic_flag=false
#             python range_test.py  --target_backend=tflite       --reference_backend=tf --magic_flag=false
#             python range_test.py  --target_backend=iree_vmla    --reference_backend=tf --magic_flag=false
#             python range_test.py  --target_backend=iree_llvmaot --reference_backend=tf --magic_flag=false
#   DISABLED: python range_test.py  --target_backend=iree_vulkan  --reference_backend=tf --magic_flag=false
#
function(iree_e2e_cartesian_product_test_suite)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    PARSE_ARGV 0
    _RULE
    ""
    "NAME"
    "MATRIX_KEYS;MATRIX_VALUES;FAILING_CONFIGURATIONS;LABELS"
  )

  list(LENGTH _RULE_MATRIX_KEYS _MATRIX_KEYS_COUNT)
  list(LENGTH _RULE_MATRIX_VALUES _MATRIX_VALUES_COUNT)

  if(NOT _MATRIX_KEYS_COUNT EQUAL _MATRIX_VALUES_COUNT)
    message(
      SEND_ERROR
        "MATRIX_KEYS count ${_MATRIX_KEYS_COUNT} does not match MATRIX_VALUES count ${_MATRIX_VALUES_COUNT}"
    )
  endif()
  list(FIND _RULE_MATRIX_KEYS "src" _SRC_KEY_INDEX)
  if(_SRC_KEY_INDEX EQUAL -1)
    message(SEND_ERROR "Did not find key `src` in MATRIX_KEYS: ${_RULE_MATRIX_KEYS}")
  endif()

  list(FIND _RULE_MATRIX_KEYS "target_backends" _TARGET_BACKENDS_KEY_INDEX)
  if(_TARGET_BACKENDS_KEY_INDEX EQUAL -1)
    message(SEND_ERROR "Did not find key `target_backends` in MATRIX_KEYS: ${_RULE_MATRIX_KEYS}")
  endif()
  math(EXPR _MAX_INDEX "${_MATRIX_KEYS_COUNT} - 1")

  # Process failing configurations, expanding empty entries to be all entries
  # for that key.
  set(_FAILING_CONFIGURATIONS "")
  foreach(_FAILING_CONFIGURATION IN LISTS _RULE_FAILING_CONFIGURATIONS)
    string(REPLACE "," ";" _CONFIGURATION_LIST "${_FAILING_CONFIGURATION}")
    list(LENGTH _CONFIGURATION_LIST _CONFIGURATION_KEY_COUNT)
    if(NOT _CONFIGURATION_KEY_COUNT EQUAL _MATRIX_KEYS_COUNT)
      message(SEND_ERROR
        "Failing configuration ${_FAILING_CONFIGURATION} does not have same number of entries (${_MATRIX_KEY_COUNT}) as MATRIX_KEYS")
    endif()

    list(GET _CONFIGURATION_LIST 0 _FIRST_CONFIG_VALUE)
    if(NOT _FIRST_CONFIG_VALUE)
      list(GET _RULE_MATRIX_VALUES 0 _EXPANDED_CONFIGURATIONS)
    else()
      set(_EXPANDED_CONFIGURATIONS "${_FIRST_CONFIG_VALUE}")
    endif()
    foreach(_INDEX RANGE 1 "${_MAX_INDEX}")
      list(GET _CONFIGURATION_LIST ${_INDEX} _CONFIG_VALUE)
      set(_KEY_CONFIGURATIONS "")
      if(_CONFIG_VALUE)
        list(TRANSFORM
          _EXPANDED_CONFIGURATIONS
          APPEND ",${_CONFIG_VALUE}"
          OUTPUT_VARIABLE _KEY_VALUE_CONFIGURATIONS)
        list(APPEND _KEY_CONFIGURATIONS ${_KEY_VALUE_CONFIGURATIONS})
      else()
        list(GET _RULE_MATRIX_VALUES ${_INDEX} _MATRIX_VALUES)
        foreach(_MATRIX_VALUE IN LISTS _MATRIX_VALUES)
          list(TRANSFORM
            _EXPANDED_CONFIGURATIONS
            APPEND ",${_MATRIX_VALUE}"
            OUTPUT_VARIABLE _KEY_VALUE_CONFIGURATIONS)
          list(APPEND _KEY_CONFIGURATIONS ${_KEY_VALUE_CONFIGURATIONS})
        endforeach()
      endif()
      set(_EXPANDED_CONFIGURATIONS "${_KEY_CONFIGURATIONS}")
    endforeach()
    list(APPEND _FAILING_CONFIGURATIONS ${_EXPANDED_CONFIGURATIONS})
  endforeach()
  list(REMOVE_DUPLICATES _FAILING_CONFIGURATIONS)

  # Build up all configurations
  list(GET _RULE_MATRIX_VALUES 0 _ALL_CONFIGURATIONS)
  foreach(_INDEX RANGE 1 "${_MAX_INDEX}")
    list(GET _RULE_MATRIX_KEYS ${_INDEX} _MATRIX_KEY)
    list(GET _RULE_MATRIX_VALUES ${_INDEX} _MATRIX_VALUES)

    set(_KEY_CONFIGURATIONS "")
    foreach(_MATRIX_VALUE IN LISTS _MATRIX_VALUES)
      list(TRANSFORM
        _ALL_CONFIGURATIONS
        APPEND ",${_MATRIX_VALUE}"
        OUTPUT_VARIABLE _KEY_VALUE_CONFIGURATIONS)
      list(APPEND _KEY_CONFIGURATIONS ${_KEY_VALUE_CONFIGURATIONS})
    endforeach()
    set(_ALL_CONFIGURATIONS ${_KEY_CONFIGURATIONS})
  endforeach()

  foreach(_CONFIGURATION IN LISTS _ALL_CONFIGURATIONS)
    list(FIND _FAILING_CONFIGURATIONS "${_CONFIGURATION}" _FAILING_INDEX)
    if (_FAILING_INDEX EQUAL -1)
      string(REGEX MATCHALL "[^,]+" _CONFIGURATION_LIST "${_CONFIGURATION}")
      list(GET _CONFIGURATION_LIST ${_SRC_KEY_INDEX} _TEST_SRC)
      list(GET _CONFIGURATION_LIST ${_TARGET_BACKENDS_KEY_INDEX} _TEST_TARGET_BACKENDS)

      # Construct the test name
      set(_TEST_NAME_LIST "${_RULE_NAME}")
      string(REGEX REPLACE "\.py$" "" _STRIPPED_SRC_NAME "${_TEST_SRC}")
      string(REGEX REPLACE "_test$" "" _STRIPPED_SRC_NAME "${_STRIPPED_SRC_NAME}")
      list(APPEND _TEST_NAME_LIST "${_STRIPPED_SRC_NAME}" "${_TEST_TARGET_BACKENDS}")
      foreach(_INDEX RANGE "${_MAX_INDEX}")
        list(GET _RULE_MATRIX_VALUES ${_INDEX} _KEY_MATRIX_VALUES)
        list(LENGTH _KEY_MATRIX_VALUES _KEY_MATRIX_VALUES_COUNT)
        if (NOT _INDEX EQUAL _SRC_KEY_INDEX AND
            NOT _INDEX EQUAL _TARGET_BACKENDS_KEY_INDEX AND
            NOT _KEY_MATRIX_VALUES_COUNT EQUAL 1)
          list(GET _RULE_MATRIX_KEYS ${_INDEX} _MATRIX_KEY)
          list(GET _CONFIGURATION_LIST ${_INDEX} _MATRIX_VALUE)
          list(APPEND _TEST_NAME_LIST "${_MATRIX_KEY}" "${_MATRIX_VALUE}")
        endif()
      endforeach()
      list(JOIN _TEST_NAME_LIST "__" _TEST_NAME)

      # Consruct the test args
      set(_TEST_ARGS "")
      foreach(_INDEX RANGE "${_MAX_INDEX}")
        if (NOT _INDEX EQUAL _SRC_KEY_INDEX)
          list(GET _RULE_MATRIX_KEYS ${_INDEX} _MATRIX_KEY)
          list(GET _CONFIGURATION_LIST ${_INDEX} _MATRIX_VALUE)
          list(APPEND _TEST_ARGS "--${_MATRIX_KEY}=${_MATRIX_VALUE}")
        endif()
      endforeach()

      # Extract the driver label
      string(REPLACE "iree_" "" _DRIVER "${_TEST_TARGET_BACKENDS}")
      set(_TEST_LABELS "${_RULE_LABELS}")
      list(APPEND _TEST_LABELS "driver=${_DRIVER}")

      iree_py_test(
        NAME
          ${_TEST_NAME}
        SRCS
          "${_TEST_SRC}"
        ARGS
          ${_TEST_ARGS}
        LABELS
          ${_TEST_LABELS}
      )
    endif()
  endforeach()
endfunction()
