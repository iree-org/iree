function(iree_py_test)
  cmake_parse_arguments(
    PARSE_ARGV 0
    _RULE
    ""
    "NAME;SRC"
    "ARGS;LABELS"
  )

  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_TEST_NAME "${_PACKAGE_PATH}/${_RULE_NAME}")

  add_test(
    NAME ${_TEST_NAME}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND
      "${CMAKE_SOURCE_DIR}/build_tools/cmake/run_test.${IREE_HOST_SCRIPT_EXT}"
      "${Python3_EXECUTABLE}" -B
      "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}"
      ${_RULE_ARGS}
  )
  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  set_property(TEST ${_TEST_NAME} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(
    TEST ${_TEST_NAME}
    PROPERTY ENVIRONMENT
    "TEST_TMPDIR=${_RULE_NAME}_test_tmpdir"
    "PYTHONPATH=${CMAKE_BINARY_DIR}/bindings/python")
endfunction()

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

    # message("Processing... ${_CONFIGURATION_LIST}")
    list(GET _CONFIGURATION_LIST 0 _FIRST_CONFIG_VALUE)
    # message("    First config value '${_FIRST_CONFIG_VALUE}'")
    if(NOT _FIRST_CONFIG_VALUE)
      list(GET _RULE_MATRIX_VALUES 0 _EXPANDED_CONFIGURATIONS)
    else()
      set(_EXPANDED_CONFIGURATIONS "${_FIRST_CONFIG_VALUE}")
    endif()
    # message("    Initial expanded: ${_EXPANDED_CONFIGURATIONS}")
    foreach(_INDEX RANGE 1 "${_MAX_INDEX}")
      list(GET _CONFIGURATION_LIST ${_INDEX} _CONFIG_VALUE)
      # message("      Config value: ${_CONFIG_VALUE}")
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
      # message("      Expanded configurations: ${_EXPANDED_CONFIGURATIONS}")
    endforeach()
    list(APPEND _FAILING_CONFIGURATIONS ${_EXPANDED_CONFIGURATIONS})
    # message("      Failing configurations: ${_FAILING_CONFIGURATIONS}")
  endforeach()
  list(REMOVE_DUPLICATES _FAILING_CONFIGURATIONS)
  list(LENGTH _FAILING_CONFIGURATIONS _FAILING_CONFIGURATIONS_COUNT)
  foreach(_FAILING_CONFIG IN LISTS _FAILING_CONFIGURATIONS)
    # message("${_FAILING_CONFIG}")
  endforeach()

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
      # message(WARNING "${_CONFIGURATION_LIST}")
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
          # message(WARNING "${_CONFIGURATION_LIST}")
          list(GET _CONFIGURATION_LIST ${_INDEX} _MATRIX_VALUE)
          list(APPEND _TEST_ARGS "--${_MATRIX_KEY}=${_MATRIX_VALUE}")
        endif()
      endforeach()

      # Extract the driver label
      string(REPLACE "iree_" "" _DRIVER "${_TEST_TARGET_BACKENDS}")
      set(_TEST_LABELS "${_RULE_LABELS}")
      list(APPEND _TEST_LABELS "driver=${_DRIVER}")

      # message(WARNING "${_TEST_NAME}: ${_TEST_ARGS} : ${_TEST_LABELS}")
      iree_py_test(
        NAME
          ${_TEST_NAME}
        SRC
          "${_TEST_SRC}"
        ARGS
          ${_TEST_ARGS}
        LABELS
          ${_TEST_LABELS}
      )
    endif()
  endforeach()
endfunction()
