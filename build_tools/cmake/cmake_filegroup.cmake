function(cmake_filegroup)
  cmake_parse_arguments(
    _ARG
    ""
    "NAME"
    "FILES;DEPENDS"
    ${ARGN}
  )
  rename_bazel_targets(_NAME "${_ARG_NAME}")
  add_custom_target(${_NAME})

  foreach(_FILE ${_ARG_FILES})
    if(IS_ABSOLUTE "${_FILE}")
      set(_INPUT_PATH "${_FILE}")
      get_filename_component(_FILE_NAME ${_FILE} NAME)
      canonize_bazel_target_names(_FILE_TARGET "${_FILE_NAME}")
      rename_bazel_targets(_TARGET "${_FILE_TARGET}")
      string(REPLACE "::" "/" _FILE_PATH ${_FILE_TARGET})
      set(_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${_FILE_PATH}")
    else()
      canonize_bazel_target_names(_FILE_TARGET "${_FILE}")
      rename_bazel_targets(_TARGET "${_FILE_TARGET}")
      string(REPLACE "::" "/" _FILE_PATH ${_FILE_TARGET})
      set(_INPUT_PATH "${PROJECT_SOURCE_DIR}/${_FILE_PATH}")
      set(_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${_FILE_PATH}")
    endif()

    if(NOT TARGET ${_TARGET})
      add_custom_command(OUTPUT "${_OUTPUT_PATH}"
        COMMAND ${CMAKE_COMMAND} -E copy "${_INPUT_PATH}" "${_OUTPUT_PATH}"
        DEPENDS "${_INPUT_PATH}")
      add_custom_target(${_TARGET} DEPENDS "${_OUTPUT_PATH}")
    endif()

    add_dependencies(${_NAME} ${_TARGET})
  endforeach()

  if(_ARG_DEPENDS)
    rename_bazel_targets(_DEPS "${_ARG_DEPENDS}")
    add_dependencies(${_NAME} ${_DEPS})
  endif()

  set_target_properties(${_NAME} PROPERTIES
    IS_FILEGROUP TRUE
    OUTPUTS "${_SRCS}")
endfunction()
