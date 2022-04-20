add_subdirectory(IR)
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Transforms")
  add_subdirectory(Transforms)
endif()
