################################################################################
# iree-compiler-api-compile-mhlo-test
# If there ever become more of these, please convert into a function instead
# of open coding.
################################################################################

set(_NAME "compile-mhlo-test")

add_executable(
  ${_NAME}
  compile-mhlo-test-main.c
)
target_link_libraries(${_NAME}
  PRIVATE
    IREECompilerCAPILib
    MLIRCAPIIR
)

iree_package_ns(_PACKAGE_NS)
string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
set(_NAME_PATH "${_PACKAGE_PATH}/${_NAME}")

add_test(
  NAME ${_NAME_PATH}
  COMMAND ${_NAME}
)
