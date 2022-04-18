################################################################################
# iree-compiler-api-compile-mhlo-test
# If there ever become more of these, please convert into a function instead
# of open coding.
################################################################################

add_executable(
  iree-compiler-api-compile-mhlo-test
  compile-mhlo-test-main.c
)
target_link_libraries(iree-compiler-api-compile-mhlo-test
  PRIVATE
    IREECompilerCAPILib
    MLIRCAPIIR
)

add_test(
  NAME iree-compiler-api-compile-mhlo-test
  COMMAND iree-compiler-api-compile-mhlo-test
)
