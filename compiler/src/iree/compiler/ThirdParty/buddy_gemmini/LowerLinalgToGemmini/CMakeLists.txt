add_mlir_library(LowerLinalgToGemminiPass
  LowerLinalgToGemmini.cpp
  
  PARTIAL_SOURCES_INTENDED
  
  DEPENDS
    BuddyGemmini
  
  LINK_LIBS PUBLIC
    BuddyGemmini
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRTransforms
    MLIRIR
  
  LINK_COMPONENTS
    Support
)

if (NOT LLVM_ENABLE_RTTI)
  target_compile_options(LowerLinalgToGemminiPass PRIVATE -fno-rtti)
  target_compile_options(obj.LowerLinalgToGemminiPass PRIVATE -fno-rtti)
endif()

target_include_directories(LowerLinalgToGemminiPass
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../_generated
)
