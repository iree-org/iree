add_subdirectory(Gemmini)
add_subdirectory(LowerGemmini)
add_subdirectory(LowerLinalgToGemmini)

# Build registration library for integrating with IREE tools
add_mlir_library(BuddyGemminiRegistration
  RegisterGemmini.cpp
  
  PARTIAL_SOURCES_INTENDED
  
  DEPENDS
    BuddyGemmini
    
  LINK_LIBS PUBLIC
    BuddyGemmini
    BuddyGemminiTransforms
    LowerLinalgToGemminiPass
    LowerGemminiPass
    MLIRPass
    MLIRIR
    MLIRSupport
  
  LINK_COMPONENTS
    Support
)

target_include_directories(BuddyGemminiRegistration
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/_generated
)

# Build test executable for Gemmini dialect
add_executable(test-gemmini-opt
  test-gemmini-opt.cpp
)

target_link_libraries(test-gemmini-opt
  PRIVATE
    BuddyGemminiRegistration
    BuddyGemmini
    BuddyGemminiTransforms
    LowerLinalgToGemminiPass
    LowerGemminiPass
    MLIROptLib
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIRIR
    MLIRAnalysis
    MLIRParser
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRLLVMDialect
    MLIRAffineDialect
    MLIRControlFlowDialect
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRFuncToLLVM
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRMemRefToLLVM
    MLIRReconcileUnrealizedCasts
)

target_include_directories(test-gemmini-opt
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/_generated
)

# Simpler test executable without full MLIR pass infrastructure
add_executable(test-gemmini-simple
  test-gemmini-simple.cpp
)

target_link_libraries(test-gemmini-simple
  PRIVATE
    BuddyGemmini
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    LLVMSupport
    LLVMCore
)

target_include_directories(test-gemmini-simple
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/_generated
)
