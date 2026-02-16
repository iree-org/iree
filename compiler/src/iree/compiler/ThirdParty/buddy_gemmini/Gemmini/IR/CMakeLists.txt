add_mlir_dialect_library(BuddyGemmini
  GemminiDialect.cpp
  GemminiRegistration.cpp

  DEPENDS
    mlir-headers

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRDialect
    MLIRSupport
    MLIRLLVMDialect
    MLIRFuncDialect
    MLIRVectorDialect
    
  LINK_COMPONENTS
    Support
)

# Disable RTTI to match LLVM build configuration
if (NOT LLVM_ENABLE_RTTI)
  target_compile_options(BuddyGemmini PRIVATE -fno-rtti)
  target_compile_options(obj.BuddyGemmini PRIVATE -fno-rtti)
endif()

# Set include directories properly
target_include_directories(BuddyGemmini
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../_generated>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_BINARY_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../_generated
)
