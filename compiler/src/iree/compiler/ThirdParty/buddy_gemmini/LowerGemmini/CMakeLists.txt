add_mlir_library(LowerGemminiPass
  LowerGemminiPass.cpp
  
  PARTIAL_SOURCES_INTENDED

  DEPENDS
    BuddyGemmini
    
  LINK_LIBS PUBLIC
    BuddyGemmini
    BuddyGemminiTransforms
    MLIRIR
    MLIRSupport
  
  LINK_COMPONENTS
    Support
)

if (NOT LLVM_ENABLE_RTTI)
  target_compile_options(LowerGemminiPass PRIVATE -fno-rtti)
  target_compile_options(obj.LowerGemminiPass PRIVATE -fno-rtti)
endif()

target_include_directories(LowerGemminiPass
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../_generated
)
