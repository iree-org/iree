# Option to enable/disable LegalizeForLLVMExport (temporarily disabled for build fixes)
option(IREE_ENABLE_BUDDY_GEMMINI_LEGALIZE "Enable Gemmini LLVM export legalization" OFF)

if(IREE_ENABLE_BUDDY_GEMMINI_LEGALIZE)
  set(GEMMINI_TRANSFORM_SOURCES
    LegalizeForLLVMExport.cpp
  )
else()
  set(GEMMINI_TRANSFORM_SOURCES
    Empty.cpp
    GemminiLegalizeStubs.cpp
  )
endif()

add_mlir_library(BuddyGemminiTransforms
  ${GEMMINI_TRANSFORM_SOURCES}
  GemminiIRDumps.cpp
  
  PARTIAL_SOURCES_INTENDED

  DEPENDS
    BuddyGemmini

  LINK_LIBS PUBLIC
    BuddyGemmini
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRLLVMDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRVectorDialect
    MLIRTransforms
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRSCFDialect
)

if (NOT LLVM_ENABLE_RTTI)
  target_compile_options(BuddyGemminiTransforms PRIVATE -fno-rtti)
  target_compile_options(obj.BuddyGemminiTransforms PRIVATE -fno-rtti)
endif()
