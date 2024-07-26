### Components built ###

set(LLVM_ENABLE_PROJECTS "clang;clang-tools-extra;lld" CACHE STRING "")

### Target architectures ###

# Compiler target architectures
set(LLVM_TARGETS_TO_BUILD "X86" CACHE STRING "")

set(LLVM_ENABLE_RUNTIMES "compiler-rt" CACHE STRING "")

# CMake builtin variables and modules are not available for this cache file
# Gather directly build/host information
execute_process(COMMAND "uname" "-m" OUTPUT_VARIABLE _UNAME_M)
string(STRIP ${_UNAME_M} BUILD_MACHINE_ARCH)

### Default settings for the toolchain ###

# Use the LLVM components
set(CLANG_DEFAULT_OBJCOPY llvm-objcopy CACHE STRING "")
set(CLANG_DEFAULT_LINKER lld CACHE STRING "")

set(CLANG_ENABLE_STATIC_ANALYZER ON CACHE BOOL "")
set(LLVM_ENABLE_LIBCXX OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB ON CACHE BOOL "")

### Disk size optimizations ###

# All the tools will use libllvm shared library
set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "")
set(LLVM_LINK_LLVM_DYLIB ON CACHE BOOL "")

# When exceptions are disabled, unwind tables are large and useless
set(LLVM_ENABLE_UNWIND_TABLES OFF CACHE BOOL "")

# Mildly useful misc stuff (which might also be hard to cross-compile)
set(CLANG_ENABLE_ARCMT OFF CACHE BOOL "")
set(CLANG_PLUGIN_SUPPORT OFF CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(LLVM_ENABLE_Z3_SOLVER OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_INCLUDE_GO_TESTS OFF CACHE BOOL "")
set(LLVM_FORCE_ENABLE_STATS ON CACHE BOOL "")

# Do not store debug information by default.
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

# Use the distributions below for the installation
set(LLVM_INSTALL_TOOLCHAIN_ONLY OFF CACHE BOOL "")

### Distributions ###

set(LLVM_DISTRIBUTIONS
      Toolchain
      Development
    CACHE STRING "")

set(LLVM_TOOLCHAIN_TOOLS
  llvm-addr2line
  llvm-ar
  llvm-cxxfilt
  llvm-dis
  llvm-dwarfdump
  llvm-lib
  llvm-link
  llvm-mc
  llvm-nm
  llvm-objcopy
  llvm-objdump
  llvm-rc
  llvm-ranlib
  llvm-readelf
  llvm-readobj
  llvm-size
  llvm-strip
  llvm-symbolizer
  llvm-xray
  CACHE STRING "")

set(LLVM_BUILD_UTILS ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_TOOLCHAIN_UTILITIES
    FileCheck
    count
    not
    CACHE STRING "")

set(LLVM_RUNTIME_DISTRIBUTION_COMPONENTS
    CACHE STRING "")

set(LLVM_Toolchain_DISTRIBUTION_COMPONENTS
      builtins
      runtimes
      clang
      clang-cpp
      clang-format
      clang-offload-bundler
      clang-resource-headers
      clang-tidy
      clangd
      libclang
      lld
      LLVM
      LTO
      ${LLVM_TOOLCHAIN_TOOLS}
      ${LLVM_TOOLCHAIN_UTILITIES}
    CACHE STRING "")

set(LLVM_Development_DISTRIBUTION_COMPONENTS
      # LLVM
      cmake-exports
      development-cmake-exports
      toolchain-cmake-exports
      llc
      llvm-config
      llvm-headers
      llvm-libraries
      opt
      Remarks
      # Clang
      clang-cmake-exports
      clang-development-cmake-exports
      clang-toolchain-cmake-exports
      clang-headers
      clang-libraries
      # LLD
      lld-cmake-exports
      lld-toolchain-cmake-exports
    CACHE STRING "")
