# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This is an example showing a basic makefile that links in the IREE runtime by
# way of the unified static library. It's recommended that IREE is added as a
# subproject and cmake is used to add the dependencies, but when using other
# build systems this is easier to adapt.
#
# Configure the runtime:
#   cmake -GNinja -B /tmp/iree-build-runtime/ .
# Build the runtime:
#   cmake --build /tmp/iree-build-runtime/ --target iree_runtime_unified
# Make this binary:
#   make
#
# Note that if IREE_SIZE_OPTIMIZED is used to build the runtime then the
# -DNDEBUG and -DIREE_STATUS_MODE=0 are required on any binaries using it. YMMV
# if changing any compiler options and not keeping them in sync. Prefer using
# cmake to ensure consistency between the builds.
#
# If cpuinfo is not supported on your platform then configure the runtime with
# -DIREE_ENABLE_CPUINFO=OFF.

RUNTIME_SRC_DIR ?= ../../runtime/src/
RUNTIME_BUILD_DIR ?= /tmp/iree-build-runtime/

SRC_FILES := main.c
INCLUDE_DIRS := ${RUNTIME_SRC_DIR}
INCLUDE_FLAGS := $(addprefix -I,${INCLUDE_DIRS})
LIBRARY_DIRS := \
		${RUNTIME_BUILD_DIR}/build_tools/third_party/flatcc/ \
		${RUNTIME_BUILD_DIR}/runtime/src/iree/runtime/ \
		${RUNTIME_BUILD_DIR}/third_party/cpuinfo/ \
		${RUNTIME_BUILD_DIR}/third_party/cpuinfo/deps/clog
LINK_LIBRARIES := \
		iree_runtime_unified \
		flatcc_parsing \
		cpuinfo \
		clog \
		dl \
		pthread
LIBRARY_FLAGS := $(addprefix -L,${LIBRARY_DIRS}) $(addprefix -l,${LINK_LIBRARIES})
CXX_FLAGS := -flto -Os ${INCLUDE_FLAGS} ${LIBRARY_FLAGS}

all: native-training 
clean:
	rm -f native-training 

native-training: ${SRC_FILES}
	${CXX} ${SRC_FILES} ${CXX_FLAGS} -o $@
