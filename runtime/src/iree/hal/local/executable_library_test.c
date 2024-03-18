// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_library.h"

#include <stdbool.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"
#include "iree/hal/local/executable_environment.h"
#include "iree/hal/local/executable_library_demo.h"

// Demonstration of the HAL-side of the iree_hal_executable_library_t ABI.
// This is the lowest level of the system right before calling into generated
// code.
//
// This shows what the various execution systems are doing (through a lot
// of fancy means): all `inline_command_buffer.c` and `task_command_buffer.c`
// lead up to just calling into the iree_hal_executable_dispatch_v0_t entry
// point functions with a state structure and a workgroup XYZ.
//
// Below walks through acquiring the library pointer (which in this case is a
// hand-coded example to show the codegen-side), setting up the I/O buffers and
// state, and calling the function to do some math.
//
// See iree/hal/local/executable_library.h for more information.
int main(int argc, char** argv) {
  // Default environment.
  iree_hal_executable_environment_v0_t environment;
  iree_hal_executable_environment_initialize(iree_allocator_default(),
                                             &environment);

  // Query the library header at the requested version.
  // The query call in this example is going into the handwritten demo code
  // but could be targeted at generated files or runtime-loaded shared objects.
  union {
    const iree_hal_executable_library_header_t** header;
    const iree_hal_executable_library_v0_t* v0;
  } library;
  library.header = demo_executable_library_query(
      IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST, &environment);
  IREE_ASSERT_NE(library.header, NULL, "version may not have matched");
  const iree_hal_executable_library_header_t* header = *library.header;
  IREE_ASSERT_NE(header, NULL, "version may not have matched");
  IREE_ASSERT_LE(
      header->version, IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
      "expecting the library to have the same or older version as us");
  IREE_ASSERT(strcmp(header->name, "demo_library") == 0,
              "library name can be used to rendezvous in a registry");
  IREE_ASSERT_GT(library.v0->exports.count, 0,
                 "expected at least one entry point");

  // Push constants are an array of 4-byte values that are much more efficient
  // to specify (no buffer pointer indirection) and more efficient to access
  // (static struct offset address calculation, all fit in a few cache lines,
  // etc). They are limited in capacity, though, so only <=64(ish) are usable.
  dispatch_tile_a_push_constants_t push_constants;
  memset(&push_constants, 0, sizeof(push_constants));
  push_constants.f0 = 5.0f;

  // Setup the two buffer bindings the entry point is expecting.
  // They only need to remain valid for the duration of the invocation and all
  // memory accessed by the invocation will come from here.
  float arg0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float ret0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float ret0_expected[4] = {6.0f, 7.0f, 8.0f, 9.0f};
  size_t binding_lengths[2] = {
      sizeof(arg0),
      sizeof(ret0),
  };
  void* binding_ptrs[2] = {
      arg0,
      ret0,
  };

  // Resolve the entry point by ordinal.
  const iree_hal_executable_dispatch_v0_t entry_fn_ptr =
      library.v0->exports.ptrs[0];

  // Dispatch each workgroup with the same state.
  const iree_hal_executable_dispatch_state_v0_t dispatch_state = {
      .workgroup_count_x = 4,
      .workgroup_count_y = 1,
      .workgroup_count_z = 1,
      .workgroup_size_x = 1,
      .workgroup_size_y = 1,
      .workgroup_size_z = 1,
      .max_concurrency = 1,
      .push_constant_count = IREE_ARRAYSIZE(push_constants.values),
      .push_constants = push_constants.values,
      .binding_count = IREE_ARRAYSIZE(binding_ptrs),
      .binding_ptrs = binding_ptrs,
      .binding_lengths = binding_lengths,
  };
  iree_hal_executable_workgroup_state_v0_t workgroup_state = {
      .processor_id = iree_cpu_query_processor_id(),
  };
  for (uint32_t z = 0; z < dispatch_state.workgroup_count_z; ++z) {
    workgroup_state.workgroup_id_z = z;
    for (uint32_t y = 0; y < dispatch_state.workgroup_count_y; ++y) {
      workgroup_state.workgroup_id_y = y;
      for (uint32_t x = 0; x < dispatch_state.workgroup_count_x; ++x) {
        workgroup_state.workgroup_id_x = x;
        // Invoke the workgroup (x, y, z).
        int ret = entry_fn_ptr(&environment, &dispatch_state, &workgroup_state);
        IREE_ASSERT_EQ(
            ret, 0,
            "if we have bounds checking enabled the executable will signal "
            "us of badness");
      }
    }
  }

  // Ensure it worked.
  bool all_match = true;
  for (size_t i = 0; i < IREE_ARRAYSIZE(ret0_expected); ++i) {
    IREE_ASSERT_EQ(ret0[i], ret0_expected[i], "math is hard");
    all_match = all_match && ret0[i] == ret0_expected[i];
  }
  return all_match ? 0 : 1;
}
