// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/local_executable.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// A stack-owned executable used only for synchronous issue_call dispatch.
struct InlineDispatchRecorder {
  iree_hal_local_executable_t base = {};
  uint32_t fail_at = 0;
  uint32_t call_count = 0;
};

static iree_status_t RecordInlineWorkgroup(
    iree_hal_local_executable_t* executable, iree_host_size_t,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state, uint32_t) {
  auto* recorder = reinterpret_cast<InlineDispatchRecorder*>(executable);
  const uint32_t workgroup_index =
      (workgroup_state->workgroup_id_z * dispatch_state->workgroup_count_y +
       workgroup_state->workgroup_id_y) *
          dispatch_state->workgroup_count_x +
      workgroup_state->workgroup_id_x;
  EXPECT_EQ(workgroup_index, recorder->call_count);
  ++recorder->call_count;
  if (workgroup_index == recorder->fail_at) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "workgroup failure");
  }
  return iree_ok_status();
}

static const iree_hal_local_executable_vtable_t kInlineDispatchVtable = {
    {}, RecordInlineWorkgroup};

TEST(LocalExecutableTest, InlineDispatchStopsAtFirstFailure) {
  iree_hal_executable_dispatch_state_v0_t dispatch_state = {};
  dispatch_state.workgroup_count_x = 2;
  dispatch_state.workgroup_count_y = 2;
  dispatch_state.workgroup_count_z = 2;
  // Cover failure before and at every row and plane boundary. Later successful
  // workgroups must neither run nor overwrite the failure returned to callers.
  for (uint32_t fail_at = 0; fail_at < 8; ++fail_at) {
    SCOPED_TRACE(fail_at);
    InlineDispatchRecorder recorder;
    recorder.fail_at = fail_at;
    iree_hal_local_executable_initialize(
        &kInlineDispatchVtable, iree_allocator_system(), &recorder.base);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        iree_hal_local_executable_issue_dispatch_inline(
            &recorder.base, 0, &dispatch_state, 0, iree_byte_span_empty()));
    EXPECT_EQ(recorder.call_count, fail_at + 1);
    iree_hal_local_executable_deinitialize(&recorder.base);
  }
}

TEST(LocalExecutableTest, InlineDispatchCompletesAllWorkgroups) {
  InlineDispatchRecorder recorder;
  recorder.fail_at = 8;
  iree_hal_local_executable_initialize(&kInlineDispatchVtable,
                                       iree_allocator_system(), &recorder.base);
  iree_hal_executable_dispatch_state_v0_t dispatch_state = {};
  dispatch_state.workgroup_count_x = 2;
  dispatch_state.workgroup_count_y = 2;
  dispatch_state.workgroup_count_z = 2;
  IREE_EXPECT_OK(iree_hal_local_executable_issue_dispatch_inline(
      &recorder.base, 0, &dispatch_state, 0, iree_byte_span_empty()));
  EXPECT_EQ(recorder.call_count, 8);
  iree_hal_local_executable_deinitialize(&recorder.base);
}

TEST(LocalExecutableTest, DecodeDispatchReturnStatus) {
  iree_hal_local_executable_t executable = {};
  // Only v0.7 libraries return statuses.
  executable.library_version = IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0_7;
  iree_hal_executable_dispatch_attrs_v0_t attrs = {};
  executable.dispatch_attrs = &attrs;
  executable.export_count = 1;

  IREE_EXPECT_OK(iree_hal_local_executable_status_from_ret(&executable, 0, 0));
  attrs.flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_RETURNS_STATUS;
  IREE_EXPECT_OK(iree_hal_local_executable_status_from_ret(&executable, 0, 0));

  attrs.flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_local_executable_status_from_ret(
                            &executable, 0, IREE_STATUS_FAILED_PRECONDITION));
  attrs.flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_RETURNS_STATUS;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_local_executable_status_from_ret(
                            &executable, 0, IREE_STATUS_FAILED_PRECONDITION));
  attrs.flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_RETURNS_STATUS;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE,
                        iree_hal_local_executable_status_from_ret(
                            &executable, 0, IREE_STATUS_INCOMPATIBLE));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_local_executable_status_from_ret(
                            &executable, 0, IREE_STATUS_DEFERRED));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      iree_hal_local_executable_status_from_ret(&executable, 0, -1));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      iree_hal_local_executable_status_from_ret(&executable, 0, 32));

  executable.dispatch_attrs = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_local_executable_status_from_ret(
                            &executable, 0, IREE_STATUS_FAILED_PRECONDITION));
  executable.dispatch_attrs = &attrs;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_local_executable_status_from_ret(
                            &executable, 1, IREE_STATUS_FAILED_PRECONDITION));
}

}  // namespace
