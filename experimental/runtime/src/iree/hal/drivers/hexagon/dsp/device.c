// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// DSP side of the FastRPC session opened by the Hexagon HAL device.

#include <stdlib.h>

#include "AEEStdErr.h"
#include "hexagon_dsp.h"

// State of a FastRPC session on the DSP.
typedef struct hexagon_dsp_session_t {
  // TODO(hexagon): add state (e.g. loaded executables) once there are RPCs.
  int reserved;
} hexagon_dsp_session_t;

// Opens a FastRPC session. |uri| contains the name of the library and the DSP
// domain. The returned |handle| is passed to all other functions.
int hexagon_dsp_open(const char* uri, remote_handle64* handle) {
  hexagon_dsp_session_t* session =
      (hexagon_dsp_session_t*)calloc(1, sizeof(*session));
  if (!session) return AEE_ENOMEMORY;
  *handle = (remote_handle64)session;
  return AEE_SUCCESS;
}

// Closes the FastRPC session with |handle| as returned by hexagon_dsp_open.
int hexagon_dsp_close(remote_handle64 handle) {
  free((hexagon_dsp_session_t*)handle);
  return AEE_SUCCESS;
}
