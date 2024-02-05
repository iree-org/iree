// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// `ret = argmax(in)`
//
// Conforms to ABI:
// #hal.pipeline.layout<push_constants = 1, sets = [
//   <0, bindings = [
//       <0, storage_buffer, ReadOnly>,
//       <1, storage_buffer>
//   ]>
// ]>

#version 450 core
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set=0, binding=0) buffer InputBuffer { float data[]; } Input;
layout(set=0, binding=1) buffer OutputBuffer { uvec2 data; } Output;

layout(push_constant) uniform PushConstants { uint totalCount; }; // Total number of scalars

// Each workgroup contains just one subgroup.

void main() {
  uint laneID = gl_LocalInvocationID.x;
  uint laneCount = gl_WorkGroupSize.x;

  float laneMax = Input.data[laneID];
  uint laneResult = 0;

  uint numBatches = totalCount / (laneCount);
  for (int i = 1; i < numBatches; ++i) {
    uint idx = laneCount * i + laneID;
    float new_in = Input.data[idx];
    laneResult = new_in > laneMax ? idx : laneResult;
    laneMax = max(laneMax, new_in);
  }

  // Final reduction with one subgroup
  float wgMax = subgroupMax(laneMax);

  // Find the smallest thread holding the maximum value.
  bool eq = wgMax == laneMax;
  uvec4 ballot = subgroupBallot(eq);
  uint lsb = subgroupBallotFindLSB(ballot);

  uint upper32bits = 0;
  if (laneID == lsb) Output.data = uvec2(laneResult, upper32bits);
}

