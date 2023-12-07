#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <float.h>

extern "C" __device__ __attribute__((const)) half __ockl_wfred_max_f16(half);
extern "C" __device__ __attribute__((const)) float __ockl_wfred_max_f32(float);

extern "C" __device__ void rocm_argmax_F32I32(
    float *inputBuffer, size_t input_offset, int32_t *outputBuffer,
    size_t output_offset, size_t reductionSize, uint32_t flag) {
  uint laneID = threadIdx.x;
  uint laneCount = blockDim.x;
  // Set identity value to handle problem non divisible by subgroupSize.
  float laneMax = laneID >= reductionSize ? -FLT_MAX : inputBuffer[input_offset + laneID];
  uint laneResult = laneID;

  uint numBatches = reductionSize / warpSize;
  for (int i = 1; i < numBatches; ++i) {
    uint idx = laneCount * i + laneID;
    float new_in = idx >= reductionSize ? -FLT_MAX : inputBuffer[input_offset + idx];
    laneResult = new_in > laneMax ? idx : laneResult;
    laneMax = __hmax(new_in, laneMax);
  }

  // Final reduction with one subgroup
  float wgMax = __ockl_wfred_max_f32(laneMax);
  if (wgMax == laneMax) outputBuffer[output_offset] = laneResult;
}

extern "C" __device__ void rocm_argmax_F32I64(
    float *inputBuffer, size_t input_offset, int64_t *outputBuffer,
    size_t output_offset, size_t reductionSize, uint32_t flag) {
  uint laneID = threadIdx.x;
  uint laneCount = blockDim.x;
  // Set identity value to handle problem non divisible by subgroupSize.
  float laneMax = laneID >= reductionSize ? -FLT_MAX : inputBuffer[input_offset + laneID];
  uint laneResult = laneID;

  uint numBatches = reductionSize / warpSize;
  for (int i = 1; i < numBatches; ++i) {
    uint idx = laneCount * i + laneID;
    float new_in = idx >= reductionSize ? -FLT_MAX : inputBuffer[input_offset + idx];
    laneResult = new_in > laneMax ? idx : laneResult;
    laneMax = __hmax(new_in, laneMax);
  }

  // Final reduction with one subgroup
  float wgMax = __ockl_wfred_max_f32(laneMax);
  if (wgMax == laneMax) outputBuffer[output_offset] = laneResult;
}

extern "C" __device__ void rocm_argmax_F16I32(
    half *inputBuffer, size_t input_offset, int32_t *outputBuffer,
    size_t output_offset, size_t reductionSize, uint32_t flag) {
  half NEG_F16_MAX = __float2half(-65504.0f);
  uint laneID = threadIdx.x;
  uint laneCount = blockDim.x;
  // Set identity value to handle problem non divisible by subgroupSize.
  half laneMax = laneID >= reductionSize ? NEG_F16_MAX : inputBuffer[input_offset + laneID];
  uint laneResult = laneID;

  uint numBatches = reductionSize / warpSize;
  for (int i = 1; i < numBatches; ++i) {
    uint idx = laneCount * i + laneID;
    half new_in = idx >= reductionSize ? NEG_F16_MAX : inputBuffer[input_offset + idx];
    laneResult = new_in > laneMax ? idx : laneResult;
    laneMax = __hmax(new_in, laneMax);
  }

  // Final reduction with one subgroup
  half wgMax = __ockl_wfred_max_f16(laneMax);
  if (wgMax == laneMax) outputBuffer[output_offset] = laneResult;
}

extern "C" __device__ void rocm_argmax_F16I64(
    half *inputBuffer, size_t input_offset, int64_t *outputBuffer,
    size_t output_offset, size_t reductionSize, uint32_t flag) {
  half NEG_F16_MAX = __float2half(-65504.0f);
  uint laneID = threadIdx.x;
  uint laneCount = blockDim.x;
  // Set identity value to handle problem non divisible by subgroupSize.
  half laneMax = laneID >= reductionSize ? NEG_F16_MAX : inputBuffer[input_offset + laneID];
  uint laneResult = laneID;

  uint numBatches = reductionSize / warpSize;
  for (int i = 1; i < numBatches; ++i) {
    uint idx = laneCount * i + laneID;
    half new_in = idx >= reductionSize ? NEG_F16_MAX : inputBuffer[input_offset + idx];
    laneResult = new_in > laneMax ? idx : laneResult;
    laneMax = __hmax(new_in, laneMax);
  }

  // Final reduction with one subgroup
  half wgMax = __ockl_wfred_max_f16(laneMax);
  if (wgMax == laneMax) outputBuffer[output_offset] = laneResult;
}
