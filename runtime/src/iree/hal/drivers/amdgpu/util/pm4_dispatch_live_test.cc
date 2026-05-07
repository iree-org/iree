// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/threading/processor.h"
#include "iree/hal/drivers/amdgpu/abi/kernel_descriptor.h"
#include "iree/hal/drivers/amdgpu/util/aql_ring.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/pm4_barrier.h"
#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"
#include "iree/hal/drivers/amdgpu/util/pm4_program.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

constexpr char kTestCodeObjectPath[] =
    "runtime/src/iree/hal/drivers/amdgpu/util/"
    "pm4_dispatch_test_kernels_gfx1100.so";
constexpr uint32_t kAqlValueA = 0xA1100001u;
constexpr uint32_t kAqlValueB = 0xB2200002u;
constexpr uint32_t kPm4ValueA = 0xA4400004u;
constexpr uint32_t kPm4ValueB = 0xB5500005u;
constexpr uint32_t kPm4BarrierValue = 0xC6600006u;
constexpr uint32_t kPm4BarrierAdd = 0x330u;
constexpr uint32_t kPm4PatchValue = 0xD7700007u;
constexpr uint32_t kPm4PatchWrongValue = 0xE8800008u;
constexpr uint16_t kWorkgroupSize[3] = {64, 1, 1};
constexpr uint32_t kDispatchGridSize[3] = {64, 1, 1};
constexpr iree_hal_amdgpu_vendor_packet_capability_flags_t
    kBarrierCapabilities =
        IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
        IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM;

struct StoreKernargs {
  uint32_t* target;
  uint32_t value;
};

struct ReadAddKernargs {
  uint32_t* source;
  uint32_t* target;
  uint32_t value;
};

struct PatchUserDataKernargs {
  uint32_t* target_dwords;
  uint32_t dword_offset;
  uint64_t kernarg_address;
};

struct alignas(64) LiveMemory {
  uint32_t outputs[4];
  uint32_t scratch[2];
  uint32_t completion;
  uint32_t reserved;
  StoreKernargs store_kernargs[4];
  ReadAddKernargs read_add_kernargs;
  PatchUserDataKernargs patch_user_data_kernargs;
};

struct KernelInfo {
  hsa_executable_symbol_t symbol = {0};
  uint64_t kernel_object = 0;
  uint32_t kernarg_size = 0;
  uint32_t kernarg_alignment = 0;
  uint32_t private_segment_size = 0;
  uint32_t group_segment_size = 0;
};

struct QueueError {
  std::atomic<uint32_t> callback_count{0};
  std::atomic<uint32_t> status{HSA_STATUS_SUCCESS};
};

static bool FileExists(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) return false;
  std::fclose(file);
  return true;
}

static std::string JoinPath(const char* lhs, const char* rhs) {
  if (!lhs || lhs[0] == 0) return std::string(rhs);
  std::string result(lhs);
  if (!result.empty() && result.back() != '/') result.push_back('/');
  result.append(rhs);
  return result;
}

static std::string FindTestCodeObjectPath() {
  std::vector<std::string> candidates;
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  const char* test_workspace = std::getenv("TEST_WORKSPACE");
  if (test_srcdir && test_workspace) {
    const std::string workspace_path = JoinPath(test_srcdir, test_workspace);
    candidates.push_back(JoinPath(workspace_path.c_str(), kTestCodeObjectPath));
  }
  if (test_srcdir) {
    const std::string main_path = JoinPath(test_srcdir, "_main");
    candidates.push_back(JoinPath(main_path.c_str(), kTestCodeObjectPath));
    candidates.push_back(JoinPath(test_srcdir, kTestCodeObjectPath));
  }
  candidates.push_back(kTestCodeObjectPath);
  candidates.push_back(JoinPath("bazel-bin", kTestCodeObjectPath));

  for (const std::string& candidate : candidates) {
    if (FileExists(candidate)) return candidate;
  }
  return candidates.front();
}

static void HsaQueueErrorCallback(hsa_status_t status, hsa_queue_t* queue,
                                  void* user_data) {
  (void)queue;
  QueueError* error = reinterpret_cast<QueueError*>(user_data);
  error->status.store(static_cast<uint32_t>(status), std::memory_order_relaxed);
  error->callback_count.fetch_add(1, std::memory_order_relaxed);
}

struct IsaQuery {
  const iree_hal_amdgpu_libhsa_t* libhsa = nullptr;
  bool supports_gfx1100 = false;
};

static hsa_status_t FindGfx1100Isa(hsa_isa_t isa, void* user_data) {
  IsaQuery* query = reinterpret_cast<IsaQuery*>(user_data);
  uint32_t name_length = 0;
  if (!iree_status_is_ok(
          iree_hsa_isa_get_info_alt(IREE_LIBHSA(query->libhsa), isa,
                                    HSA_ISA_INFO_NAME_LENGTH, &name_length))) {
    return HSA_STATUS_ERROR;
  }
  std::vector<char> name(name_length + 1);
  if (!iree_status_is_ok(iree_hsa_isa_get_info_alt(
          IREE_LIBHSA(query->libhsa), isa, HSA_ISA_INFO_NAME, name.data()))) {
    return HSA_STATUS_ERROR;
  }
  if (std::strstr(name.data(), "gfx1100")) {
    query->supports_gfx1100 = true;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

static bool AgentSupportsGfx1100(const iree_hal_amdgpu_libhsa_t* libhsa,
                                 hsa_agent_t agent) {
  IsaQuery query = {.libhsa = libhsa};
  iree_status_t status = iree_hsa_agent_iterate_isas(IREE_LIBHSA(libhsa), agent,
                                                     FindGfx1100Isa, &query);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
  }
  return query.supports_gfx1100;
}

static iree_status_t LookupKernel(const iree_hal_amdgpu_libhsa_t* libhsa,
                                  hsa_executable_t executable,
                                  hsa_agent_t agent, const char* kernel_name,
                                  KernelInfo* out_info) {
  memset(out_info, 0, sizeof(*out_info));

  char descriptor_symbol_name[128] = {0};
  std::snprintf(descriptor_symbol_name, sizeof(descriptor_symbol_name), "%s.kd",
                kernel_name);
  hsa_status_t raw_status = iree_hsa_executable_get_symbol_by_name_raw(
      libhsa, executable, descriptor_symbol_name, &agent, &out_info->symbol);
  if (raw_status != HSA_STATUS_SUCCESS) {
    IREE_RETURN_IF_ERROR(iree_hsa_executable_get_symbol_by_name(
        IREE_LIBHSA(libhsa), executable, kernel_name, &agent,
        &out_info->symbol));
  }
  IREE_RETURN_IF_ERROR(iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(libhsa), out_info->symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &out_info->kernel_object));
  IREE_RETURN_IF_ERROR(iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(libhsa), out_info->symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
      &out_info->kernarg_size));
  IREE_RETURN_IF_ERROR(iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(libhsa), out_info->symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
      &out_info->kernarg_alignment));
  IREE_RETURN_IF_ERROR(iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(libhsa), out_info->symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
      &out_info->private_segment_size));
  return iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(libhsa), out_info->symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
      &out_info->group_segment_size);
}

static void EmitDispatchDirect(uint32_t* target_dwords,
                               uint32_t workgroup_count_x,
                               uint32_t workgroup_count_y,
                               uint32_t workgroup_count_z,
                               uint32_t dispatch_initiator) {
  target_dwords[0] = iree_hal_amdgpu_pm4_make_compute_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_DISPATCH_DIRECT,
      IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT);
  target_dwords[1] = workgroup_count_x;
  target_dwords[2] = workgroup_count_y;
  target_dwords[3] = workgroup_count_z;
  target_dwords[4] = dispatch_initiator;
}

static void EmitReleaseMem32Gfx10Gfx11(uint32_t* target_dwords, void* target,
                                       uint32_t value) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(target);
  target_dwords[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_RELEASE_MEM, /*dword_count=*/8);
  target_dwords[1] = (1u << 22) | (1u << 21) | (1u << 13) | (1u << 12) |
                     (3u << 25) | (0x14u << 0) | (5u << 8);
  target_dwords[2] =
      IREE_HAL_AMDGPU_PM4_RELEASE_MEM_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM |
      (1u << 29);
  target_dwords[3] = iree_hal_amdgpu_pm4_addr_lo(address);
  target_dwords[4] = iree_hal_amdgpu_pm4_addr_hi(address);
  target_dwords[5] = value;
  target_dwords[6] = 0;
  target_dwords[7] = 0;
}

static iree_status_t AppendPm4DispatchDirect(
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t& launch_state,
    uint64_t kernarg_address, uint32_t* dwords, uint32_t capacity,
    uint32_t* inout_dword_count) {
  if (*inout_dword_count > capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "PM4 dispatch append cursor %u exceeds capacity %u",
                            *inout_dword_count, capacity);
  }
  uint32_t emitted_dword_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_emit_setup(
      &launch_state, capacity - *inout_dword_count, &dwords[*inout_dword_count],
      &emitted_dword_count));
  *inout_dword_count += emitted_dword_count;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_emit_user_data(
      &launch_state, kernarg_address, capacity - *inout_dword_count,
      &dwords[*inout_dword_count], &emitted_dword_count));
  *inout_dword_count += emitted_dword_count;
  if (capacity - *inout_dword_count <
      IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "PM4 dispatch packet requires %u dwords but only %u are available",
        IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT,
        capacity - *inout_dword_count);
  }
  EmitDispatchDirect(&dwords[*inout_dword_count], /*workgroup_count_x=*/1,
                     /*workgroup_count_y=*/1, /*workgroup_count_z=*/1,
                     launch_state.dispatch_initiator);
  *inout_dword_count += IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT;
  return iree_ok_status();
}

static void WaitForCompletionWord(volatile uint32_t* completion,
                                  uint32_t value) {
  while (*completion != value) {
    iree_processor_yield();
  }
}

class PM4DispatchLiveTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator, &libhsa);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(
        iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &topology));
    if (topology.gpu_agent_count == 0 || topology.cpu_agent_count == 0) {
      GTEST_SKIP() << "CPU and GPU agents are required, skipping tests";
    }
    if (!AgentSupportsGfx1100(&libhsa, topology.gpu_agents[0])) {
      GTEST_SKIP() << "gfx1100 GPU is required for this PM4 dispatch smoke";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }

  static iree_allocator_t host_allocator;
  static iree_hal_amdgpu_libhsa_t libhsa;
  static iree_hal_amdgpu_topology_t topology;
};

iree_allocator_t PM4DispatchLiveTest::host_allocator;
iree_hal_amdgpu_libhsa_t PM4DispatchLiveTest::libhsa;
iree_hal_amdgpu_topology_t PM4DispatchLiveTest::topology;

TEST_F(PM4DispatchLiveTest, AqlAndAqlPm4IbLaunchMixedKernels) {
  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_agent_t gpu_agent = topology.gpu_agents[0];

  std::string code_object_path = FindTestCodeObjectPath();
  iree_io_file_contents_t* code_object_contents = nullptr;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      iree_make_cstring_view(code_object_path.c_str()), host_allocator,
      &code_object_contents));

  hsa_code_object_reader_t code_object_reader = {0};
  IREE_ASSERT_OK(iree_hsa_code_object_reader_create_from_memory(
      IREE_LIBHSA(&libhsa), code_object_contents->const_buffer.data,
      code_object_contents->const_buffer.data_length, &code_object_reader));

  hsa_executable_t executable = {0};
  IREE_ASSERT_OK(
      iree_hsa_executable_create_alt(IREE_LIBHSA(&libhsa), HSA_PROFILE_FULL,
                                     HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                     /*options=*/nullptr, &executable));
  hsa_loaded_code_object_t loaded_code_object = {0};
  IREE_ASSERT_OK(iree_hsa_executable_load_agent_code_object(
      IREE_LIBHSA(&libhsa), executable, gpu_agent, code_object_reader,
      /*options=*/nullptr, &loaded_code_object));
  IREE_ASSERT_OK(iree_hsa_executable_freeze(IREE_LIBHSA(&libhsa), executable,
                                            /*options=*/nullptr));
  IREE_ASSERT_OK(iree_hsa_code_object_reader_destroy(IREE_LIBHSA(&libhsa),
                                                     code_object_reader));
  code_object_reader = {0};
  iree_io_file_contents_free(code_object_contents);
  code_object_contents = nullptr;

  KernelInfo kernels[4];
  IREE_ASSERT_OK(LookupKernel(&libhsa, executable, gpu_agent,
                              "iree_hal_amdgpu_pm4_dispatch_test_store_a",
                              &kernels[0]));
  IREE_ASSERT_OK(LookupKernel(&libhsa, executable, gpu_agent,
                              "iree_hal_amdgpu_pm4_dispatch_test_store_b",
                              &kernels[1]));
  IREE_ASSERT_OK(LookupKernel(&libhsa, executable, gpu_agent,
                              "iree_hal_amdgpu_pm4_dispatch_test_read_add",
                              &kernels[2]));
  IREE_ASSERT_OK(LookupKernel(
      &libhsa, executable, gpu_agent,
      "iree_hal_amdgpu_pm4_dispatch_test_patch_user_data", &kernels[3]));
  for (const KernelInfo& kernel : kernels) {
    EXPECT_LE(kernel.kernarg_size, sizeof(PatchUserDataKernargs));
    EXPECT_LE(kernel.kernarg_alignment, alignof(LiveMemory));
    EXPECT_EQ(kernel.private_segment_size, 0u);
    EXPECT_EQ(kernel.group_segment_size, 0u);
  }

  hsa_amd_memory_pool_t host_memory_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_fine_global_memory_pool(
      &libhsa, cpu_agent, &host_memory_pool));
  LiveMemory* memory = nullptr;
  IREE_ASSERT_OK(iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(&libhsa), host_memory_pool, sizeof(LiveMemory),
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG, reinterpret_cast<void**>(&memory)));
  IREE_ASSERT_OK(iree_hsa_amd_agents_allow_access(IREE_LIBHSA(&libhsa),
                                                  /*num_agents=*/1, &gpu_agent,
                                                  /*flags=*/nullptr, memory));

  hsa_amd_memory_pool_t pm4_memory_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &pm4_memory_pool));

  QueueError queue_error;
  hsa_queue_t* queue = nullptr;
  IREE_ASSERT_OK(iree_hsa_queue_create(
      IREE_LIBHSA(&libhsa), gpu_agent, /*size=*/64, HSA_QUEUE_TYPE_MULTI,
      HsaQueueErrorCallback, &queue_error, UINT32_MAX, UINT32_MAX, &queue));
  iree_hal_amdgpu_aql_ring_t aql_ring;
  iree_hal_amdgpu_aql_ring_initialize(
      reinterpret_cast<iree_amd_queue_t*>(queue), &aql_ring);

  hsa_signal_t completion_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/nullptr, /*attributes=*/0, &completion_signal));

  memset(memory, 0, sizeof(*memory));
  memory->store_kernargs[0] = {.target = &memory->outputs[0],
                               .value = kAqlValueA};
  memory->store_kernargs[1] = {.target = &memory->outputs[1],
                               .value = kAqlValueB};

  const uint64_t aql_first_packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&aql_ring, /*count=*/2);
  for (uint32_t i = 0; i < 2; ++i) {
    iree_hal_amdgpu_aql_packet_t* packet =
        iree_hal_amdgpu_aql_ring_packet(&aql_ring, aql_first_packet_id + i);
    memset(packet, 0, sizeof(*packet));
    uint16_t setup = 0;
    const uint16_t header = iree_hal_amdgpu_aql_emit_dispatch(
        &packet->dispatch, kernels[i].kernel_object, &memory->store_kernargs[i],
        kWorkgroupSize, kDispatchGridSize, kernels[i].private_segment_size,
        kernels[i].group_segment_size,
        iree_hal_amdgpu_aql_packet_control_barrier_system(),
        i == 1 ? completion_signal : iree_hsa_signal_null(), &setup);
    iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
  }
  iree_hal_amdgpu_aql_ring_doorbell(&aql_ring, aql_first_packet_id + 1);
  EXPECT_EQ(
      iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(&libhsa), completion_signal, HSA_SIGNAL_CONDITION_EQ,
          /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
      0);
  EXPECT_EQ(memory->outputs[0], kAqlValueA);
  EXPECT_EQ(memory->outputs[1], kAqlValueB + 0x100u);

  iree_hal_amdgpu_pm4_dispatch_launch_state_t launch_states[4];
  for (uint32_t i = 0; i < 4; ++i) {
    const iree_hal_amdgpu_kernel_descriptor_t* descriptor =
        reinterpret_cast<const iree_hal_amdgpu_kernel_descriptor_t*>(
            static_cast<uintptr_t>(kernels[i].kernel_object));
    IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize_gfx10(
        descriptor, kernels[i].kernel_object, kWorkgroupSize,
        IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE,
        &launch_states[i]));
  }

  uint32_t pm4_dwords[256] = {0};
  uint32_t pm4_dword_count = 0;
  memset(memory, 0, sizeof(*memory));
  memory->store_kernargs[0] = {.target = &memory->outputs[0],
                               .value = kPm4ValueA};
  memory->store_kernargs[1] = {.target = &memory->outputs[1],
                               .value = kPm4ValueB};

  for (uint32_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(AppendPm4DispatchDirect(
        launch_states[i],
        reinterpret_cast<uintptr_t>(&memory->store_kernargs[i]), pm4_dwords,
        IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  }
  ASSERT_LE(pm4_dword_count + 8, IREE_ARRAYSIZE(pm4_dwords));
  EmitReleaseMem32Gfx10Gfx11(&pm4_dwords[pm4_dword_count], &memory->completion,
                             /*value=*/1);
  pm4_dword_count += 8;

  iree_hal_amdgpu_pm4_program_t pm4_program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_initialize(
      &libhsa, gpu_agent, pm4_memory_pool, pm4_dwords, pm4_dword_count,
      &pm4_program));

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa), completion_signal, 1);
  const uint64_t pm4_packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&aql_ring, /*count=*/1);
  iree_hal_amdgpu_aql_packet_t* pm4_packet =
      iree_hal_amdgpu_aql_ring_packet(&aql_ring, pm4_packet_id);
  memset(pm4_packet, 0, sizeof(*pm4_packet));
  uint16_t pm4_setup = 0;
  const uint16_t pm4_header = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
      &pm4_packet->pm4_ib, pm4_program.dwords, pm4_program.dword_count,
      iree_hal_amdgpu_aql_packet_control_barrier_system(), completion_signal,
      &pm4_setup);
  iree_hal_amdgpu_aql_ring_commit(pm4_packet, pm4_header, pm4_setup);
  iree_hal_amdgpu_aql_ring_doorbell(&aql_ring, pm4_packet_id);
  EXPECT_EQ(
      iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(&libhsa), completion_signal, HSA_SIGNAL_CONDITION_EQ,
          /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
      0);
  WaitForCompletionWord(&memory->completion, /*value=*/1);
  EXPECT_EQ(memory->outputs[0], kPm4ValueA);
  EXPECT_EQ(memory->outputs[1], kPm4ValueB + 0x100u);
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_release(&pm4_program));

  memset(memory, 0, sizeof(*memory));
  pm4_dword_count = 0;
  memory->store_kernargs[0] = {.target = &memory->scratch[0],
                               .value = kPm4BarrierValue};
  memory->read_add_kernargs = {.source = &memory->scratch[0],
                               .target = &memory->outputs[2],
                               .value = kPm4BarrierAdd};
  IREE_ASSERT_OK(AppendPm4DispatchDirect(
      launch_states[0], reinterpret_cast<uintptr_t>(&memory->store_kernargs[0]),
      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  uint32_t barrier_dword_count = 0;
  ASSERT_TRUE(iree_hal_amdgpu_pm4_barrier_emit_gfx10(
      kBarrierCapabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
      IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_SYSTEM,
      IREE_ARRAYSIZE(pm4_dwords) - pm4_dword_count,
      &pm4_dwords[pm4_dword_count], &barrier_dword_count));
  pm4_dword_count += barrier_dword_count;
  IREE_ASSERT_OK(AppendPm4DispatchDirect(
      launch_states[2], reinterpret_cast<uintptr_t>(&memory->read_add_kernargs),
      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  ASSERT_LE(pm4_dword_count + 8, IREE_ARRAYSIZE(pm4_dwords));
  EmitReleaseMem32Gfx10Gfx11(&pm4_dwords[pm4_dword_count], &memory->completion,
                             /*value=*/2);
  pm4_dword_count += 8;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_initialize(
      &libhsa, gpu_agent, pm4_memory_pool, pm4_dwords, pm4_dword_count,
      &pm4_program));

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa), completion_signal, 1);
  memory->completion = 0;
  const uint64_t barrier_pm4_packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&aql_ring, /*count=*/1);
  pm4_packet =
      iree_hal_amdgpu_aql_ring_packet(&aql_ring, barrier_pm4_packet_id);
  memset(pm4_packet, 0, sizeof(*pm4_packet));
  pm4_setup = 0;
  const uint16_t barrier_pm4_header = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
      &pm4_packet->pm4_ib, pm4_program.dwords, pm4_program.dword_count,
      iree_hal_amdgpu_aql_packet_control_barrier_system(), completion_signal,
      &pm4_setup);
  iree_hal_amdgpu_aql_ring_commit(pm4_packet, barrier_pm4_header, pm4_setup);
  iree_hal_amdgpu_aql_ring_doorbell(&aql_ring, barrier_pm4_packet_id);
  EXPECT_EQ(
      iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(&libhsa), completion_signal, HSA_SIGNAL_CONDITION_EQ,
          /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
      0);
  WaitForCompletionWord(&memory->completion, /*value=*/2);
  EXPECT_EQ(memory->scratch[0], kPm4BarrierValue);
  EXPECT_EQ(memory->outputs[2], kPm4BarrierValue + kPm4BarrierAdd);
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_release(&pm4_program));

  memset(memory, 0, sizeof(*memory));
  memory->store_kernargs[2] = {.target = &memory->outputs[3],
                               .value = kPm4PatchWrongValue};
  memory->store_kernargs[3] = {.target = &memory->outputs[2],
                               .value = kPm4PatchValue};
  pm4_dword_count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_emit_setup(
      &launch_states[1], IREE_ARRAYSIZE(pm4_dwords) - pm4_dword_count,
      &pm4_dwords[pm4_dword_count], &barrier_dword_count));
  pm4_dword_count += barrier_dword_count;
  const uint32_t patched_user_data_offset = pm4_dword_count + 2;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_emit_user_data(
      &launch_states[1],
      reinterpret_cast<uintptr_t>(&memory->store_kernargs[2]),
      IREE_ARRAYSIZE(pm4_dwords) - pm4_dword_count,
      &pm4_dwords[pm4_dword_count], &barrier_dword_count));
  pm4_dword_count += barrier_dword_count;
  ASSERT_LE(pm4_dword_count + IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT,
            IREE_ARRAYSIZE(pm4_dwords));
  EmitDispatchDirect(&pm4_dwords[pm4_dword_count],
                     /*workgroup_count_x=*/1, /*workgroup_count_y=*/1,
                     /*workgroup_count_z=*/1,
                     launch_states[1].dispatch_initiator);
  pm4_dword_count += IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT;
  ASSERT_LE(pm4_dword_count + 8, IREE_ARRAYSIZE(pm4_dwords));
  EmitReleaseMem32Gfx10Gfx11(&pm4_dwords[pm4_dword_count], &memory->completion,
                             /*value=*/3);
  pm4_dword_count += 8;
  iree_hal_amdgpu_pm4_program_t target_pm4_program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_initialize(
      &libhsa, gpu_agent, pm4_memory_pool, pm4_dwords, pm4_dword_count,
      &target_pm4_program));

  memory->patch_user_data_kernargs = {
      .target_dwords = target_pm4_program.dwords,
      .dword_offset = patched_user_data_offset,
      .kernarg_address =
          reinterpret_cast<uintptr_t>(&memory->store_kernargs[3]),
  };

  // The target userdata lives in a following IB. Later dwords in the current IB
  // may already be fetched by the command processor before a shader fixup runs.
  pm4_dword_count = 0;
  IREE_ASSERT_OK(AppendPm4DispatchDirect(
      launch_states[3],
      reinterpret_cast<uintptr_t>(&memory->patch_user_data_kernargs),
      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  barrier_dword_count = 0;
  ASSERT_TRUE(iree_hal_amdgpu_pm4_barrier_emit_gfx10(
      kBarrierCapabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_ARRAYSIZE(pm4_dwords) - pm4_dword_count,
      &pm4_dwords[pm4_dword_count], &barrier_dword_count));
  pm4_dword_count += barrier_dword_count;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_initialize(
      &libhsa, gpu_agent, pm4_memory_pool, pm4_dwords, pm4_dword_count,
      &pm4_program));

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa), completion_signal, 1);
  memory->completion = 0;
  const uint64_t fixup_pm4_packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&aql_ring, /*count=*/2);
  pm4_packet = iree_hal_amdgpu_aql_ring_packet(&aql_ring, fixup_pm4_packet_id);
  memset(pm4_packet, 0, sizeof(*pm4_packet));
  pm4_setup = 0;
  const uint16_t fixup_pm4_header = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
      &pm4_packet->pm4_ib, pm4_program.dwords, pm4_program.dword_count,
      iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_SYSTEM,
                                                 IREE_HSA_FENCE_SCOPE_NONE),
      iree_hsa_signal_null(), &pm4_setup);
  iree_hal_amdgpu_aql_ring_commit(pm4_packet, fixup_pm4_header, pm4_setup);
  pm4_packet =
      iree_hal_amdgpu_aql_ring_packet(&aql_ring, fixup_pm4_packet_id + 1);
  memset(pm4_packet, 0, sizeof(*pm4_packet));
  pm4_setup = 0;
  const uint16_t target_pm4_header = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
      &pm4_packet->pm4_ib, target_pm4_program.dwords,
      target_pm4_program.dword_count,
      iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_NONE,
                                                 IREE_HSA_FENCE_SCOPE_NONE),
      completion_signal, &pm4_setup);
  iree_hal_amdgpu_aql_ring_commit(pm4_packet, target_pm4_header, pm4_setup);
  iree_hal_amdgpu_aql_ring_doorbell(&aql_ring, fixup_pm4_packet_id + 1);
  EXPECT_EQ(
      iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(&libhsa), completion_signal, HSA_SIGNAL_CONDITION_EQ,
          /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
      0);
  WaitForCompletionWord(&memory->completion, /*value=*/3);
  EXPECT_EQ(memory->outputs[2], kPm4PatchValue + 0x100u);
  EXPECT_EQ(memory->outputs[3], 0u);
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_release(&pm4_program));
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_release(&target_pm4_program));
  EXPECT_EQ(queue_error.callback_count.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(queue_error.status.load(std::memory_order_relaxed),
            static_cast<uint32_t>(HSA_STATUS_SUCCESS));

  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_release(&pm4_program));
  IREE_ASSERT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa), completion_signal));
  IREE_ASSERT_OK(iree_hsa_queue_destroy(IREE_LIBHSA(&libhsa), queue));
  IREE_ASSERT_OK(iree_hsa_amd_memory_pool_free(IREE_LIBHSA(&libhsa), memory));
  IREE_ASSERT_OK(iree_hsa_executable_destroy(IREE_LIBHSA(&libhsa), executable));
}

}  // namespace
}  // namespace iree::hal::amdgpu
