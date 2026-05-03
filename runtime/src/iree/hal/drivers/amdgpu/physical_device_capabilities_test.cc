// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/physical_device_capabilities.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static hsa_agent_t Agent(uint64_t handle) {
  hsa_agent_t agent = {};
  agent.handle = handle;
  return agent;
}

static hsa_amd_memory_pool_t MemoryPool(uint64_t handle) {
  hsa_amd_memory_pool_t memory_pool = {};
  memory_pool.handle = handle;
  return memory_pool;
}

static hsa_amd_hdp_flush_t HdpFlush(uintptr_t mem_flush_control,
                                    uintptr_t register_flush_control) {
  hsa_amd_hdp_flush_t hdp_flush = {};
  hdp_flush.HDP_MEM_FLUSH_CNTL = reinterpret_cast<uint32_t*>(mem_flush_control);
  hdp_flush.HDP_REG_FLUSH_CNTL =
      reinterpret_cast<uint32_t*>(register_flush_control);
  return hdp_flush;
}

static iree_hal_amdgpu_gfxip_version_t GfxIp(uint16_t major, uint16_t minor,
                                             uint16_t stepping) {
  iree_hal_amdgpu_gfxip_version_t version = {};
  version.major = major;
  version.minor = minor;
  version.stepping = stepping;
  return version;
}

static iree_hal_amdgpu_gfxip_version_t GfxIpFromProcessor(
    const char* processor) {
  iree_hal_amdgpu_target_id_t target_id = {};
  IREE_CHECK_OK(iree_hal_amdgpu_target_id_parse(
      iree_make_cstring_view(processor),
      IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_NONE, &target_id));
  return target_id.version;
}

static hsa_amd_memory_pool_link_info_t LinkInfo(
    hsa_amd_link_info_type_t link_type) {
  hsa_amd_memory_pool_link_info_t link_info = {};
  link_info.link_type = link_type;
  link_info.atomic_support_32bit = true;
  link_info.atomic_support_64bit = true;
  link_info.coherent_support = true;
  return link_info;
}

class PhysicalDeviceCapabilitiesTest : public ::testing::Test {
 protected:
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t
  MakeCoarseMemorySelection() {
    iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection = {};
    selection.device_agent = Agent(10);
    selection.memory_pool = MemoryPool(20);
    selection.gfxip_version = GfxIp(11, 0, 0);
    selection.cpu.agents = cpu_agents_.data();
    selection.cpu.access = cpu_access_.data();
    selection.cpu.count = cpu_agents_.size();
    selection.hdp.registers = HdpFlush(0xCAFE, 0xBEEF);
    selection.flags =
        IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_HOST_WRITE_PUBLICATION_SUPPORTED;
    return selection;
  }

  iree_hal_amdgpu_memory_system_capabilities_selection_t
  MakeMemorySystemSelection() {
    iree_hal_amdgpu_memory_system_capabilities_selection_t selection = {};
    selection.svm.supported = 1;
    selection.svm.accessible_by_default = 0;
    selection.svm.xnack_enabled = 0;
    selection.svm.direct_host_access = 0;
    selection.device_local.fine_memory_pool = MemoryPool(30);
    selection.device_local.coarse_cpu_visible_memory = nullptr;
    return selection;
  }

  iree_hal_amdgpu_physical_topology_edge_selection_t MakeTopologyEdgeSelection(
      const hsa_amd_memory_pool_link_info_t* link_hops,
      iree_host_size_t link_hop_count) {
    iree_hal_amdgpu_physical_topology_edge_selection_t selection = {};
    selection.memory_access.coarse =
        HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT;
    selection.memory_access.fine =
        HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT;
    selection.link.hops = link_hops;
    selection.link.count = link_hop_count;
    return selection;
  }

  std::array<hsa_agent_t, 2> cpu_agents_ = {Agent(1), Agent(2)};
  std::array<hsa_amd_memory_pool_access_t, 2> cpu_access_ = {
      HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT,
      HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT};
};

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsAvailableCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));

  EXPECT_TRUE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));
  EXPECT_EQ(capability.memory_pool.handle, selection.memory_pool.handle);
  ASSERT_EQ(capability.access_agent_count, 3u);
  EXPECT_EQ(capability.access_agents[0].handle, cpu_agents_[0].handle);
  EXPECT_EQ(capability.access_agents[1].handle, cpu_agents_[1].handle);
  EXPECT_EQ(capability.access_agents[2].handle, selection.device_agent.handle);
  EXPECT_EQ(capability.host_write_publication.mode,
            IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_HDP_FLUSH);
  EXPECT_EQ(capability.host_write_publication.hdp_mem_flush_control,
            selection.hdp.registers.HDP_MEM_FLUSH_CNTL);
  EXPECT_TRUE(iree_all_bits_set(
      capability.flags,
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE |
          IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_HDP_FLUSH));
}

TEST_F(PhysicalDeviceCapabilitiesTest, EmptyInputsDisableCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;

  selection.memory_pool = MemoryPool(0);
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));

  selection = MakeCoarseMemorySelection();
  selection.cpu.count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, PublicationGatesDisableCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;

  selection.flags =
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_SELECTION_FLAG_NONE;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));

  selection = MakeCoarseMemorySelection();
  selection.hdp.registers.HDP_MEM_FLUSH_CNTL = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));

  selection = MakeCoarseMemorySelection();
  selection.hdp.registers.HDP_REG_FLUSH_CNTL = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, GfxIpGatesHdpPublication) {
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(9, 0, 7)));
  EXPECT_TRUE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(9, 0, 8)));
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(10, 0, 0)));
  EXPECT_FALSE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(10, 1, 0)));
  EXPECT_TRUE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(10, 3, 0)));
  EXPECT_TRUE(
      iree_hal_amdgpu_gfxip_allows_hdp_kernarg_publication(GfxIp(11, 0, 0)));
}

TEST_F(PhysicalDeviceCapabilitiesTest, UnsupportedGfxIpDisablesCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  selection.gfxip_version = GfxIp(10, 1, 0);
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, CpuAccessGatesCoarseMemory) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;

  cpu_access_[1] = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  IREE_ASSERT_OK(iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
      &selection, &capability));
  EXPECT_FALSE(iree_hal_amdgpu_cpu_visible_device_coarse_memory_is_available(
      &capability));

  cpu_access_[1] = (hsa_amd_memory_pool_access_t)99;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
                            &selection, &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       MemoryPoolAccessMapsToSafeTopologyModes) {
  EXPECT_TRUE(iree_hal_amdgpu_memory_pool_access_is_valid(
      HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED));
  EXPECT_EQ(iree_hal_amdgpu_memory_pool_access_topology_mode(
                HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_amdgpu_memory_pool_access_topology_capabilities(
                HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED),
            IREE_HAL_TOPOLOGY_CAPABILITY_NONE);

  EXPECT_TRUE(iree_hal_amdgpu_memory_pool_access_is_valid(
      HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT));
  EXPECT_EQ(iree_hal_amdgpu_memory_pool_access_topology_mode(
                HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(iree_hal_amdgpu_memory_pool_access_topology_capabilities(
                HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT),
            IREE_HAL_TOPOLOGY_CAPABILITY_NONE);

  EXPECT_TRUE(iree_hal_amdgpu_memory_pool_access_is_valid(
      HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT));
  EXPECT_EQ(iree_hal_amdgpu_memory_pool_access_topology_mode(
                HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT),
            IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(iree_hal_amdgpu_memory_pool_access_topology_capabilities(
                HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT),
            IREE_HAL_TOPOLOGY_CAPABILITY_PEER_ACCESS_REQUIRES_GRANT);

  EXPECT_FALSE(iree_hal_amdgpu_memory_pool_access_is_valid(
      (hsa_amd_memory_pool_access_t)99));
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsXgmiPhysicalTopologyEdge) {
  std::array<hsa_amd_memory_pool_link_info_t, 1> link_hops = {
      LinkInfo(HSA_AMD_LINK_INFO_TYPE_XGMI)};
  link_hops[0].numa_distance = 16;

  iree_hal_amdgpu_physical_topology_edge_selection_t selection =
      MakeTopologyEdgeSelection(link_hops.data(), link_hops.size());
  iree_hal_amdgpu_physical_topology_edge_t edge;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));

  EXPECT_EQ(edge.memory_access.coarse,
            HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT);
  EXPECT_EQ(edge.memory_access.fine,
            HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT);
  EXPECT_TRUE(edge.memory_access.coarse_accessible);
  EXPECT_TRUE(edge.memory_access.fine_accessible);
  EXPECT_TRUE(edge.coherency.all_hops_coherent);
  EXPECT_TRUE(edge.atomics.all_hops_32bit);
  EXPECT_TRUE(edge.atomics.all_hops_64bit);
  EXPECT_TRUE(iree_any_bit_set(
      edge.link.flags, IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_XGMI));
  EXPECT_EQ(edge.link.link_class, IREE_HAL_TOPOLOGY_LINK_CLASS_NVLINK_IF);
  EXPECT_EQ(edge.link.copy_cost, 3);
  EXPECT_EQ(edge.link.latency_class, 3);
  EXPECT_EQ(edge.link.numa_distance, 3);
  EXPECT_TRUE(
      iree_all_bits_set(edge.capabilities.guaranteed,
                        IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY |
                            IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT |
                            IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE |
                            IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM));
  EXPECT_EQ(edge.capabilities.required, IREE_HAL_TOPOLOGY_CAPABILITY_NONE);
  EXPECT_EQ(edge.modes.noncoherent_read, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(edge.modes.coherent_read, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       SelectsWorstMultiHopPhysicalTopologyEdge) {
  std::array<hsa_amd_memory_pool_link_info_t, 2> link_hops = {
      LinkInfo(HSA_AMD_LINK_INFO_TYPE_XGMI),
      LinkInfo(HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT)};
  link_hops[0].numa_distance = 12;
  link_hops[1].numa_distance = 28;
  link_hops[1].atomic_support_32bit = false;
  link_hops[1].coherent_support = false;

  iree_hal_amdgpu_physical_topology_edge_selection_t selection =
      MakeTopologyEdgeSelection(link_hops.data(), link_hops.size());
  iree_hal_amdgpu_physical_topology_edge_t edge;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));

  EXPECT_FALSE(edge.coherency.all_hops_coherent);
  EXPECT_FALSE(edge.atomics.all_hops_32bit);
  EXPECT_TRUE(edge.atomics.all_hops_64bit);
  EXPECT_TRUE(iree_all_bits_set(
      edge.link.flags,
      IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_XGMI |
          IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_HYPERTRANSPORT));
  EXPECT_EQ(edge.link.link_class, IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_CROSS_ROOT);
  EXPECT_EQ(edge.link.copy_cost, 9);
  EXPECT_EQ(edge.link.latency_class, 9);
  EXPECT_EQ(edge.link.numa_distance, 9);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       SelectsPciePhysicalTopologyEdgeWithoutSystemAtomics) {
  std::array<hsa_amd_memory_pool_link_info_t, 1> link_hops = {
      LinkInfo(HSA_AMD_LINK_INFO_TYPE_PCIE)};
  link_hops[0].atomic_support_64bit = false;
  link_hops[0].coherent_support = false;

  iree_hal_amdgpu_physical_topology_edge_selection_t selection =
      MakeTopologyEdgeSelection(link_hops.data(), link_hops.size());
  selection.memory_access.fine = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  iree_hal_amdgpu_physical_topology_edge_t edge;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));

  EXPECT_TRUE(edge.memory_access.coarse_accessible);
  EXPECT_FALSE(edge.memory_access.fine_accessible);
  EXPECT_FALSE(edge.coherency.all_hops_coherent);
  EXPECT_TRUE(edge.atomics.all_hops_32bit);
  EXPECT_FALSE(edge.atomics.all_hops_64bit);
  EXPECT_TRUE(iree_any_bit_set(
      edge.link.flags, IREE_HAL_AMDGPU_PHYSICAL_TOPOLOGY_LINK_FLAG_PCIE));
  EXPECT_EQ(edge.link.link_class, IREE_HAL_TOPOLOGY_LINK_CLASS_PCIE_SAME_ROOT);
  EXPECT_EQ(edge.link.copy_cost, 7);
  EXPECT_EQ(edge.link.latency_class, 7);
  EXPECT_TRUE(iree_any_bit_set(edge.capabilities.guaranteed,
                               IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY));
  EXPECT_FALSE(iree_any_bit_set(edge.capabilities.guaranteed,
                                IREE_HAL_TOPOLOGY_CAPABILITY_PEER_COHERENT));
  EXPECT_TRUE(iree_any_bit_set(edge.capabilities.guaranteed,
                               IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_DEVICE));
  EXPECT_FALSE(iree_any_bit_set(edge.capabilities.guaranteed,
                                IREE_HAL_TOPOLOGY_CAPABILITY_ATOMIC_SYSTEM));
  EXPECT_EQ(edge.modes.noncoherent_read, IREE_HAL_TOPOLOGY_INTEROP_MODE_NATIVE);
  EXPECT_EQ(edge.modes.coherent_read, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       GrantablePhysicalTopologyEdgeRequiresGrant) {
  std::array<hsa_amd_memory_pool_link_info_t, 1> link_hops = {
      LinkInfo(HSA_AMD_LINK_INFO_TYPE_PCIE)};
  iree_hal_amdgpu_physical_topology_edge_selection_t selection =
      MakeTopologyEdgeSelection(link_hops.data(), link_hops.size());
  selection.memory_access.coarse =
      HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT;
  selection.memory_access.fine =
      HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT;

  iree_hal_amdgpu_physical_topology_edge_t edge;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));

  EXPECT_TRUE(edge.memory_access.coarse_accessible);
  EXPECT_TRUE(edge.memory_access.fine_accessible);
  EXPECT_TRUE(iree_any_bit_set(edge.capabilities.guaranteed,
                               IREE_HAL_TOPOLOGY_CAPABILITY_P2P_COPY));
  EXPECT_TRUE(iree_any_bit_set(
      edge.capabilities.required,
      IREE_HAL_TOPOLOGY_CAPABILITY_PEER_ACCESS_REQUIRES_GRANT));
  EXPECT_EQ(edge.modes.noncoherent_read, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
  EXPECT_EQ(edge.modes.coherent_read, IREE_HAL_TOPOLOGY_INTEROP_MODE_COPY);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       NeverAllowedPhysicalTopologyEdgeIsHostStaged) {
  std::array<hsa_amd_memory_pool_link_info_t, 1> link_hops = {
      LinkInfo(HSA_AMD_LINK_INFO_TYPE_XGMI)};
  iree_hal_amdgpu_physical_topology_edge_selection_t selection =
      MakeTopologyEdgeSelection(link_hops.data(), link_hops.size());
  selection.memory_access.coarse = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
  selection.memory_access.fine = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;

  iree_hal_amdgpu_physical_topology_edge_t edge;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));

  EXPECT_FALSE(edge.memory_access.coarse_accessible);
  EXPECT_FALSE(edge.memory_access.fine_accessible);
  EXPECT_FALSE(edge.coherency.all_hops_coherent);
  EXPECT_FALSE(edge.atomics.all_hops_32bit);
  EXPECT_FALSE(edge.atomics.all_hops_64bit);
  EXPECT_EQ(edge.link.link_class, IREE_HAL_TOPOLOGY_LINK_CLASS_HOST_STAGED);
  EXPECT_EQ(edge.link.copy_cost, 13);
  EXPECT_EQ(edge.link.latency_class, 11);
  EXPECT_EQ(edge.capabilities.guaranteed, IREE_HAL_TOPOLOGY_CAPABILITY_NONE);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       InvalidPhysicalTopologyEdgeInputsFailLoud) {
  iree_hal_amdgpu_physical_topology_edge_selection_t selection =
      MakeTopologyEdgeSelection(nullptr, 1);
  iree_hal_amdgpu_physical_topology_edge_t edge;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));

  selection = MakeTopologyEdgeSelection(nullptr, 0);
  selection.memory_access.coarse = (hsa_amd_memory_pool_access_t)99;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_select_physical_topology_edge(&selection, &edge));
}

TEST_F(PhysicalDeviceCapabilitiesTest, CpuAccessInputsAreRequiredWhenNeeded) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  selection.cpu.agents = nullptr;
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
                            &selection, &capability));

  selection = MakeCoarseMemorySelection();
  selection.cpu.access = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
                            &selection, &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, TooManyCpuAgentsFails) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_selection_t selection =
      MakeCoarseMemorySelection();
  selection.cpu.count = IREE_HAL_AMDGPU_MAX_CPU_AGENT + 1;
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t capability;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_select_cpu_visible_device_coarse_memory(
                            &selection, &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, SvmDefaultAccessDoesNotImplyPeerFlags) {
  iree_hal_amdgpu_memory_system_capabilities_selection_t selection =
      MakeMemorySystemSelection();
  selection.svm.accessible_by_default = 1;
  selection.svm.xnack_enabled = 1;

  iree_hal_amdgpu_memory_system_capabilities_t capability;
  iree_hal_amdgpu_select_memory_system_capabilities(&selection, &capability);

  EXPECT_TRUE(capability.svm.supported);
  EXPECT_TRUE(capability.svm.accessible_by_default);
  EXPECT_TRUE(capability.svm.xnack_enabled);
  EXPECT_FALSE(capability.svm.direct_host_access);
  EXPECT_TRUE(capability.device_local.fine_host_visible);
  EXPECT_FALSE(capability.device_local.coarse_cpu_visible);

  iree_hal_device_capability_bits_t flags =
      iree_hal_amdgpu_select_memory_system_device_capability_flags(&capability);
  EXPECT_TRUE(flags & IREE_HAL_DEVICE_CAPABILITY_SHARED_VIRTUAL_ADDRESS);
  EXPECT_TRUE(flags & IREE_HAL_DEVICE_CAPABILITY_UNIFIED_MEMORY);
  EXPECT_FALSE(flags & IREE_HAL_DEVICE_CAPABILITY_PEER_ADDRESSABLE);
  EXPECT_FALSE(flags & IREE_HAL_DEVICE_CAPABILITY_PEER_COHERENT);
  EXPECT_FALSE(iree_hal_amdgpu_memory_system_requires_svm_access_attributes(
      &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       LargeBarDoesNotImplyPageableSvmDefaultAccess) {
  iree_hal_amdgpu_cpu_visible_device_coarse_memory_t coarse_memory = {};
  coarse_memory.memory_pool = MemoryPool(40);
  coarse_memory.flags =
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE |
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_HDP_FLUSH;

  iree_hal_amdgpu_memory_system_capabilities_selection_t selection =
      MakeMemorySystemSelection();
  selection.svm.direct_host_access = 1;
  selection.device_local.coarse_cpu_visible_memory = &coarse_memory;

  iree_hal_amdgpu_memory_system_capabilities_t capability;
  iree_hal_amdgpu_select_memory_system_capabilities(&selection, &capability);

  EXPECT_TRUE(capability.svm.supported);
  EXPECT_FALSE(capability.svm.accessible_by_default);
  EXPECT_FALSE(capability.svm.xnack_enabled);
  EXPECT_TRUE(capability.svm.direct_host_access);
  EXPECT_TRUE(capability.device_local.fine_host_visible);
  EXPECT_TRUE(capability.device_local.coarse_cpu_visible);

  iree_hal_device_capability_bits_t flags =
      iree_hal_amdgpu_select_memory_system_device_capability_flags(&capability);
  EXPECT_TRUE(flags & IREE_HAL_DEVICE_CAPABILITY_SHARED_VIRTUAL_ADDRESS);
  EXPECT_FALSE(flags & IREE_HAL_DEVICE_CAPABILITY_UNIFIED_MEMORY);
  EXPECT_FALSE(flags & IREE_HAL_DEVICE_CAPABILITY_PEER_ADDRESSABLE);
  EXPECT_FALSE(flags & IREE_HAL_DEVICE_CAPABILITY_PEER_COHERENT);
  EXPECT_TRUE(iree_hal_amdgpu_memory_system_requires_svm_access_attributes(
      &capability));
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsPrepublishedKernargStorage) {
  iree_hal_amdgpu_aql_prepublished_kernarg_storage_t storage =
      iree_hal_amdgpu_select_prepublished_kernarg_storage(MemoryPool(0));
  EXPECT_EQ(storage.strategy,
            IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_STRATEGY_DISABLED);

  storage = iree_hal_amdgpu_select_prepublished_kernarg_storage(MemoryPool(42));
  EXPECT_EQ(
      storage.strategy,
      IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_STRATEGY_DEVICE_FINE_HOST_COHERENT);
  EXPECT_TRUE(iree_all_bits_set(storage.buffer_params.type,
                                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsCdnaBarrierValueCapabilities) {
  iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities =
      iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(9, 0, 10));
  EXPECT_TRUE(iree_all_bits_set(
      capabilities,
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE));
  EXPECT_FALSE(iree_any_bit_set(
      capabilities, IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_WAIT_REG_MEM64));

  capabilities =
      iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(9, 4, 2));
  EXPECT_TRUE(iree_all_bits_set(
      capabilities,
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE));

  capabilities =
      iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(9, 5, 2));
  EXPECT_TRUE(iree_all_bits_set(
      capabilities,
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE));

  capabilities =
      iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(9, 5, 0));
  EXPECT_TRUE(iree_all_bits_set(
      capabilities,
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE));

  capabilities =
      iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(9, 4, 3));
  EXPECT_EQ(capabilities, IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB);

  capabilities =
      iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(9, 5, 3));
  EXPECT_EQ(capabilities, IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB);
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsValidatedGfx1100Capabilities) {
  iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities =
      iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(11, 0, 0));
  EXPECT_TRUE(iree_all_bits_set(
      capabilities,
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_WAIT_REG_MEM64 |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_WRITE_DATA_MEMORY |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COPY_DATA_MEMORY |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_SH_REG |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_UCONFIG_REG |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_REGISTER_READBACK |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_PERFCOUNTER_READBACK |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_IMMEDIATE_WRITE));
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       LeavesUnvalidatedGfxFamiliesOnBaseAqlPath) {
  EXPECT_EQ(iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(8, 0, 0)),
            0u);
  EXPECT_EQ(iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(10, 3, 0)),
            IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB);
  EXPECT_EQ(iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(11, 0, 1)),
            IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB);
  EXPECT_EQ(iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(12, 0, 0)),
            IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB);
  EXPECT_EQ(iree_hal_amdgpu_select_vendor_packet_capabilities(GfxIp(13, 0, 0)),
            0u);
}

TEST_F(PhysicalDeviceCapabilitiesTest,
       KeepsUnvalidatedAqlprofilePhysicalFamiliesOnBaseAqlPm4IbPath) {
  struct ProcessorCase {
    const char* processor;
  };
  const ProcessorCase processors[] = {
      {"gfx900"},  {"gfx902"},  {"gfx904"},  {"gfx906"},  {"gfx908"},
      {"gfx909"},  {"gfx90c"},  {"gfx90a"},  {"gfx940"},  {"gfx941"},
      {"gfx942"},  {"gfx950"},  {"gfx1010"}, {"gfx1011"}, {"gfx1012"},
      {"gfx1013"}, {"gfx1030"}, {"gfx1031"}, {"gfx1032"}, {"gfx1033"},
      {"gfx1034"}, {"gfx1035"}, {"gfx1036"}, {"gfx1101"}, {"gfx1102"},
      {"gfx1103"}, {"gfx1150"}, {"gfx1151"}, {"gfx1152"}, {"gfx1153"},
      {"gfx1170"}, {"gfx1171"}, {"gfx1172"}, {"gfx1200"}, {"gfx1201"},
      {"gfx1250"}, {"gfx1251"},
  };
  const iree_hal_amdgpu_vendor_packet_capability_flags_t direct_pm4_families =
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_WAIT_REG_MEM64 |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_WRITE_DATA_MEMORY |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COPY_DATA_MEMORY |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_SH_REG |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_UCONFIG_REG |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_REGISTER_READBACK |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_PERFCOUNTER_READBACK |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_IMMEDIATE_WRITE;
  for (const ProcessorCase& processor : processors) {
    SCOPED_TRACE(processor.processor);
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities =
        iree_hal_amdgpu_select_vendor_packet_capabilities(
            GfxIpFromProcessor(processor.processor));
    EXPECT_TRUE(iree_any_bit_set(
        capabilities, IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB));
    EXPECT_FALSE(iree_any_bit_set(capabilities, direct_pm4_families));
  }
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsPm4TimestampStrategy) {
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(8, 0, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(9, 0, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(9, 5, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(10, 3, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(11, 0, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(11, 5, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(11, 7, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(12, 0, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(12, 0, 1)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(12, 5, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(12, 5, 1)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU);
  EXPECT_EQ(iree_hal_amdgpu_select_pm4_timestamp_strategy(GfxIp(13, 0, 0)),
            IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE);
}

TEST_F(PhysicalDeviceCapabilitiesTest, SelectsWaitBarrierStrategy) {
  EXPECT_EQ(iree_hal_amdgpu_select_wait_barrier_strategy(
                IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE |
                IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_WAIT_REG_MEM64),
            IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_AQL_BARRIER_VALUE);
  EXPECT_EQ(iree_hal_amdgpu_select_wait_barrier_strategy(
                IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_WAIT_REG_MEM64),
            IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_PM4_WAIT_REG_MEM64);
  EXPECT_EQ(iree_hal_amdgpu_select_wait_barrier_strategy(0),
            IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER);
}

}  // namespace
}  // namespace iree::hal::amdgpu
