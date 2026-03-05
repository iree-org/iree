// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/transport_registry.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Mock factory for testing.
typedef struct mock_factory_t {
  iree_net_transport_factory_t base;
  iree_allocator_t allocator;
  bool freed;
} mock_factory_t;

static void mock_factory_destroy(iree_net_transport_factory_t* factory) {
  mock_factory_t* mock = (mock_factory_t*)factory;
  mock->freed = true;
  iree_allocator_t allocator = mock->allocator;
  iree_allocator_free(allocator, mock);
}

static iree_net_transport_capabilities_t mock_factory_query_capabilities(
    iree_net_transport_factory_t* factory) {
  return IREE_NET_TRANSPORT_CAPABILITY_RELIABLE |
         IREE_NET_TRANSPORT_CAPABILITY_ORDERED;
}

static iree_status_t mock_factory_connect(
    iree_net_transport_factory_t* factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_transport_connect_callback_t callback, void* user_data) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
}

static iree_status_t mock_factory_create_listener(
    iree_net_transport_factory_t* factory, iree_string_view_t bind_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_listener_accept_callback_t accept_callback, void* user_data,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
}

static const iree_net_transport_factory_vtable_t mock_factory_vtable = {
    mock_factory_destroy,
    mock_factory_query_capabilities,
    mock_factory_connect,
    mock_factory_create_listener,
};

static iree_status_t mock_factory_create(iree_allocator_t allocator,
                                         mock_factory_t** out_factory) {
  mock_factory_t* factory = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*factory), (void**)&factory));
  iree_atomic_ref_count_init(&factory->base.ref_count);
  factory->base.vtable = &mock_factory_vtable;
  factory->allocator = allocator;
  factory->freed = false;
  *out_factory = factory;
  return iree_ok_status();
}

class TransportRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_net_transport_registry_allocate(iree_allocator_system(),
                                                        &registry_));
  }

  void TearDown() override {
    if (registry_) {
      iree_net_transport_registry_free(registry_);
    }
  }

  iree_net_transport_registry_t* registry_ = nullptr;
};

TEST_F(TransportRegistryTest, AllocateFree) {
  // Just test that SetUp/TearDown work.
  EXPECT_NE(registry_, nullptr);
  EXPECT_EQ(iree_net_transport_registry_count(registry_), 0);
}

TEST_F(TransportRegistryTest, RegisterLookup) {
  mock_factory_t* factory = nullptr;
  IREE_ASSERT_OK(mock_factory_create(iree_allocator_system(), &factory));

  IREE_ASSERT_OK(iree_net_transport_registry_register(registry_, IREE_SV("tcp"),
                                                      &factory->base));
  iree_net_transport_factory_release(&factory->base);
  EXPECT_EQ(iree_net_transport_registry_count(registry_), 1);

  iree_net_transport_factory_t* found =
      iree_net_transport_registry_lookup(registry_, IREE_SV("tcp"));
  EXPECT_EQ(found, &factory->base);

  // Not found.
  EXPECT_EQ(iree_net_transport_registry_lookup(registry_, IREE_SV("udp")),
            nullptr);
}

TEST_F(TransportRegistryTest, RegisterDuplicate) {
  mock_factory_t* factory1 = nullptr;
  mock_factory_t* factory2 = nullptr;
  IREE_ASSERT_OK(mock_factory_create(iree_allocator_system(), &factory1));
  IREE_ASSERT_OK(mock_factory_create(iree_allocator_system(), &factory2));

  IREE_ASSERT_OK(iree_net_transport_registry_register(registry_, IREE_SV("tcp"),
                                                      &factory1->base));
  iree_net_transport_factory_release(&factory1->base);

  // Duplicate should fail.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        iree_net_transport_registry_register(
                            registry_, IREE_SV("tcp"), &factory2->base));

  // factory2 was not registered — release our reference to destroy it.
  iree_net_transport_factory_release(&factory2->base);
}

TEST_F(TransportRegistryTest, RegisterEmptyScheme) {
  mock_factory_t* factory = nullptr;
  IREE_ASSERT_OK(mock_factory_create(iree_allocator_system(), &factory));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_transport_registry_register(registry_, iree_string_view_empty(),
                                           &factory->base));

  // factory was not registered — release our reference to destroy it.
  iree_net_transport_factory_release(&factory->base);
}

TEST_F(TransportRegistryTest, RegisterMultiple) {
  mock_factory_t* tcp_factory = nullptr;
  mock_factory_t* quic_factory = nullptr;
  mock_factory_t* rdma_factory = nullptr;
  IREE_ASSERT_OK(mock_factory_create(iree_allocator_system(), &tcp_factory));
  IREE_ASSERT_OK(mock_factory_create(iree_allocator_system(), &quic_factory));
  IREE_ASSERT_OK(mock_factory_create(iree_allocator_system(), &rdma_factory));

  IREE_ASSERT_OK(iree_net_transport_registry_register(registry_, IREE_SV("tcp"),
                                                      &tcp_factory->base));
  iree_net_transport_factory_release(&tcp_factory->base);
  IREE_ASSERT_OK(iree_net_transport_registry_register(
      registry_, IREE_SV("quic"), &quic_factory->base));
  iree_net_transport_factory_release(&quic_factory->base);
  IREE_ASSERT_OK(iree_net_transport_registry_register(
      registry_, IREE_SV("rdma"), &rdma_factory->base));
  iree_net_transport_factory_release(&rdma_factory->base);

  EXPECT_EQ(iree_net_transport_registry_count(registry_), 3);

  // Lookup each.
  EXPECT_EQ(iree_net_transport_registry_lookup(registry_, IREE_SV("tcp")),
            &tcp_factory->base);
  EXPECT_EQ(iree_net_transport_registry_lookup(registry_, IREE_SV("quic")),
            &quic_factory->base);
  EXPECT_EQ(iree_net_transport_registry_lookup(registry_, IREE_SV("rdma")),
            &rdma_factory->base);
}

struct EnumerateContext {
  std::vector<std::string> schemes;
};

static iree_status_t enumerate_callback(void* user_data,
                                        iree_string_view_t scheme,
                                        iree_net_transport_factory_t* factory) {
  auto* ctx = static_cast<EnumerateContext*>(user_data);
  ctx->schemes.push_back(std::string(scheme.data, scheme.size));
  return iree_ok_status();
}

TEST_F(TransportRegistryTest, Enumerate) {
  mock_factory_t* tcp_factory = nullptr;
  mock_factory_t* quic_factory = nullptr;
  IREE_ASSERT_OK(mock_factory_create(iree_allocator_system(), &tcp_factory));
  IREE_ASSERT_OK(mock_factory_create(iree_allocator_system(), &quic_factory));

  IREE_ASSERT_OK(iree_net_transport_registry_register(registry_, IREE_SV("tcp"),
                                                      &tcp_factory->base));
  iree_net_transport_factory_release(&tcp_factory->base);
  IREE_ASSERT_OK(iree_net_transport_registry_register(
      registry_, IREE_SV("quic"), &quic_factory->base));
  iree_net_transport_factory_release(&quic_factory->base);

  EnumerateContext ctx;
  IREE_ASSERT_OK(iree_net_transport_registry_enumerate(
      registry_, enumerate_callback, &ctx));

  EXPECT_EQ(ctx.schemes.size(), 2);
  // Order is registration order.
  EXPECT_EQ(ctx.schemes[0], "tcp");
  EXPECT_EQ(ctx.schemes[1], "quic");
}

static iree_status_t enumerate_stop_callback(
    void* user_data, iree_string_view_t scheme,
    iree_net_transport_factory_t* factory) {
  auto* count = static_cast<int*>(user_data);
  (*count)++;
  if (*count >= 1) {
    return iree_make_status(IREE_STATUS_CANCELLED, "stop enumeration");
  }
  return iree_ok_status();
}

TEST_F(TransportRegistryTest, EnumerateStopEarly) {
  mock_factory_t* tcp_factory = nullptr;
  mock_factory_t* quic_factory = nullptr;
  IREE_ASSERT_OK(mock_factory_create(iree_allocator_system(), &tcp_factory));
  IREE_ASSERT_OK(mock_factory_create(iree_allocator_system(), &quic_factory));

  IREE_ASSERT_OK(iree_net_transport_registry_register(registry_, IREE_SV("tcp"),
                                                      &tcp_factory->base));
  iree_net_transport_factory_release(&tcp_factory->base);
  IREE_ASSERT_OK(iree_net_transport_registry_register(
      registry_, IREE_SV("quic"), &quic_factory->base));
  iree_net_transport_factory_release(&quic_factory->base);

  int count = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_net_transport_registry_enumerate(
                            registry_, enumerate_stop_callback, &count));
  EXPECT_EQ(count, 1);
}

TEST_F(TransportRegistryTest, FreeReleasesFactories) {
  // Track whether factory is destroyed when registry releases its reference.
  static bool factory_destroyed = false;

  // Custom vtable that sets flag on destroy.
  static const iree_net_transport_factory_vtable_t tracking_vtable = {
      [](iree_net_transport_factory_t* factory) {
        factory_destroyed = true;
        iree_allocator_free(iree_allocator_system(), factory);
      },
      mock_factory_query_capabilities,
      mock_factory_connect,
      mock_factory_create_listener,
  };

  mock_factory_t* factory = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                       sizeof(*factory), (void**)&factory));
  iree_atomic_ref_count_init(&factory->base.ref_count);
  factory->base.vtable = &tracking_vtable;
  factory->allocator = iree_allocator_system();

  IREE_ASSERT_OK(iree_net_transport_registry_register(
      registry_, IREE_SV("test"), &factory->base));
  // Release caller's reference — registry holds the only remaining one.
  iree_net_transport_factory_release(&factory->base);

  factory_destroyed = false;
  iree_net_transport_registry_free(registry_);
  registry_ = nullptr;  // Prevent double-free in TearDown.

  EXPECT_TRUE(factory_destroyed);
}

TEST_F(TransportRegistryTest, QueryCapabilities) {
  mock_factory_t* factory = nullptr;
  IREE_ASSERT_OK(mock_factory_create(iree_allocator_system(), &factory));

  IREE_ASSERT_OK(iree_net_transport_registry_register(registry_, IREE_SV("tcp"),
                                                      &factory->base));
  iree_net_transport_factory_release(&factory->base);

  iree_net_transport_factory_t* found =
      iree_net_transport_registry_lookup(registry_, IREE_SV("tcp"));
  ASSERT_NE(found, nullptr);

  iree_net_transport_capabilities_t caps =
      iree_net_transport_factory_query_capabilities(found);
  EXPECT_TRUE(caps & IREE_NET_TRANSPORT_CAPABILITY_RELIABLE);
  EXPECT_TRUE(caps & IREE_NET_TRANSPORT_CAPABILITY_ORDERED);
  EXPECT_FALSE(caps & IREE_NET_TRANSPORT_CAPABILITY_RDMA);
}

}  // namespace
