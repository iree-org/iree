// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./loop.h"

#include <cstdio>
#include <vector>

#include "./hal.h"
#include "iree/base/internal/synchronization.h"

namespace iree::python {

namespace {

static const char kHalDeviceLoopBridgeDocstring[] =
    R"(Bridges device semaphore signalling to asyncio futures.

This is intended to be run alongside an asyncio loop, allowing arbitrary
semaphore timepoints to be bridged to the loop, satisfying futures.

Internally, it starts a thread which spins to poll the requested semaphores
(which all must be from the same device). It can be used in single-device
cases as a simpler implementation than a full integration with an asyncio
event loop, theoretically resulting in fewer heavy-weight, kernel/device
synchronization interactions.
)";

class HalDeviceLoopBridge {
 public:
  HalDeviceLoopBridge(HalDevice device, py::object loop)
      : device_(std::move(device)), loop_(std::move(loop)) {
    IREE_PY_TRACEF("new HalDeviceLoopBridge (%p)", this);
    iree_slim_mutex_initialize(&mu_);
    CheckApiStatus(
        iree_hal_semaphore_create(device_.raw_ptr(), 0, &control_sem_),
        "create semaphore");

    loop_call_soon_ = loop_.attr("call_soon_threadsafe");

    // Start the thread.
    auto threading_m = py::module_::import_("threading");
    thread_ = threading_m.attr("Thread")(
        /*group=*/py::none(),
        /*target=*/py::cpp_function([this]() { Run(); }),
        /*name=*/"HalDeviceLoopBridge");
    thread_.attr("start")();
  }
  ~HalDeviceLoopBridge() {
    IREE_PY_TRACEF("~HalDeviceLoopBridge(%p)", this);

    // Stopping the thread during destruction is not great. But it is better
    // than invalidating live memory.
    if (!thread_.is_none()) {
      auto warnings_m = py::module_::import_("warnings");
      warnings_m.attr("warn")(
          "HalDeviceLoopBridge deleted while running. Recommend explicitly "
          "calling stop() to avoid hanging the gc");
      Stop();
    }

    // Cancel all futures.
    iree_slim_mutex_lock(&mu_);
    for (auto &entry : next_pending_futures_) {
      iree_hal_semaphore_release(std::get<0>(entry));
      py::handle future = std::get<2>(entry);
      py::handle value = std::get<3>(entry);
      CancelFuture(future, value);
    }
    next_pending_futures_.clear();
    iree_slim_mutex_unlock(&mu_);

    iree_slim_mutex_deinitialize(&mu_);
    iree_hal_semaphore_release(control_sem_);
  }

  void Stop() {
    if (thread_.is_none()) {
      IREE_PY_TRACEF("HalDeviceLoopBridge::Stop(%p): Already stopped", this);
      return;
    }
    IREE_PY_TRACEF("HalDeviceLoopBridge::Stop(%p)", this);
    iree_slim_mutex_lock(&mu_);
    shutdown_signaled_ = true;
    auto status = iree_hal_semaphore_signal(control_sem_, control_next_++);
    iree_slim_mutex_unlock(&mu_);
    CheckApiStatus(status, "iree_hal_semaphore_signal");
    thread_.attr("join")();
    thread_ = py::none();
    IREE_PY_TRACEF("HalDeviceLoopBridge::Stop(%p): Joined", this);
  }

  void Run() {
    IREE_PY_TRACEF("HalDeviceLoopBridge::Run(%p)", this);
    py::gil_scoped_release gil_release;
    // Wait list.
    std::vector<iree_hal_semaphore_t *> wait_semaphores;
    std::vector<uint64_t> wait_payloads;
    wait_semaphores.reserve(5);
    wait_payloads.reserve(5);
    // Pending futures that are actively being waited on. Owned by Run().
    std::vector<
        std::tuple<iree_hal_semaphore_t *, uint64_t, py::handle, py::handle>>
        pending_futures;
    // Scratch pad of pending futures that we must keep waiting on. Owned by
    // Run().
    std::vector<
        std::tuple<iree_hal_semaphore_t *, uint64_t, py::handle, py::handle>>
        scratch_pending_futures;
    pending_futures.reserve(next_pending_futures_.capacity());

    bool keep_running = true;
    uint64_t next_control_wakeup = 1;
    while (true) {
      IREE_PY_TRACEF("HalDeviceLoopBridge::Run(%p): Loop begin", this);
      // Transfer any pending futures into the current list.
      iree_slim_mutex_lock(&mu_);
      while (!next_pending_futures_.empty()) {
        pending_futures.push_back(std::move(next_pending_futures_.back()));
        next_pending_futures_.pop_back();
      }
      keep_running = !shutdown_signaled_;
      iree_slim_mutex_unlock(&mu_);
      if (!keep_running) {
        IREE_PY_TRACEF("HalDeviceLoopBridge::Run(%p): Loop break", this);
        break;
      }
      wait_semaphores.clear();
      wait_payloads.clear();

      // Poll all futures and dispatch. Any that are still pending are routed
      // to the scratch_pending_futures. Important: we don't hold the gil so
      // can not do anything that toggles reference counts or calls Python yet.
      iree_status_t status;
      for (size_t i = 0; i < pending_futures.size(); ++i) {
        auto &entry = pending_futures[i];
        uint64_t current_payload;
        iree_hal_semaphore_t *semaphore = std::get<0>(entry);
        status = iree_hal_semaphore_query(semaphore, &current_payload);
        if (iree_status_is_ok(status)) {
          if (current_payload >= std::get<1>(entry)) {
            // All done.
            iree_hal_semaphore_release(semaphore);
            SignalFuture(std::get<2>(entry), std::get<3>(entry));
          } else {
            // Keep it pending.
            IREE_PY_TRACEF("  Add to wait list: semaphore=%p, payload=%" PRIu64,
                           semaphore, std::get<1>(entry));
            wait_semaphores.push_back(semaphore);
            wait_payloads.push_back(std::get<1>(entry));
            scratch_pending_futures.push_back(std::move(entry));
          }
        } else {
          iree_hal_semaphore_release(semaphore);
          SignalFutureFailure(std::get<2>(entry), std::get<3>(entry), status);
        }
      }
      pending_futures.clear();
      pending_futures.swap(scratch_pending_futures);

      // Add the control semaphore.
      wait_semaphores.push_back(control_sem_);
      wait_payloads.push_back(next_control_wakeup);

      // Wait.
      IREE_PY_TRACEF("HalDeviceLoopBridge::Run(%p): wait_semaphores(%zu)", this,
                     wait_semaphores.size());
      status = iree_hal_device_wait_semaphores(
          device_.raw_ptr(), IREE_HAL_WAIT_MODE_ANY,
          {wait_semaphores.size(), wait_semaphores.data(),
           wait_payloads.data()},
          iree_infinite_timeout());
      if (!iree_status_is_ok(status)) {
        py::gil_scoped_acquire acquire_gil;
        CheckApiStatus(
            status, "iree_hal_device_wait_semaphores from HalDeviceLoopBridge");
      }

      status = iree_hal_semaphore_query(control_sem_, &next_control_wakeup);
      if (!iree_status_is_ok(status)) {
        py::gil_scoped_acquire acquire_gil;
        CheckApiStatus(
            status, "iree_hal_device_wait_semaphores from HalDeviceLoopBridge");
      }
      next_control_wakeup += 1;
      IREE_PY_TRACEF("HalDeviceLoopBridge::Run(%p): Loop end", this);
    }

    // Cancel all pending futures.
    {
      for (auto &entry : pending_futures) {
        iree_hal_semaphore_release(std::get<0>(entry));
        py::handle future = std::get<2>(entry);
        py::handle value = std::get<3>(entry);
        CancelFuture(future, value);
      }
    }

    IREE_PY_TRACEF("HalDeviceLoopBridge::Run(%p): Thread complete", this);
  }

  void CancelFuture(py::handle future, py::handle value) {
    IREE_PY_TRACEF("HalDeviceLoopBridge::CancelFuture(%p)", future.ptr());
    py::gil_scoped_acquire acquire_gil;
    try {
      future.attr("cancel")();
    } catch (py::python_error &e) {
      ReportUncaughtException(e);
    }
    future.dec_ref();
    value.dec_ref();
  }

  void SignalFuture(py::handle future, py::handle value) {
    IREE_PY_TRACEF("HalDeviceLoopBridge::SignalFuture(%p)", future.ptr());
    py::gil_scoped_acquire acquire_gil;
    py::object future_owned = py::steal(future);
    py::object value_owned = py::steal(value);
    loop_call_soon_(py::cpp_function([future_owned = std::move(future_owned),
                                      value_owned = std::move(value_owned)]() {
      future_owned.attr("set_result")(value_owned);
    }));
  }

  void SignalFutureFailure(py::handle future, py::handle value,
                           iree_status_t status) {
    py::gil_scoped_acquire acquire_gil;
    py::object future_owned = py::steal(future);
    py::object value_owned = py::steal(value);
    std::string message = ApiStatusToString(status);
    IREE_PY_TRACEF("HalDeviceLoopBridge::SignalFutureFailure(future=%p) : %s",
                   future.ptr(), message.c_str());
    iree_status_ignore(status);
    loop_call_soon_(py::cpp_function([future_owned = std::move(future_owned),
                                      value_owned = std::move(value_owned),
                                      message = std::move(message)]() {
      PyErr_SetString(PyExc_RuntimeError, message.c_str());
      PyObject *exc_type;
      PyObject *exc_value;
      PyObject *exc_tb;
      PyErr_Fetch(&exc_type, &exc_value, &exc_tb);
      future_owned.attr("set_exception")(exc_value);
      Py_XDECREF(exc_type);
      Py_XDECREF(exc_tb);
      Py_XDECREF(exc_value);
    }));
  }

  py::object OnSemaphore(HalSemaphore semaphore, uint64_t payload,
                         py::object value) {
    IREE_PY_TRACEF(
        "HalDeviceLoopBridge::OnSemaphore(semaphore=%p, payload=%" PRIu64 ")",
        semaphore.raw_ptr(), payload);
    py::object future = loop_.attr("create_future")();
    iree_slim_mutex_lock(&mu_);
    next_pending_futures_.push_back(std::make_tuple(
        semaphore.steal_raw_ptr(), payload, future, value.release()));
    future.inc_ref();
    auto status = iree_hal_semaphore_signal(control_sem_, control_next_++);
    iree_slim_mutex_unlock(&mu_);
    CheckApiStatus(status, "iree_hal_semaphore_signal");
    return future;
  }

 private:
  // Certain calls into Futures may raise exceptions because of illegal states.
  // There is really not much we can do about this, so we attempt to report.
  // TODO: Have some kind of fatal exception hook.
  void ReportUncaughtException(py::python_error &e) {
    e.discard_as_unraisable(py::str(__func__));
  }

  iree_slim_mutex_t mu_;
  HalDevice device_;
  py::object loop_;
  py::object thread_;
  py::object loop_call_soon_;
  iree_hal_semaphore_t *control_sem_ = nullptr;
  uint64_t control_next_ = 1;
  bool shutdown_signaled_ = false;

  // Incoming futures to add to the pending list on next cycle. Must be locked
  // with mu_.
  // Note that because these structures are processed without the GIL being
  // held, we cannot unexpectedly do any reference count manipulation.
  // Therefore, when added here, it is added with a reference. And the reference
  // must be returned when retired.
  // Fields: Semaphore, wait_payload_value, future, future_value
  std::vector<
      std::tuple<iree_hal_semaphore_t *, uint64_t, py::handle, py::handle>>
      next_pending_futures_;
};

}  // namespace

void SetupLoopBindings(py::module_ &m) {
  py::class_<HalDeviceLoopBridge>(m, "HalDeviceLoopBridge")
      .def(py::init<HalDevice, py::object>(), py::arg("device"),
           py::arg("loop"))
      .def("stop", &HalDeviceLoopBridge::Stop)
      .def("on_semaphore", &HalDeviceLoopBridge::OnSemaphore,
           py::arg("semaphore"), py::arg("payload"), py::arg("value"))
      .doc() = kHalDeviceLoopBridgeDocstring;
}

}  // namespace iree::python
