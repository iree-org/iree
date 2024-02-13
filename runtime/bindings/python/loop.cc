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

static const char kHalDeviceLoopDocstring[] =
    R"(Loop for handling events/timepoints associated with a single device.

In the single device case, this is the best option for running a host event
loop for dispatching callbacks upon satisfaction of semaphores and fences as
it just calls down into the driver and does not involve complicated
choreography to join across devices, etc.

This is not a general purpose asyncio.loop but it can interop with asyncio
by satisfying futures derived from a single device. It should generally be
launched on a background thread in such cases.
)";

class HalDeviceLoop {
 public:
  HalDeviceLoop(HalDevice device, size_t initial_capacity)
      : device_(std::move(device)) {
    iree_slim_mutex_initialize(&mu_);
    CheckApiStatus(
        iree_hal_semaphore_create(device_.raw_ptr(), 0, &control_sem_),
        "create semaphore");

    next_pending_futures_.reserve(initial_capacity);
  }
  ~HalDeviceLoop() {
    // Cancel all futures.
    iree_slim_mutex_lock(&mu_);
    for (auto &entry : next_pending_futures_) {
      py::handle future = std::get<2>(entry);
      CancelFuture(future);
    }
    next_pending_futures_.clear();
    iree_slim_mutex_unlock(&mu_);

    iree_slim_mutex_deinitialize(&mu_);
    iree_hal_semaphore_release(control_sem_);
  }

  void Run() {
    py::gil_scoped_release gil_release;
    // Wait list.
    size_t wait_capacity = 0;
    iree_hal_semaphore_t **wait_semaphores = nullptr;
    uint64_t *wait_payloads = nullptr;
    // Pending futures that are actively being waited on. Owned by Run().
    std::vector<std::tuple<iree_hal_semaphore_t *, uint64_t, py::handle>>
        pending_futures;
    // Scratch pad of pending futures that we must keep waiting on. Owned by
    // Run().
    std::vector<std::tuple<iree_hal_semaphore_t *, uint64_t, py::handle>>
        scratch_pending_futures;
    pending_futures.reserve(next_pending_futures_.capacity());

    bool keep_running = true;
    uint64_t next_control_wakeup = 1;
    while (true) {
      // Transfer any pending futures into the current list.
      iree_slim_mutex_lock(&mu_);
      while (!next_pending_futures_.empty()) {
        pending_futures.push_back(std::move(next_pending_futures_.back()));
        next_pending_futures_.pop_back();
      }
      keep_running = !shutdown_signalled_;
      iree_slim_mutex_unlock(&mu_);
      if (!keep_running) {
        break;
      }

      // Ensure we are sized for all pending futures plus one for the control
      // semaphore (we just double to keep from needing to re-allocate often).
      if (wait_capacity < pending_futures.size() + 1) {
        // Start with a capacity of 5 to avoid multiple allocations for 0, 1, 2.
        wait_capacity =
            std::max(pending_futures.size() * 2 + 1, static_cast<size_t>(5));
        wait_semaphores = static_cast<iree_hal_semaphore_t **>(realloc(
            wait_semaphores, sizeof(iree_hal_semaphore_t *) * wait_capacity));
        wait_payloads = static_cast<uint64_t *>(
            realloc(wait_payloads, sizeof(uint64_t *) * wait_capacity));
      }

      // Poll all futures and dispatch. Any that are still pending are routed
      // to the scratch_pending_futures. Important: we don't hold the gil so
      // can not do anything that toggles reference counts or calls Python yet.
      size_t wait_size = 0;
      iree_status_t status;
      for (size_t i = 0; i < pending_futures.size(); ++i) {
        auto &entry = pending_futures[i];
        uint64_t current_payload;
        status = iree_hal_semaphore_query(std::get<0>(entry), &current_payload);
        if (iree_status_is_ok(status)) {
          if (current_payload >= std::get<1>(entry)) {
            // All done.
            SignalFuture(std::get<2>(entry));
          } else {
            // Keep it pending.
            wait_semaphores[wait_size] = std::get<0>(entry);
            wait_payloads[wait_size] = std::get<1>(entry);
            wait_size += 1;
            scratch_pending_futures.push_back(std::move(entry));
          }
        } else {
          SignalFutureFailure(std::get<2>(entry), status);
        }
      }
      pending_futures.clear();
      pending_futures.swap(scratch_pending_futures);

      // Add the control semaphore.
      wait_semaphores[wait_size] = control_sem_;
      wait_payloads[wait_size] = next_control_wakeup;
      wait_size += 1;

      // Wait.
      status = iree_hal_device_wait_semaphores(
          device_.raw_ptr(), IREE_HAL_WAIT_MODE_ANY,
          {wait_size, wait_semaphores, wait_payloads}, iree_infinite_timeout());
      if (!iree_status_is_ok(status)) {
        py::gil_scoped_acquire acquire_gil;
        CheckApiStatus(status,
                       "iree_hal_device_wait_semaphores from HalDeviceLoop");
      }

      status = iree_hal_semaphore_query(control_sem_, &next_control_wakeup);
      if (!iree_status_is_ok(status)) {
        py::gil_scoped_acquire acquire_gil;
        CheckApiStatus(status,
                       "iree_hal_device_wait_semaphores from HalDeviceLoop");
      }
      next_control_wakeup += 1;
    }

    free(wait_semaphores);
    free(wait_payloads);

    // Cancel all pending futures.
    {
      for (auto &entry : pending_futures) {
        py::handle future = std::get<2>(entry);
        CancelFuture(future);
      }
    }
  }

  void CancelFuture(py::handle future) {
    py::gil_scoped_acquire acquire_gil;
    try {
      future.attr("cancel")();
    } catch (py::python_error &e) {
      ReportUncaughtException(e);
    }
    future.dec_ref();
  }

  void SignalFuture(py::handle future) {
    py::gil_scoped_acquire acquire_gil;
    try {
      future.attr("set_result")(true);
    } catch (py::python_error &e) {
      ReportUncaughtException(e);
    }
    future.dec_ref();
  }

  void SignalFutureFailure(py::handle future, iree_status_t status) {
    py::gil_scoped_acquire acquire_gil;
    std::string message = ApiStatusToString(status);
    iree_status_ignore(status);
    PyErr_SetString(PyExc_RuntimeError, message.c_str());
    PyObject *exc_type;
    PyObject *exc_value;
    PyObject *exc_tb;
    PyErr_Fetch(&exc_type, &exc_value, &exc_tb);
    future.attr("set_exception")(exc_value);
    Py_XDECREF(exc_type);
    Py_XDECREF(exc_tb);
    Py_XDECREF(exc_value);
    future.dec_ref();
  }

  void SignalShutdown() {
    iree_slim_mutex_lock(&mu_);
    shutdown_signalled_ = true;
    auto status = iree_hal_semaphore_signal(control_sem_, control_next_++);
    iree_slim_mutex_unlock(&mu_);
    CheckApiStatus(status, "iree_hal_semaphore_signal");
  }

  void OnSemaphore(HalSemaphore semaphore, uint64_t payload,
                   py::handle future) {
    iree_slim_mutex_lock(&mu_);
    next_pending_futures_.push_back(
        std::make_tuple(semaphore.steal_raw_ptr(), payload, std::move(future)));
    future.inc_ref();
    auto status = iree_hal_semaphore_signal(control_sem_, control_next_++);
    iree_slim_mutex_unlock(&mu_);
    CheckApiStatus(status, "iree_hal_semaphore_signal");
  }

 private:
  // Certain calls into Futures may raise exceptions because of illegal states.
  // There is really not much we can do about this, so we attempt to report.
  // TODO: Have some kind of fatal exception hook.
  void ReportUncaughtException(py::python_error &e) {
    e.discard_as_unraisable(__func__);
  }

  iree_slim_mutex_t mu_;
  HalDevice device_;
  iree_hal_semaphore_t *control_sem_ = nullptr;
  uint64_t control_next_ = 1;
  bool shutdown_signalled_ = false;

  // Incoming futures to add to the pending list on next cycle. Must be locked
  // with mu_.
  // Note that because these structures are processed without the GIL being
  // held, we cannot unexpectedly do any reference count manipulation.
  // Therefore, when added here, it is added with a reference. And the reference
  // must be returned when retired.
  std::vector<std::tuple<iree_hal_semaphore_t *, uint64_t, py::handle>>
      next_pending_futures_;
};

}  // namespace

void SetupLoopBindings(py::module_ &m) {
  py::class_<HalDeviceLoop>(m, "HalDeviceLoop")
      .def(py::init<HalDevice, size_t>(), py::arg("device"),
           py::arg("initial_capacity") = 10)
      .def("on_semaphore", &HalDeviceLoop::OnSemaphore, py::arg("semaphore"),
           py::arg("payload"), py::arg("future"))
      .def("run", &HalDeviceLoop::Run)
      .def("signal_shutdown", &HalDeviceLoop::SignalShutdown)
      .doc() = kHalDeviceLoopDocstring;
}

}  // namespace iree::python
