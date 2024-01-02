// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <fstream>
#include <iostream>
#include <chrono>
#include <map>
#include <thread>

#include "iree/base/internal/flags.h"
#include "build_tools/third_party/tsl/xplane.pb.h"
#include "third_party/tracy/public/common/TracyProtocol.hpp"
#include "third_party/tracy/server/TracyFileRead.hpp"
#include "third_party/tracy/server/TracyFileWrite.hpp"
#include "third_party/tracy/server/TracyWorker.hpp"

IREE_FLAG(string, input_tracy_file, "",
          "Tracy file to read. Ignored if executable command is given.");
IREE_FLAG(string, output_tracy_file, "",
          "Tracy file to write as the output of the given executable command.");
IREE_FLAG(string, output_xplane_file, "",
          "Xplane file to write as the output of execution or conversion.");
IREE_FLAG(int32_t, tracy_port, 18086, "TCP port number for tracy profiler.");

namespace {

// Flag to inform main thread to disconnect from tracy profiler on SIGINT.
std::atomic<bool> g_disconnect = false;

void HandleSigInt(int sigid) {
  g_disconnect = true;
}

// Runs a subprogram with |argv| which is null-terminated at the end.
int RunExecutable(char** argv) {
  int pid = fork();
  if (pid == 0) {
    auto tracy_port_str = std::to_string(FLAG_tracy_port);
    setenv("TRACY_NO_EXIT", "1", /*overwrite=*/1);
    setenv("TRACY_PORT", tracy_port_str.c_str(), /*overwrite=*/1);
    execv(argv[0], argv);
    exit(errno);  // Not expected to be reached.
  }

  if (pid < 0) {
    return -1;
  }

  // Wait a little bit to check if a child process is still running.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  int wstatus;
  int ret = waitpid(pid, &wstatus, WNOHANG);
  if (ret == 0) {
    return 0;
  }

  if (ret > 0) {
    if (WIFSIGNALED(wstatus)) {
      errno = EINTR;
    } else {
      errno = WEXITSTATUS(wstatus);
    }
  }
  return -1;
}

template <typename T>
struct Zone {
  int16_t source_location_id;
  const T* zone;
};

template <typename T>
std::vector<Zone<T>> GetTopZones(
    const tracy::unordered_flat_map<int16_t, T>& zones,
    int num_zones_to_return) {
  std::map<int, Zone<T>> ordered_map;
  for (const auto& z : zones) {
    ordered_map[z.second.total] = Zone<T>{z.first, &z.second};
    if (ordered_map.size() > num_zones_to_return) {
      ordered_map.erase(ordered_map.begin());
    }
  }

  std::vector<Zone<T>> vector_to_return;
  for (auto it = ordered_map.rbegin(); it != ordered_map.rend(); ++it) {
    vector_to_return.push_back(it->second);
  }
  return vector_to_return;
};

const char* GetZoneName(const tracy::Worker& worker,
                        int16_t source_location_id) {
  const auto& srcloc = worker.GetSourceLocation(source_location_id);
  return worker.GetString(srcloc.name.active ? srcloc.name : srcloc.function);
}

const char* ArchToString(tracy::CpuArchitecture arch) {
  switch (arch) {
    case tracy::CpuArchUnknown: return "Unknown";
    case tracy::CpuArchX86: return "x86";
    case tracy::CpuArchX64: return "x86_64";
    case tracy::CpuArchArm32: return "arm";
    case tracy::CpuArchArm64: return "aarch64";
    default: return "Unknown";
  }
}

void PrintWorker(const tracy::Worker& worker) {
  std::cout << "[TRACY] CaptureName = " << worker.GetCaptureName() << "\n";
  std::cout << "[TRACY]     CpuArch = " << ArchToString(worker.GetCpuArch())
            << "\n";

  std::cout << "[TRACY]   CPU Zones = " << worker.GetZoneCount() << "\n";
  auto cpu_zones = GetTopZones(worker.GetSourceLocationZones(), 10);
  for (const auto& z : cpu_zones) {
    std::cout << "[TRACY]         " << GetZoneName(worker, z.source_location_id)
              << ": total_ns=" << z.zone->total
              << ", count=" << z.zone->zones.size() << "\n";
  }

  std::cout << "[TRACY]   GPU Zones = " << worker.GetGpuZoneCount() << "\n";
  auto gpu_zones = GetTopZones(worker.GetGpuSourceLocationZones(), 10);
  for (const auto& z : gpu_zones) {
    std::cout << "[TRACY]         " << GetZoneName(worker, z.source_location_id)
              << ": total_ns=" << z.zone->total
              << ", count=" << z.zone->zones.size() << "\n";
  }
}

tensorflow::profiler::XSpace ToXplane(const tracy::Worker& worker) {
  tensorflow::profiler::XSpace xspace;
  auto* xplane = xspace.add_planes();
  xplane->set_id(0);
  xplane->set_name(worker.GetCaptureName());

  // XLine corresponds to each Thread.
  // XEvent corresponds to each Zone.
  std::unordered_map<uint16_t, tensorflow::profiler::XLine*> xlines;

  for (const auto& z : worker.GetSourceLocationZones()) {
    auto& event_metadata = (*xplane->mutable_event_metadata())[z.first];
    event_metadata.set_id(z.first);
    event_metadata.set_name(GetZoneName(worker, z.first));
    event_metadata.set_display_name(event_metadata.name());

    for (const auto& t : z.second.zones) {
      auto it = xlines.find(t.Thread());
      if (it == xlines.end()) {
        it = xlines.insert(std::make_pair(t.Thread(), xplane->add_lines()))
             .first;
        auto* xline = it->second;
        xline->set_id(t.Thread());
        xline->set_display_id(t.Thread());
        xline->set_name(worker.GetThreadName(
            worker.DecompressThread(t.Thread())));
        xline->set_display_name(xline->name());
        //xline->set_timestamp_ns();
        //xline->set_duration_ps();
      }

      auto* event = it->second->add_events();
      event->set_metadata_id(z.first);
      event->set_offset_ps(t.Zone()->Start() * 1000);
      event->set_duration_ps((t.Zone()->End() - t.Zone()->Start()) * 1000);
    }
  }

  for (const auto& z : worker.GetGpuSourceLocationZones()) {
    auto& event_metadata = (*xplane->mutable_event_metadata())[z.first];
    event_metadata.set_id(z.first);
    event_metadata.set_name(GetZoneName(worker, z.first));
    event_metadata.set_display_name(event_metadata.name());

    for (const auto& t : z.second.zones) {
      auto it = xlines.find(t.Thread());
      if (it == xlines.end()) {
        it = xlines.insert(std::make_pair(t.Thread(), xplane->add_lines()))
             .first;
        auto* xline = it->second;
        xline->set_id(t.Thread());
        xline->set_display_id(t.Thread());
        xline->set_name(worker.GetThreadName(
            worker.DecompressThread(t.Thread())));
        xline->set_display_name(xline->name());
        //xline->set_timestamp_ns();
        //xline->set_duration_ps();
      }

      auto* event = it->second->add_events();
      event->set_metadata_id(z.first);
      event->set_offset_ps(t.Zone()->GpuStart() * 1000);
      event->set_duration_ps(
          (t.Zone()->GpuEnd() - t.Zone()->GpuStart()) * 1000);
    }
  }

  return xspace;
}

}  // namespace

int main(int argc, char** argv) {
  // IREE tries to parse all flags. Reduce argc for parsing to exclude
  // subprogram's arguments.
  int argc_for_parse = 1;
  while (argc_for_parse < argc && argv[argc_for_parse][0] == '-') {
    ++argc_for_parse;
  }
  int argc_for_subprogram = argc - argc_for_parse;
  char** argv_for_subprogram = argv + argc_for_parse;

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc_for_parse,
                           &argv);

  std::unique_ptr<tracy::Worker> worker;
  // Build a worker from a tracy file,
  if (argc_for_subprogram == 0) {
    if (strlen(FLAG_input_tracy_file) == 0) {
      std::cerr << "[ERROR] Either executable command or a tracy file to read "
                << "is not provided.\n";
      return EXIT_FAILURE;
    }

    auto f = std::unique_ptr<tracy::FileRead>(
        tracy::FileRead::Open(FLAG_input_tracy_file));
    if (!f) {
      std::cerr << "[ERROR] Could not open file: " << FLAG_input_tracy_file
                << "\n";
      return EXIT_FAILURE;
    }

    worker = std::make_unique<tracy::Worker>(*f);
    while (!worker->AreSourceLocationZonesReady()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

  // Or, by connecting to tracy profiler of subprogram.
  } else {
    if (RunExecutable(argv_for_subprogram) < 0) {
      std::cerr << "[ERROR] Could not execute " << argv_for_subprogram[0]
                << ": " << strerror(errno) << "\n";
      return EXIT_FAILURE;
    }

    worker = std::make_unique<tracy::Worker>("127.0.0.1", FLAG_tracy_port);
    while (!worker->HasData()) {
      auto status = worker->GetHandshakeStatus();
      if (status != tracy::HandshakePending &&
          status != tracy::HandshakeWelcome) {
        std::cerr << "[ERROR] Could not connect to " << argv_for_subprogram[0]
                  << "\n";
        return EXIT_FAILURE;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "[TRACY] Connected to " << argv_for_subprogram[0]
              << " through 127.0.0.1:" << FLAG_tracy_port << "\n";

    signal(SIGINT, HandleSigInt);

    while(worker->IsConnected()) {
      if (g_disconnect) {
        worker->Disconnect();
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  }

  PrintWorker(*worker);

  if (strlen(FLAG_output_tracy_file) > 0) {
    auto f = std::unique_ptr<tracy::FileWrite>(
        tracy::FileWrite::Open(FLAG_output_tracy_file));
    if (f) {
      worker->Write(*f, /*fiDict=*/false);
    } else {
      std::cerr << "[ERROR] Could not write tracy file "
                << FLAG_output_tracy_file << "\n";
    }
  }

  if (strlen(FLAG_output_xplane_file) > 0) {
    std::fstream f(FLAG_output_xplane_file, std::ios::out|std::ios::binary);
    if (!ToXplane(*worker).SerializeToOstream(&f)) {
      std::cerr << "[ERROR] Could not write xplane file "
                << FLAG_output_xplane_file << "\n";
    }
  }

  return EXIT_SUCCESS;
}
