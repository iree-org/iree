// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hal_executable_library_call_hooks/perf_event_linux.h"

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

PerfEventFd::PerfEventFd(PerfEventType perf_event_type) {
  perf_event_attr pe;
  memset(&pe, 0, sizeof pe);
  pe.size = sizeof(pe);
  pe.type = perf_event_type.type;
  pe.config = perf_event_type.config;
  pe.exclude_kernel = 1;
  pe.disabled = 1;
  fd_ = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
  if (fd_ < 0) {
    fprintf(stderr,
            "perf_event_open failed for event %s. Either an unknown perf event "
            "config, or a permissions issue? need to lower "
            "`perf_event_paranoid` ?\n",
            perf_event_type.name);
    exit(1);
  }
}

PerfEventFd::~PerfEventFd() { close(fd_); }

void PerfEventFd::reset() { ioctl(fd_, PERF_EVENT_IOC_RESET, 0); }

void PerfEventFd::enable() { ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0); }

void PerfEventFd::disable() { ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0); }

int64_t PerfEventFd::read() const {
  int64_t count = 0;
  if (::read(fd_, &count, sizeof count) != sizeof count) {
    return 0;
  }
  return count;
}

// Returns a list of all known perf event types.
static const std::vector<PerfEventType> &listAllPerfEventTypes();

static PerfEventType parsePerfEventType(int type_str_length,
                                        const char *type_str) {
  for (PerfEventType event_type : listAllPerfEventTypes()) {
    if (strncmp(event_type.name, type_str, type_str_length)) {
      continue;
    }
    return event_type;
  }
  fprintf(stderr, "Unhandled perf event type: %s\n", type_str);
  exit(1);
  return {};
}

std::vector<PerfEventType> parsePerfEventTypes(const char *types_str) {
  std::vector<PerfEventType> out_event_types;
  while (*types_str) {
    const char *segment_ptr = types_str;
    int segment_length = 0;
    const char *comma_ptr = strchr(types_str, ',');
    if (comma_ptr) {
      segment_length = comma_ptr - types_str;
      types_str = comma_ptr + 1;
    } else {
      segment_length = strlen(types_str);
      types_str += segment_length;
    }
    out_event_types.push_back(parsePerfEventType(segment_length, segment_ptr));
  }
  return out_event_types;
}

void printAllEventTypesAndDescriptions(FILE *file) {
  for (PerfEventType event_type : listAllPerfEventTypes()) {
    fprintf(file, "%-40s [%s] %s\n", event_type.name,
            strlen(event_type.target) ? event_type.target : "generic",
            event_type.description);
  }
}

static const std::vector<PerfEventType> &listAllPerfEventTypes() {
  static const std::vector<PerfEventType> sAllPerfEventTypes{
      // Standard event types, not specific to a target.
      // These are not always useful, as some targets don't always implement
      // them. For instance, AMD Zen4 CPUs do not implement LLC-loads,
      // LLC-load-misses. Table from
      // https://android.googlesource.com/platform/system/extras/+/refs/heads/main/simpleperf/event_type_table.h
      {"cpu-cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "", ""},
      {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "", ""},
      {"cache-references", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES,
       "", ""},
      {"cache-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "", ""},
      {"branch-instructions", PERF_TYPE_HARDWARE,
       PERF_COUNT_HW_BRANCH_INSTRUCTIONS, "", ""},
      {"branch-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, "",
       ""},
      {"bus-cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BUS_CYCLES, "", ""},
      {"stalled-cycles-frontend", PERF_TYPE_HARDWARE,
       PERF_COUNT_HW_STALLED_CYCLES_FRONTEND, "", ""},
      {"stalled-cycles-backend", PERF_TYPE_HARDWARE,
       PERF_COUNT_HW_STALLED_CYCLES_BACKEND, "", ""},
      {"cpu-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_CLOCK, "", ""},
      {"task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, "", ""},
      {"page-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS, "", ""},
      {"context-switches", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES,
       "", ""},
      {"cpu-migrations", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS, "",
       ""},
      {"minor-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN, "",
       ""},
      {"major-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ, "",
       ""},
      {"alignment-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_ALIGNMENT_FAULTS,
       "", ""},
      {"emulation-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_EMULATION_FAULTS,
       "", ""},
      {"L1-dcache-loads", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"L1-dcache-load-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"L1-dcache-stores", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"L1-dcache-store-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"L1-dcache-prefetches", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"L1-dcache-prefetch-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"L1-icache-loads", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_L1I) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"L1-icache-load-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_L1I) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"L1-icache-stores", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_L1I) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"L1-icache-store-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_L1I) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"L1-icache-prefetches", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_L1I) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"L1-icache-prefetch-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_L1I) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"LLC-loads", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"LLC-load-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"LLC-stores", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"LLC-store-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"LLC-prefetches", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"LLC-prefetch-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"dTLB-loads", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_DTLB) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"dTLB-load-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_DTLB) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"dTLB-stores", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_DTLB) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"dTLB-store-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_DTLB) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"dTLB-prefetches", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_DTLB) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"dTLB-prefetch-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_DTLB) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"iTLB-loads", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_ITLB) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"iTLB-load-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_ITLB) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"iTLB-stores", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_ITLB) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"iTLB-store-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_ITLB) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"iTLB-prefetches", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_ITLB) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"iTLB-prefetch-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_ITLB) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"branch-loads", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_BPU) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"branch-load-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_BPU) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"branch-stores", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_BPU) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"branch-store-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_BPU) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"branch-prefetches", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_BPU) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"branch-prefetch-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_BPU) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"node-loads", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_NODE) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"node-load-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_NODE) | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"node-stores", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_NODE) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"node-store-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_NODE) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},
      {"node-prefetches", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_NODE) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)),
       "", ""},
      {"node-prefetch-misses", PERF_TYPE_HW_CACHE,
       ((PERF_COUNT_HW_CACHE_NODE) | (PERF_COUNT_HW_CACHE_OP_PREFETCH << 8) |
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)),
       "", ""},

      // Some AMD events, scraped from:
      // https://elixir.bootlin.com/linux/latest/source/tools/perf/pmu-events/arch/x86/amdzen4
      // The mapping from (event code, umask) to config value is
      //   config = (umask << 8) + (event code)  // assuming event code <= 0xff
      // This is documented under /sys/devices/cpu/format:
      //   $ cat /sys/devices/cpu/format/event
      //   config:0-7,32-35
      //   $ cat /sys/devices/cpu/format/umask
      //   config:8-15
      {"bp_l2_btb_correct", PERF_TYPE_RAW, 0x8b, "AMD",
       "L2 branch prediction overrides existing prediction (speculative)."},
      {"bp_dyn_ind_pred", PERF_TYPE_RAW, 0x8e, "AMD",
       "Dynamic indirect predictions (branch used the indirect predictor to "
       "make a prediction)."},
      {"bp_de_redirect", PERF_TYPE_RAW, 0x91, "AMD",
       "Instruction decoder corrects the predicted target and resteers the "
       "branch predictor."},
      {"ex_ret_brn", PERF_TYPE_RAW, 0xc2, "AMD",
       "Retired branch instructions (all types of architectural control flow "
       "changes, including exceptions and interrupts)."},
      {"ex_ret_brn_misp", PERF_TYPE_RAW, 0xc3, "AMD",
       "Retired branch instructions mispredicted."},
      {"ex_ret_brn_tkn", PERF_TYPE_RAW, 0xc4, "AMD",
       "Retired taken branch instructions (all types of architectural control "
       "flow changes, including exceptions and interrupts)."},
      {"ex_ret_brn_tkn_misp", PERF_TYPE_RAW, 0xc5, "AMD",
       "Retired taken branch instructions mispredicted."},
      {"ex_ret_brn_far", PERF_TYPE_RAW, 0xc6, "AMD",
       "Retired far control transfers (far call/jump/return, IRET, SYSCALL and "
       "SYSRET, plus exceptions and interrupts). Far control transfers are not "
       "subject to branch prediction."},
      {"ex_ret_near_ret", PERF_TYPE_RAW, 0xc8, "AMD",
       "Retired near returns (RET or RET Iw)."},
      {"ex_ret_near_ret_mispred", PERF_TYPE_RAW, 0xc9, "AMD",
       "Retired near returns mispredicted. Each misprediction incurs the same "
       "penalty as a mispredicted conditional branch instruction."},
      {"ex_ret_brn_ind_misp", PERF_TYPE_RAW, 0xca, "AMD",
       "Retired indirect branch instructions mispredicted (only EX "
       "mispredicts). "
       "Each misprediction incurs the same penalty as a mispredicted "
       "conditional branch instruction."},
      {"ex_ret_ind_brch_instr", PERF_TYPE_RAW, 0xcc, "AMD",
       "Retired indirect branch instructions."},
      {"ex_ret_cond", PERF_TYPE_RAW, 0xd1, "AMD",
       "Retired conditional branch instructions."},
      {"ex_ret_msprd_brnch_instr_dir_msmtch", PERF_TYPE_RAW, 0x1c7, "AMD",
       "Retired branch instructions mispredicted due to direction mismatch."},
      {"ex_ret_uncond_brnch_instr_mispred", PERF_TYPE_RAW, 0x1c8, "AMD",
       "Retired unconditional indirect branch instructions mispredicted."},
      {"ex_ret_uncond_brnch_instr", PERF_TYPE_RAW, 0x1c9, "AMD",
       "Retired unconditional branch instructions."},
      {"ls_mab_alloc.load_store_allocations", PERF_TYPE_RAW, 0x3f41, "AMD",
       "Miss Address Buffer (MAB) entries allocated by a Load-Store (LS) pipe "
       "for load-store allocations."},
      {"ls_mab_alloc.hardware_prefetcher_allocations", PERF_TYPE_RAW, 0x4041,
       "AMD",
       "Miss Address Buffer (MAB) entries allocated by a Load-Store (LS) pipe "
       "for hardware prefetcher allocations."},
      {"ls_mab_alloc.all_allocations", PERF_TYPE_RAW, 0x7f41, "AMD",
       "Miss Address Buffer (MAB) entries allocated by a Load-Store (LS) pipe "
       "for all types of allocations."},
      {"ls_dmnd_fills_from_sys.local_l2", PERF_TYPE_RAW, 0x143, "AMD",
       "Demand data cache fills from local L2 cache."},
      {"ls_dmnd_fills_from_sys.local_ccx", PERF_TYPE_RAW, 0x243, "AMD",
       "Demand data cache fills from L3 cache or different L2 cache in the "
       "same CCX."},
      {"ls_dmnd_fills_from_sys.near_cache", PERF_TYPE_RAW, 0x443, "AMD",
       "Demand data cache fills from cache of another CCX when the address was "
       "in the same NUMA node."},
      {"ls_dmnd_fills_from_sys.dram_io_near", PERF_TYPE_RAW, 0x843, "AMD",
       "Demand data cache fills from either DRAM or MMIO in the same NUMA "
       "node."},
      {"ls_dmnd_fills_from_sys.far_cache", PERF_TYPE_RAW, 0x1043, "AMD",
       "Demand data cache fills from cache of another CCX when the address was "
       "in a different NUMA node."},
      {"ls_dmnd_fills_from_sys.dram_io_far", PERF_TYPE_RAW, 0x4043, "AMD",
       "Demand data cache fills from either DRAM or MMIO in a different NUMA "
       "node (same or different socket)."},
      {"ls_dmnd_fills_from_sys.alternate_memories", PERF_TYPE_RAW, 0x8043,
       "AMD", "Demand data cache fills from extension memory."},
      {"ls_dmnd_fills_from_sys.all", PERF_TYPE_RAW, 0xff43, "AMD",
       "Demand data cache fills from all types of data sources."},
      {"ls_any_fills_from_sys.local_l2", PERF_TYPE_RAW, 0x144, "AMD",
       "Any data cache fills from local L2 cache."},
      {"ls_any_fills_from_sys.local_ccx", PERF_TYPE_RAW, 0x244, "AMD",
       "Any data cache fills from L3 cache or different L2 cache in the same "
       "CCX."},
      {"ls_any_fills_from_sys.local_all", PERF_TYPE_RAW, 0x344, "AMD",
       "Any data cache fills from local L2 cache or L3 cache or different L2 "
       "cache in the same CCX."},
      {"ls_any_fills_from_sys.near_cache", PERF_TYPE_RAW, 0x444, "AMD",
       "Any data cache fills from cache of another CCX when the address was in "
       "the same NUMA node."},
      {"ls_any_fills_from_sys.dram_io_near", PERF_TYPE_RAW, 0x844, "AMD",
       "Any data cache fills from either DRAM or MMIO in the same NUMA node."},
      {"ls_any_fills_from_sys.far_cache", PERF_TYPE_RAW, 0x1044, "AMD",
       "Any data cache fills from cache of another CCX when the address was in "
       "a different NUMA node."},
      {"ls_any_fills_from_sys.remote_cache", PERF_TYPE_RAW, 0x1444, "AMD",
       "Any data cache fills from cache of another CCX when the address was in "
       "the same or a different NUMA node."},
      {"ls_any_fills_from_sys.dram_io_far", PERF_TYPE_RAW, 0x4044, "AMD",
       "Any data cache fills from either DRAM or MMIO in a different NUMA node "
       "(same or different socket)."},
      {"ls_any_fills_from_sys.dram_io_all", PERF_TYPE_RAW, 0x4844, "AMD",
       "Any data cache fills from either DRAM or MMIO in any NUMA node (same "
       "or different socket)."},
      {"ls_any_fills_from_sys.far_all", PERF_TYPE_RAW, 0x5044, "AMD",
       "Any data cache fills from either cache of another CCX, DRAM or MMIO "
       "when "
       "the address was in a different NUMA node (same or different socket)."},
      {"ls_any_fills_from_sys.all_dram_io", PERF_TYPE_RAW, 0x4844, "AMD",
       "Any data cache fills from either DRAM or MMIO in any NUMA node (same "
       "or different socket)."},
      {"ls_any_fills_from_sys.alternate_memories", PERF_TYPE_RAW, 0x8044, "AMD",
       "Any data cache fills from extension memory."},
      {"ls_any_fills_from_sys.all", PERF_TYPE_RAW, 0xff44, "AMD",
       "Any data cache fills from all types of data sources."},
      {"ls_pref_instr_disp.prefetch", PERF_TYPE_RAW, 0x14b, "AMD",
       "Software prefetch instructions dispatched (speculative) of type "
       "PrefetchT0 (move data to all cache levels), T1 (move data to all cache "
       "levels except L1) and T2 (move data to all cache levels except L1 and "
       "L2)."},
      {"ls_pref_instr_disp.prefetch_w", PERF_TYPE_RAW, 0x24b, "AMD",
       "Software prefetch instructions dispatched (speculative) of type "
       "PrefetchW (move data to L1 cache and mark it modifiable)."},
      {"ls_pref_instr_disp.prefetch_nta", PERF_TYPE_RAW, 0x44b, "AMD",
       "Software prefetch instructions dispatched (speculative) of type "
       "PrefetchNTA (move data with minimum cache pollution i.e. non-temporal "
       "access)."},
      {"ls_pref_instr_disp.all", PERF_TYPE_RAW, 0x74b, "AMD",
       "Software prefetch instructions dispatched (speculative) of all types."},
      {"ls_inef_sw_pref.data_pipe_sw_pf_dc_hit", PERF_TYPE_RAW, 0x152, "AMD",
       "Software prefetches that did not fetch data outside of the processor "
       "core as the PREFETCH instruction saw a data cache hit."},
      {"ls_inef_sw_pref.mab_mch_cnt", PERF_TYPE_RAW, 0x252, "AMD",
       "Software prefetches that did not fetch data outside of the processor "
       "core as the PREFETCH instruction saw a match on an already allocated "
       "Miss Address Buffer (MAB)."},
      {"ls_inef_sw_pref.all", PERF_TYPE_RAW, 0x352, "AMD", ""},
      {"ls_sw_pf_dc_fills.local_l2", PERF_TYPE_RAW, 0x159, "AMD",
       "Software prefetch data cache fills from local L2 cache."},
      {"ls_sw_pf_dc_fills.local_ccx", PERF_TYPE_RAW, 0x259, "AMD",
       "Software prefetch data cache fills from L3 cache or different L2 cache "
       "in the same CCX."},
      {"ls_sw_pf_dc_fills.near_cache", PERF_TYPE_RAW, 0x459, "AMD",
       "Software prefetch data cache fills from cache of another CCX in the "
       "same NUMA node."},
      {"ls_sw_pf_dc_fills.dram_io_near", PERF_TYPE_RAW, 0x859, "AMD",
       "Software prefetch data cache fills from either DRAM or MMIO in the "
       "same NUMA node."},
      {"ls_sw_pf_dc_fills.far_cache", PERF_TYPE_RAW, 0x1059, "AMD",
       "Software prefetch data cache fills from cache of another CCX in a "
       "different NUMA node."},
      {"ls_sw_pf_dc_fills.dram_io_far", PERF_TYPE_RAW, 0x4059, "AMD",
       "Software prefetch data cache fills from either DRAM or MMIO in a "
       "different NUMA node (same or different socket)."},
      {"ls_sw_pf_dc_fills.alternate_memories", PERF_TYPE_RAW, 0x8059, "AMD",
       "Software prefetch data cache fills from extension memory."},
      {"ls_sw_pf_dc_fills.all", PERF_TYPE_RAW, 0xdf59, "AMD",
       "Software prefetch data cache fills from all types of data sources."},
      {"ls_hw_pf_dc_fills.local_l2", PERF_TYPE_RAW, 0x15a, "AMD",
       "Hardware prefetch data cache fills from local L2 cache."},
      {"ls_hw_pf_dc_fills.local_ccx", PERF_TYPE_RAW, 0x25a, "AMD",
       "Hardware prefetch data cache fills from L3 cache or different L2 cache "
       "in the same CCX."},
      {"ls_hw_pf_dc_fills.near_cache", PERF_TYPE_RAW, 0x45a, "AMD",
       "Hardware prefetch data cache fills from cache of another CCX when the "
       "address was in the same NUMA node."},
      {"ls_hw_pf_dc_fills.dram_io_near", PERF_TYPE_RAW, 0x85a, "AMD",
       "Hardware prefetch data cache fills from either DRAM or MMIO in the "
       "same NUMA node."},
      {"ls_hw_pf_dc_fills.far_cache", PERF_TYPE_RAW, 0x105a, "AMD",
       "Hardware prefetch data cache fills from cache of another CCX when the "
       "address was in a different NUMA node."},
      {"ls_hw_pf_dc_fills.dram_io_far", PERF_TYPE_RAW, 0x405a, "AMD",
       "Hardware prefetch data cache fills from either DRAM or MMIO in a "
       "different NUMA node (same or different socket)."},
      {"ls_hw_pf_dc_fills.alternate_memories", PERF_TYPE_RAW, 0x805a, "AMD",
       "Hardware prefetch data cache fills from extension memory."},
      {"ls_hw_pf_dc_fills.all", PERF_TYPE_RAW, 0xdf5a, "AMD",
       "Hardware prefetch data cache fills from all types of data sources."},
      {"ls_alloc_mab_count", PERF_TYPE_RAW, 0x5f, "AMD",
       "In-flight L1 data cache misses i.e. Miss Address Buffer (MAB) "
       "allocations each cycle."},
      {"l2_request_g1.group2", PERF_TYPE_RAW, 0x160, "AMD",
       "L2 cache requests of non-cacheable type (non-cached data and "
       "instructions reads, self-modifying code checks)."},
      {"l2_request_g1.l2_hw_pf", PERF_TYPE_RAW, 0x260, "AMD",
       "L2 cache requests: from hardware prefetchers to prefetch directly into "
       "L2 (hit or miss)."},
      {"l2_request_g1.prefetch_l2_cmd", PERF_TYPE_RAW, 0x460, "AMD",
       "L2 cache requests: prefetch directly into L2."},
      {"l2_request_g1.change_to_x", PERF_TYPE_RAW, 0x860, "AMD",
       "L2 cache requests: data cache state change to writable, check L2 for "
       "current state."},
      {"l2_request_g1.cacheable_ic_read", PERF_TYPE_RAW, 0x1060, "AMD",
       "L2 cache requests: instruction cache reads."},
      {"l2_request_g1.ls_rd_blk_c_s", PERF_TYPE_RAW, 0x2060, "AMD",
       "L2 cache requests: data cache shared reads."},
      {"l2_request_g1.rd_blk_x", PERF_TYPE_RAW, 0x4060, "AMD",
       "L2 cache requests: data cache stores."},
      {"l2_request_g1.rd_blk_l", PERF_TYPE_RAW, 0x8060, "AMD",
       "L2 cache requests: data cache reads including hardware and software "
       "prefetch."},
      {"l2_request_g1.all_dc", PERF_TYPE_RAW, 0xe860, "AMD",
       "L2 cache requests of common types from L1 data cache (including "
       "prefetches)."},
      {"l2_request_g1.all_no_prefetch", PERF_TYPE_RAW, 0xf960, "AMD",
       "L2 cache requests of common types not including prefetches."},
      {"l2_request_g1.all", PERF_TYPE_RAW, 0xff60, "AMD",
       "L2 cache requests of all types."},
      {"l2_cache_req_stat.ic_fill_miss", PERF_TYPE_RAW, 0x164, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) with status: "
       "instruction cache request miss in L2."},
      {"l2_cache_req_stat.ic_fill_hit_s", PERF_TYPE_RAW, 0x264, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) with status: "
       "instruction cache hit non-modifiable line in L2."},
      {"l2_cache_req_stat.ic_fill_hit_x", PERF_TYPE_RAW, 0x464, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) with status: "
       "instruction cache hit modifiable line in L2."},
      {"l2_cache_req_stat.ic_hit_in_l2", PERF_TYPE_RAW, 0x664, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) for instruction "
       "cache hits."},
      {"l2_cache_req_stat.ic_access_in_l2", PERF_TYPE_RAW, 0x764, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) for instruction "
       "cache access."},
      {"l2_cache_req_stat.ls_rd_blk_c", PERF_TYPE_RAW, 0x864, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) with status: "
       "data cache request miss in L2."},
      {"l2_cache_req_stat.ic_dc_miss_in_l2", PERF_TYPE_RAW, 0x964, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) for data and "
       "instruction cache misses."},
      {"l2_cache_req_stat.ls_rd_blk_x", PERF_TYPE_RAW, 0x1064, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) with status: "
       "data cache store or state change hit in L2."},
      {"l2_cache_req_stat.ls_rd_blk_l_hit_s", PERF_TYPE_RAW, 0x2064, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) with status: "
       "data cache read hit non-modifiable line in L2."},
      {"l2_cache_req_stat.ls_rd_blk_l_hit_x", PERF_TYPE_RAW, 0x4064, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) with status: "
       "data cache read hit modifiable line in L2."},
      {"l2_cache_req_stat.ls_rd_blk_cs", PERF_TYPE_RAW, 0x8064, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) with status: "
       "data cache shared read hit in L2."},
      {"l2_cache_req_stat.dc_hit_in_l2", PERF_TYPE_RAW, 0xf064, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) for data cache "
       "hits."},
      {"l2_cache_req_stat.ic_dc_hit_in_l2", PERF_TYPE_RAW, 0xf664, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) for data and "
       "instruction cache hits."},
      {"l2_cache_req_stat.dc_access_in_l2", PERF_TYPE_RAW, 0xf864, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) for data cache "
       "access."},
      {"l2_cache_req_stat.all", PERF_TYPE_RAW, 0xff64, "AMD",
       "Core to L2 cache requests (not including L2 prefetch) for data and "
       "instruction cache access."},
      {"l2_pf_hit_l2.l2_stream", PERF_TYPE_RAW, 0x170, "AMD",
       "L2 prefetches accepted by the L2 pipeline which hit in the L2 cache of "
       "type L2Stream (fetch additional sequential lines into L2 cache)."},
      {"l2_pf_hit_l2.l2_next_line", PERF_TYPE_RAW, 0x270, "AMD",
       "L2 prefetches accepted by the L2 pipeline which hit in the L2 cache of "
       "type L2NextLine (fetch the next line into L2 cache)."},
      {"l2_pf_hit_l2.l2_up_down", PERF_TYPE_RAW, 0x470, "AMD",
       "L2 prefetches accepted by the L2 pipeline which hit in the L2 cache of "
       "type L2UpDown (fetch the next or previous line into L2 cache for all "
       "memory accesses)."},
      {"l2_pf_hit_l2.l2_burst", PERF_TYPE_RAW, 0x870, "AMD",
       "L2 prefetches accepted by the L2 pipeline which hit in the L2 cache of "
       "type L2Burst (aggressively fetch additional sequential lines into L2 "
       "cache)."},
      {"l2_pf_hit_l2.l2_stride", PERF_TYPE_RAW, 0x1070, "AMD",
       "L2 prefetches accepted by the L2 pipeline which hit in the L2 cache of "
       "type L2Stride (fetch additional lines into L2 cache when each access "
       "is at a constant distance from the previous)."},
      {"l2_pf_hit_l2.l1_stream", PERF_TYPE_RAW, 0x2070, "AMD",
       "L2 prefetches accepted by the L2 pipeline which hit in the L2 cache of "
       "type L1Stream (fetch additional sequential lines into L1 cache)."},
      {"l2_pf_hit_l2.l1_stride", PERF_TYPE_RAW, 0x4070, "AMD",
       "L2 prefetches accepted by the L2 pipeline which hit in the L2 cache of "
       "type L1Stride (fetch additional lines into L1 cache when each access "
       "is a constant distance from the previous)."},
      {"l2_pf_hit_l2.l1_region", PERF_TYPE_RAW, 0x8070, "AMD",
       "L2 prefetches accepted by the L2 pipeline which hit in the L2 cache of "
       "type L1Region (fetch additional lines into L1 cache when the data "
       "access for a given instruction tends to be followed by a consistent "
       "pattern "
       "of other accesses within a localized region)."},
      {"l2_pf_hit_l2.all", PERF_TYPE_RAW, 0xff70, "AMD",
       "L2 prefetches accepted by the L2 pipeline which hit in the L2 cache of "
       "all types."},
      {"l2_pf_miss_l2_hit_l3.l2_stream", PERF_TYPE_RAW, 0x171, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 cache and "
       "hit in the L3 cache of type L2Stream (fetch additional sequential "
       "lines into L2 cache)."},
      {"l2_pf_miss_l2_hit_l3.l2_next_line", PERF_TYPE_RAW, 0x271, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 cache and "
       "hit in the L3 cache of type L2NextLine (fetch the next line into L2 "
       "cache)."},
      {"l2_pf_miss_l2_hit_l3.l2_up_down", PERF_TYPE_RAW, 0x471, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 cache and "
       "hit in the L3 cache of type L2UpDown (fetch the next or previous line "
       "into L2 cache for all memory accesses)."},
      {"l2_pf_miss_l2_hit_l3.l2_burst", PERF_TYPE_RAW, 0x871, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 cache and "
       "hit in the L3 cache of type L2Burst (aggressively fetch additional "
       "sequential lines into L2 cache)."},
      {"l2_pf_miss_l2_hit_l3.l2_stride", PERF_TYPE_RAW, 0x1071, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 cache and "
       "hit in the L3 cache of type L2Stride (fetch additional lines into L2 "
       "cache when each access is a constant distance from the previous)."},
      {"l2_pf_miss_l2_hit_l3.l1_stream", PERF_TYPE_RAW, 0x2071, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 cache and "
       "hit in the L3 cache of type L1Stream (fetch additional sequential "
       "lines into L1 cache)."},
      {"l2_pf_miss_l2_hit_l3.l1_stride", PERF_TYPE_RAW, 0x4071, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 cache and "
       "hit in the L3 cache of type L1Stride (fetch additional lines into L1 "
       "cache when each access is a constant distance from the previous)."},
      {"l2_pf_miss_l2_hit_l3.l1_region", PERF_TYPE_RAW, 0x8071, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 cache and "
       "hit in the L3 cache of type L1Region (fetch additional lines into L1 "
       "cache when the data access for a given instruction tends to be "
       "followed by a consistent pattern of other accesses within a localized "
       "region)."},
      {"l2_pf_miss_l2_hit_l3.all", PERF_TYPE_RAW, 0xff71, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 cache and "
       "hit in the L3 cache cache of all types."},
      {"l2_pf_miss_l2_l3.l2_stream", PERF_TYPE_RAW, 0x172, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 and the L3 "
       "caches of type L2Stream (fetch additional sequential lines into L2 "
       "cache)."},
      {"l2_pf_miss_l2_l3.l2_next_line", PERF_TYPE_RAW, 0x272, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 and the L3 "
       "caches of type L2NextLine (fetch the next line into L2 cache)."},
      {"l2_pf_miss_l2_l3.l2_up_down", PERF_TYPE_RAW, 0x472, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 and the L3 "
       "caches of type L2UpDown (fetch the next or previous line into L2 cache "
       "for all memory accesses)."},
      {"l2_pf_miss_l2_l3.l2_burst", PERF_TYPE_RAW, 0x872, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 and the L3 "
       "caches of type L2Burst (aggressively fetch additional sequential lines "
       "into L2 cache)."},
      {"l2_pf_miss_l2_l3.l2_stride", PERF_TYPE_RAW, 0x1072, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 and the L3 "
       "caches of type L2Stride (fetch additional lines into L2 cache when "
       "each access is a constant distance from the previous)."},
      {"l2_pf_miss_l2_l3.l1_stream", PERF_TYPE_RAW, 0x2072, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 and the L3 "
       "caches of type L1Stream (fetch additional sequential lines into L1 "
       "cache)."},
      {"l2_pf_miss_l2_l3.l1_stride", PERF_TYPE_RAW, 0x4072, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 and the L3 "
       "caches of type L1Stride (fetch additional lines into L1 cache when "
       "each access is a constant distance from the previous)."},
      {"l2_pf_miss_l2_l3.l1_region", PERF_TYPE_RAW, 0x8072, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 and the L3 "
       "caches of type L1Region (fetch additional lines into L1 cache when the "
       "data access for a given instruction tends to be followed by a "
       "consistent "
       "pattern of other accesses within a localized region)."},
      {"l2_pf_miss_l2_l3.all", PERF_TYPE_RAW, 0xff72, "AMD",
       "L2 prefetches accepted by the L2 pipeline which miss the L2 and the L3 "
       "caches of all types."},
      {"ic_cache_fill_l2", PERF_TYPE_RAW, 0x82, "AMD",
       "Instruction cache lines (64 bytes) fulfilled from the L2 cache."},
      {"ic_cache_fill_sys", PERF_TYPE_RAW, 0x83, "AMD",
       "Instruction cache lines (64 bytes) fulfilled from system memory or "
       "another cache."},
      {"ic_tag_hit_miss.instruction_cache_hit", PERF_TYPE_RAW, 0x88e, "AMD",
       "Instruction cache hits."},
      {"ic_tag_hit_miss.instruction_cache_miss", PERF_TYPE_RAW, 0x198e, "AMD",
       "Instruction cache misses."},
      {"ic_tag_hit_miss.all_instruction_cache_accesses", PERF_TYPE_RAW, 0x208e,
       "AMD", "Instruction cache accesses of all types."},
      {"op_cache_hit_miss.op_cache_hit", PERF_TYPE_RAW, 0x58f, "AMD",
       "Op cache hits."},
      {"op_cache_hit_miss.op_cache_miss", PERF_TYPE_RAW, 0x68f, "AMD",
       "Op cache misses."},
      {"op_cache_hit_miss.all_op_cache_accesses", PERF_TYPE_RAW, 0x98f, "AMD",
       "Op cache accesses of all types."},
      {"l3_lookup_state.l3_miss", PERF_TYPE_RAW, 0x104, "AMD",
       "L3 cache misses."},
      {"l3_lookup_state.l3_hit", PERF_TYPE_RAW, 0xfe04, "AMD",
       "L3 cache hits."},
      {"l3_lookup_state.all_coherent_accesses_to_l3", PERF_TYPE_RAW, 0xff04,
       "AMD", "L3 cache requests for all coherent accesses."},
      {"l3_xi_sampled_latency.dram_near", PERF_TYPE_RAW, 0x1ac, "AMD",
       "Average sampled latency when data is sourced from DRAM in the same "
       "NUMA node."},
      {"l3_xi_sampled_latency.dram_far", PERF_TYPE_RAW, 0x2ac, "AMD",
       "Average sampled latency when data is sourced from DRAM in a different "
       "NUMA node."},
      {"l3_xi_sampled_latency.near_cache", PERF_TYPE_RAW, 0x4ac, "AMD",
       "Average sampled latency when data is sourced from another CCX's cache "
       "when the address was in the same NUMA node."},
      {"l3_xi_sampled_latency.far_cache", PERF_TYPE_RAW, 0x8ac, "AMD",
       "Average sampled latency when data is sourced from another CCX's cache "
       "when the address was in a different NUMA node."},
      {"l3_xi_sampled_latency.ext_near", PERF_TYPE_RAW, 0x10ac, "AMD",
       "Average sampled latency when data is sourced from extension memory "
       "(CXL) in the same NUMA node."},
      {"l3_xi_sampled_latency.ext_far", PERF_TYPE_RAW, 0x20ac, "AMD",
       "Average sampled latency when data is sourced from extension memory "
       "(CXL) in a different NUMA node."},
      {"l3_xi_sampled_latency.all", PERF_TYPE_RAW, 0x3fac, "AMD",
       "Average sampled latency from all data sources."},
      {"l3_xi_sampled_latency_requests.dram_near", PERF_TYPE_RAW, 0x1ad, "AMD",
       "L3 cache fill requests sourced from DRAM in the same NUMA node."},
      {"l3_xi_sampled_latency_requests.dram_far", PERF_TYPE_RAW, 0x2ad, "AMD",
       "L3 cache fill requests sourced from DRAM in a different NUMA node."},
      {"l3_xi_sampled_latency_requests.near_cache", PERF_TYPE_RAW, 0x4ad, "AMD",
       "L3 cache fill requests sourced from another CCX's cache when the "
       "address was in the same NUMA node."},
      {"l3_xi_sampled_latency_requests.far_cache", PERF_TYPE_RAW, 0x8ad, "AMD",
       "L3 cache fill requests sourced from another CCX's cache when the "
       "address was in a different NUMA node."},
      {"l3_xi_sampled_latency_requests.ext_near", PERF_TYPE_RAW, 0x10ad, "AMD",
       "L3 cache fill requests sourced from extension memory (CXL) in the same "
       "NUMA node."},
      {"l3_xi_sampled_latency_requests.ext_far", PERF_TYPE_RAW, 0x20ad, "AMD",
       "L3 cache fill requests sourced from extension memory (CXL) in a "
       "different NUMA node."},
      {"l3_xi_sampled_latency_requests.all", PERF_TYPE_RAW, 0x3fad, "AMD",
       "L3 cache fill requests sourced from all data sources."},
      {"ls_locks.bus_lock", PERF_TYPE_RAW, 0x125, "AMD",
       "Retired Lock instructions which caused a bus lock."},
      {"ls_ret_cl_flush", PERF_TYPE_RAW, 0x26, "AMD",
       "Retired CLFLUSH instructions."},
      {"ls_ret_cpuid", PERF_TYPE_RAW, 0x27, "AMD",
       "Retired CPUID instructions."},
      {"ls_smi_rx", PERF_TYPE_RAW, 0x2b, "AMD", "SMIs received."},
      {"ls_int_taken", PERF_TYPE_RAW, 0x2c, "AMD", "Interrupts taken."},
      {"ls_not_halted_cyc", PERF_TYPE_RAW, 0x76, "AMD",
       "Core cycles not in halt."},
      {"ex_ret_instr", PERF_TYPE_RAW, 0xc0, "AMD", "Retired instructions."},
      {"ex_ret_ops", PERF_TYPE_RAW, 0xc1, "AMD", "Retired macro-ops."},
      {"ex_div_busy", PERF_TYPE_RAW, 0xd3, "AMD",
       "Number of cycles the divider is busy."},
      {"ex_div_count", PERF_TYPE_RAW, 0xd4, "AMD", "Divide ops executed."},
      {"ex_no_retire.empty", PERF_TYPE_RAW, 0x1d6, "AMD",
       "Cycles with no retire due  to the lack of valid ops in the retire "
       "queue (may be caused by front-end bottlenecks or pipeline redirects)."},
      {"ex_no_retire.not_complete", PERF_TYPE_RAW, 0x2d6, "AMD",
       "Cycles with no retire while the oldest op is waiting to be executed."},
      {"ex_no_retire.other", PERF_TYPE_RAW, 0x8d6, "AMD",
       "Cycles with no retire caused by other reasons (retire breaks, traps, "
       "faults, etc.)."},
      {"ex_no_retire.thread_not_selected", PERF_TYPE_RAW, 0x10d6, "AMD",
       "Cycles with no retire because thread arbitration did not select the "
       "thread."},
      {"ex_no_retire.load_not_complete", PERF_TYPE_RAW, 0xa2d6, "AMD",
       "Cycles with no retire while the oldest op is waiting for load data."},
      {"ex_no_retire.all", PERF_TYPE_RAW, 0x1bd6, "AMD",
       "Cycles with no retire for any reason."},
      {"ls_not_halted_p0_cyc.p0_freq_cyc", PERF_TYPE_RAW, 0x220, "AMD",
       "Reference cycles (P0 frequency) not in halt ."},
      {"ex_ret_ucode_instr", PERF_TYPE_RAW, 0x1c1, "AMD",
       "Retired microcoded instructions."},
      {"ex_ret_ucode_ops", PERF_TYPE_RAW, 0x1c2, "AMD",
       "Retired microcode ops."},
      {"ex_tagged_ibs_ops.ibs_tagged_ops", PERF_TYPE_RAW, 0x2cf, "AMD",
       "Ops tagged by IBS."},
      {"ex_tagged_ibs_ops.ibs_tagged_ops_ret", PERF_TYPE_RAW, 0x3cf, "AMD",
       "Ops tagged by IBS that retired."},
      {"ex_ret_fused_instr", PERF_TYPE_RAW, 0x1d0, "AMD",
       "Retired fused instructions."},
      {"ls_bad_status2.stli_other", PERF_TYPE_RAW, 0x224, "AMD",
       "Store-to-load conflicts (load unable to complete due to a "
       "non-forwardable conflict with an older store)."},
      {"ls_dispatch.ld_dispatch", PERF_TYPE_RAW, 0x129, "AMD",
       "Number of memory load operations dispatched to the load-store unit."},
      {"ls_dispatch.store_dispatch", PERF_TYPE_RAW, 0x229, "AMD",
       "Number of memory store operations dispatched to the load-store unit."},
      {"ls_dispatch.ld_st_dispatch", PERF_TYPE_RAW, 0x429, "AMD",
       "Number of memory load-store operations dispatched to the load-store "
       "unit."},
      {"ls_stlf", PERF_TYPE_RAW, 0x35, "AMD",
       "Store-to-load-forward (STLF) hits."},
      {"ls_st_commit_cancel2.st_commit_cancel_wcb_full", PERF_TYPE_RAW, 0x137,
       "AMD",
       "Non-cacheable store commits cancelled due to the non-cacheable commit "
       "buffer being full."},
      {"ls_l1_d_tlb_miss.tlb_reload_4k_l2_hit", PERF_TYPE_RAW, 0x145, "AMD",
       "L1 DTLB misses with L2 DTLB hits for 4k pages."},
      {"ls_l1_d_tlb_miss.tlb_reload_coalesced_page_hit", PERF_TYPE_RAW, 0x245,
       "AMD",
       "L1 DTLB misses with L2 DTLB hits for coalesced pages. A coalesced page "
       "is a 16k page created from four adjacent 4k pages."},
      {"ls_l1_d_tlb_miss.tlb_reload_2m_l2_hit", PERF_TYPE_RAW, 0x445, "AMD",
       "L1 DTLB misses with L2 DTLB hits for 2M pages."},
      {"ls_l1_d_tlb_miss.tlb_reload_1g_l2_hit", PERF_TYPE_RAW, 0x845, "AMD",
       "L1 DTLB misses with L2 DTLB hits for 1G pages."},
      {"ls_l1_d_tlb_miss.tlb_reload_4k_l2_miss", PERF_TYPE_RAW, 0x1045, "AMD",
       "L1 DTLB misses with L2 DTLB misses (page-table walks are requested) "
       "for 4k pages."},
      {"ls_l1_d_tlb_miss.tlb_reload_coalesced_page_miss", PERF_TYPE_RAW, 0x2045,
       "AMD",
       "L1 DTLB misses with L2 DTLB misses (page-table walks are requested) "
       "for coalesced pages. A coalesced page is a 16k page created from four "
       "adjacent 4k pages."},
      {"ls_l1_d_tlb_miss.tlb_reload_2m_l2_miss", PERF_TYPE_RAW, 0x4045, "AMD",
       "L1 DTLB misses with L2 DTLB misses (page-table walks are requested) "
       "for 2M pages."},
      {"ls_l1_d_tlb_miss.tlb_reload_1g_l2_miss", PERF_TYPE_RAW, 0x8045, "AMD",
       "L1 DTLB misses with L2 DTLB misses (page-table walks are requested) "
       "for 1G pages."},
      {"ls_l1_d_tlb_miss.all_l2_miss", PERF_TYPE_RAW, 0xf045, "AMD",
       "L1 DTLB misses with L2 DTLB misses (page-table walks are requested) "
       "for all page sizes."},
      {"ls_l1_d_tlb_miss.all", PERF_TYPE_RAW, 0xff45, "AMD",
       "L1 DTLB misses for all page sizes."},
      {"ls_misal_loads.ma64", PERF_TYPE_RAW, 0x147, "AMD",
       "64B misaligned (cacheline crossing) loads."},
      {"ls_misal_loads.ma4k", PERF_TYPE_RAW, 0x247, "AMD",
       "4kB misaligned (page crossing) loads."},
      {"ls_tlb_flush.all", PERF_TYPE_RAW, 0xff78, "AMD", "All TLB Flushes."},
      {"bp_l1_tlb_miss_l2_tlb_hit", PERF_TYPE_RAW, 0x84, "AMD",
       "Instruction fetches that miss in the L1 ITLB but hit in the L2 ITLB."},
      {"bp_l1_tlb_miss_l2_tlb_miss.if4k", PERF_TYPE_RAW, 0x185, "AMD",
       "Instruction fetches that miss in both the L1 and L2 ITLBs (page-table "
       "walks are requested) for 4k pages."},
      {"bp_l1_tlb_miss_l2_tlb_miss.if2m", PERF_TYPE_RAW, 0x285, "AMD",
       "Instruction fetches that miss in both the L1 and L2 ITLBs (page-table "
       "walks are requested) for 2M pages."},
      {"bp_l1_tlb_miss_l2_tlb_miss.if1g", PERF_TYPE_RAW, 0x485, "AMD",
       "Instruction fetches that miss in both the L1 and L2 ITLBs (page-table "
       "walks are requested) for 1G pages."},
      {"bp_l1_tlb_miss_l2_tlb_miss.coalesced_4k", PERF_TYPE_RAW, 0x885, "AMD",
       "Instruction fetches that miss in both the L1 and L2 ITLBs (page-table "
       "walks are requested) for coalesced pages. A coalesced page is a 16k "
       "page created from four adjacent 4k pages."},
      {"bp_l1_tlb_miss_l2_tlb_miss.all", PERF_TYPE_RAW, 0xf85, "AMD",
       "Instruction fetches that miss in both the L1 and L2 ITLBs (page-table "
       "walks are requested) for all page sizes."},
      {"bp_l1_tlb_fetch_hit.if4k", PERF_TYPE_RAW, 0x194, "AMD",
       "Instruction fetches that hit in the L1 ITLB for 4k or coalesced pages. "
       "A coalesced page is a 16k page created from four adjacent 4k pages."},
      {"bp_l1_tlb_fetch_hit.if2m", PERF_TYPE_RAW, 0x294, "AMD",
       "Instruction fetches that hit in the L1 ITLB for 2M pages."},
      {"bp_l1_tlb_fetch_hit.if1g", PERF_TYPE_RAW, 0x494, "AMD",
       "Instruction fetches that hit in the L1 ITLB for 1G pages."},
      {"bp_l1_tlb_fetch_hit.all", PERF_TYPE_RAW, 0x794, "AMD",
       "Instruction fetches that hit in the L1 ITLB for all page sizes."},
      {"fp_ret_x87_fp_ops.add_sub_ops", PERF_TYPE_RAW, 0x102, "AMD",
       "Retired x87 floating-point add and subtract ops."},
      {"fp_ret_x87_fp_ops.mul_ops", PERF_TYPE_RAW, 0x202, "AMD",
       "Retired x87 floating-point multiply ops."},
      {"fp_ret_x87_fp_ops.div_sqrt_ops", PERF_TYPE_RAW, 0x402, "AMD",
       "Retired x87 floating-point divide and square root ops."},
      {"fp_ret_x87_fp_ops.all", PERF_TYPE_RAW, 0x702, "AMD",
       "Retired x87 floating-point ops of all types."},
      {"fp_ret_sse_avx_ops.add_sub_flops", PERF_TYPE_RAW, 0x103, "AMD",
       "Retired SSE and AVX floating-point add and subtract ops."},
      {"fp_ret_sse_avx_ops.mult_flops", PERF_TYPE_RAW, 0x203, "AMD",
       "Retired SSE and AVX floating-point multiply ops."},
      {"fp_ret_sse_avx_ops.div_flops", PERF_TYPE_RAW, 0x403, "AMD",
       "Retired SSE and AVX floating-point divide and square root ops."},
      {"fp_ret_sse_avx_ops.mac_flops", PERF_TYPE_RAW, 0x803, "AMD",
       "Retired SSE and AVX floating-point multiply-accumulate ops (each "
       "operation is counted as 2 ops)."},
      {"fp_ret_sse_avx_ops.bfloat_mac_flops", PERF_TYPE_RAW, 0x1003, "AMD",
       "Retired SSE and AVX floating-point bfloat multiply-accumulate ops "
       "(each operation is counted as 2 ops)."},
      {"fp_ret_sse_avx_ops.all", PERF_TYPE_RAW, 0x1f03, "AMD",
       "Retired SSE and AVX floating-point ops of all types."},
      {"fp_retired_ser_ops.x87_ctrl_ret", PERF_TYPE_RAW, 0x105, "AMD",
       "Retired x87 control word mispredict traps due to mispredictions in RC "
       "or PC, or changes in exception mask bits."},
      {"fp_retired_ser_ops.x87_bot_ret", PERF_TYPE_RAW, 0x205, "AMD",
       "Retired x87 bottom-executing ops. Bottom-executing ops wait for all "
       "older ops to retire before executing."},
      {"fp_retired_ser_ops.sse_ctrl_ret", PERF_TYPE_RAW, 0x405, "AMD",
       "Retired SSE and AVX control word mispredict traps."},
      {"fp_retired_ser_ops.sse_bot_ret", PERF_TYPE_RAW, 0x805, "AMD",
       "Retired SSE and AVX bottom-executing ops. Bottom-executing ops wait "
       "for all older ops to retire before executing."},
      {"fp_retired_ser_ops.all", PERF_TYPE_RAW, 0xf05, "AMD",
       "Retired SSE and AVX serializing ops of all types."},
      {"fp_ops_retired_by_width.x87_uops_retired", PERF_TYPE_RAW, 0x108, "AMD",
       "Retired x87 floating-point ops."},
      {"fp_ops_retired_by_width.mmx_uops_retired", PERF_TYPE_RAW, 0x208, "AMD",
       "Retired MMX floating-point ops."},
      {"fp_ops_retired_by_width.scalar_uops_retired", PERF_TYPE_RAW, 0x408,
       "AMD", "Retired scalar floating-point ops."},
      {"fp_ops_retired_by_width.pack_128_uops_retired", PERF_TYPE_RAW, 0x808,
       "AMD", "Retired packed 128-bit floating-point ops."},
      {"fp_ops_retired_by_width.pack_256_uops_retired", PERF_TYPE_RAW, 0x1008,
       "AMD", "Retired packed 256-bit floating-point ops."},
      {"fp_ops_retired_by_width.pack_512_uops_retired", PERF_TYPE_RAW, 0x2008,
       "AMD", "Retired packed 512-bit floating-point ops."},
      {"fp_ops_retired_by_width.all", PERF_TYPE_RAW, 0x3f08, "AMD",
       "Retired floating-point ops of all widths."},
      {"fp_ops_retired_by_type.scalar_add", PERF_TYPE_RAW, 0x10a, "AMD",
       "Retired scalar floating-point add ops."},
      {"fp_ops_retired_by_type.scalar_sub", PERF_TYPE_RAW, 0x20a, "AMD",
       "Retired scalar floating-point subtract ops."},
      {"fp_ops_retired_by_type.scalar_mul", PERF_TYPE_RAW, 0x30a, "AMD",
       "Retired scalar floating-point multiply ops."},
      {"fp_ops_retired_by_type.scalar_mac", PERF_TYPE_RAW, 0x40a, "AMD",
       "Retired scalar floating-point multiply-accumulate ops."},
      {"fp_ops_retired_by_type.scalar_div", PERF_TYPE_RAW, 0x50a, "AMD",
       "Retired scalar floating-point divide ops."},
      {"fp_ops_retired_by_type.scalar_sqrt", PERF_TYPE_RAW, 0x60a, "AMD",
       "Retired scalar floating-point square root ops."},
      {"fp_ops_retired_by_type.scalar_cmp", PERF_TYPE_RAW, 0x70a, "AMD",
       "Retired scalar floating-point compare ops."},
      {"fp_ops_retired_by_type.scalar_cvt", PERF_TYPE_RAW, 0x80a, "AMD",
       "Retired scalar floating-point convert ops."},
      {"fp_ops_retired_by_type.scalar_blend", PERF_TYPE_RAW, 0x90a, "AMD",
       "Retired scalar floating-point blend ops."},
      {"fp_ops_retired_by_type.scalar_other", PERF_TYPE_RAW, 0xe0a, "AMD",
       "Retired scalar floating-point ops of other types."},
      {"fp_ops_retired_by_type.scalar_all", PERF_TYPE_RAW, 0xf0a, "AMD",
       "Retired scalar floating-point ops of all types."},
      {"fp_ops_retired_by_type.vector_add", PERF_TYPE_RAW, 0x100a, "AMD",
       "Retired vector floating-point add ops."},
      {"fp_ops_retired_by_type.vector_sub", PERF_TYPE_RAW, 0x200a, "AMD",
       "Retired vector floating-point subtract ops."},
      {"fp_ops_retired_by_type.vector_mul", PERF_TYPE_RAW, 0x300a, "AMD",
       "Retired vector floating-point multiply ops."},
      {"fp_ops_retired_by_type.vector_mac", PERF_TYPE_RAW, 0x400a, "AMD",
       "Retired vector floating-point multiply-accumulate ops."},
      {"fp_ops_retired_by_type.vector_div", PERF_TYPE_RAW, 0x500a, "AMD",
       "Retired vector floating-point divide ops."},
      {"fp_ops_retired_by_type.vector_sqrt", PERF_TYPE_RAW, 0x600a, "AMD",
       "Retired vector floating-point square root ops."},
      {"fp_ops_retired_by_type.vector_cmp", PERF_TYPE_RAW, 0x700a, "AMD",
       "Retired vector floating-point compare ops."},
      {"fp_ops_retired_by_type.vector_cvt", PERF_TYPE_RAW, 0x800a, "AMD",
       "Retired vector floating-point convert ops."},
      {"fp_ops_retired_by_type.vector_blend", PERF_TYPE_RAW, 0x900a, "AMD",
       "Retired vector floating-point blend ops."},
      {"fp_ops_retired_by_type.vector_shuffle", PERF_TYPE_RAW, 0xb00a, "AMD",
       "Retired vector floating-point shuffle ops (may include instructions "
       "not necessarily thought of as including shuffles e.g. horizontal add, "
       "dot "
       "product, and certain MOV instructions)."},
      {"fp_ops_retired_by_type.vector_logical", PERF_TYPE_RAW, 0xd00a, "AMD",
       "Retired vector floating-point logical ops."},
      {"fp_ops_retired_by_type.vector_other", PERF_TYPE_RAW, 0xe00a, "AMD",
       "Retired vector floating-point ops of other types."},
      {"fp_ops_retired_by_type.vector_all", PERF_TYPE_RAW, 0xf00a, "AMD",
       "Retired vector floating-point ops of all types."},
      {"fp_ops_retired_by_type.all", PERF_TYPE_RAW, 0xff0a, "AMD",
       "Retired floating-point ops of all types."},
      {"sse_avx_ops_retired.mmx_add", PERF_TYPE_RAW, 0x10b, "AMD",
       "Retired MMX integer add."},
      {"sse_avx_ops_retired.mmx_sub", PERF_TYPE_RAW, 0x20b, "AMD",
       "Retired MMX integer subtract ops."},
      {"sse_avx_ops_retired.mmx_mul", PERF_TYPE_RAW, 0x30b, "AMD",
       "Retired MMX integer multiply ops."},
      {"sse_avx_ops_retired.mmx_mac", PERF_TYPE_RAW, 0x40b, "AMD",
       "Retired MMX integer multiply-accumulate ops."},
      {"sse_avx_ops_retired.mmx_cmp", PERF_TYPE_RAW, 0x70b, "AMD",
       "Retired MMX integer compare ops."},
      {"sse_avx_ops_retired.mmx_shift", PERF_TYPE_RAW, 0x90b, "AMD",
       "Retired MMX integer shift ops."},
      {"sse_avx_ops_retired.mmx_mov", PERF_TYPE_RAW, 0xa0b, "AMD",
       "Retired MMX integer MOV ops."},
      {"sse_avx_ops_retired.mmx_shuffle", PERF_TYPE_RAW, 0xb0b, "AMD",
       "Retired MMX integer shuffle ops (may include instructions not "
       "necessarily thought of as including shuffles e.g. horizontal add, dot "
       "product, and certain MOV instructions)."},
      {"sse_avx_ops_retired.mmx_pack", PERF_TYPE_RAW, 0xc0b, "AMD",
       "Retired MMX integer pack ops."},
      {"sse_avx_ops_retired.mmx_logical", PERF_TYPE_RAW, 0xd0b, "AMD",
       "Retired MMX integer logical ops."},
      {"sse_avx_ops_retired.mmx_other", PERF_TYPE_RAW, 0xe0b, "AMD",
       "Retired MMX integer multiply ops of other types."},
      {"sse_avx_ops_retired.mmx_all", PERF_TYPE_RAW, 0xf0b, "AMD",
       "Retired MMX integer ops of all types."},
      {"sse_avx_ops_retired.sse_avx_add", PERF_TYPE_RAW, 0x100b, "AMD",
       "Retired SSE and AVX integer add ops."},
      {"sse_avx_ops_retired.sse_avx_sub", PERF_TYPE_RAW, 0x200b, "AMD",
       "Retired SSE and AVX integer subtract ops."},
      {"sse_avx_ops_retired.sse_avx_mul", PERF_TYPE_RAW, 0x300b, "AMD",
       "Retired SSE and AVX integer multiply ops."},
      {"sse_avx_ops_retired.sse_avx_mac", PERF_TYPE_RAW, 0x400b, "AMD",
       "Retired SSE and AVX integer multiply-accumulate ops."},
      {"sse_avx_ops_retired.sse_avx_aes", PERF_TYPE_RAW, 0x500b, "AMD",
       "Retired SSE and AVX integer AES ops."},
      {"sse_avx_ops_retired.sse_avx_sha", PERF_TYPE_RAW, 0x600b, "AMD",
       "Retired SSE and AVX integer SHA ops."},
      {"sse_avx_ops_retired.sse_avx_cmp", PERF_TYPE_RAW, 0x700b, "AMD",
       "Retired SSE and AVX integer compare ops."},
      {"sse_avx_ops_retired.sse_avx_clm", PERF_TYPE_RAW, 0x800b, "AMD",
       "Retired SSE and AVX integer CLM ops."},
      {"sse_avx_ops_retired.sse_avx_shift", PERF_TYPE_RAW, 0x900b, "AMD",
       "Retired SSE and AVX integer shift ops."},
      {"sse_avx_ops_retired.sse_avx_mov", PERF_TYPE_RAW, 0xa00b, "AMD",
       "Retired SSE and AVX integer MOV ops."},
      {"sse_avx_ops_retired.sse_avx_shuffle", PERF_TYPE_RAW, 0xb00b, "AMD",
       "Retired SSE and AVX integer shuffle ops (may include instructions not "
       "necessarily thought of as including shuffles e.g. horizontal add, dot "
       "product, and certain MOV instructions)."},
      {"sse_avx_ops_retired.sse_avx_pack", PERF_TYPE_RAW, 0xc00b, "AMD",
       "Retired SSE and AVX integer pack ops."},
      {"sse_avx_ops_retired.sse_avx_logical", PERF_TYPE_RAW, 0xd00b, "AMD",
       "Retired SSE and AVX integer logical ops."},
      {"sse_avx_ops_retired.sse_avx_other", PERF_TYPE_RAW, 0xe00b, "AMD",
       "Retired SSE and AVX integer ops of other types."},
      {"sse_avx_ops_retired.sse_avx_all", PERF_TYPE_RAW, 0xf00b, "AMD",
       "Retired SSE and AVX integer ops of all types."},
      {"sse_avx_ops_retired.all", PERF_TYPE_RAW, 0xff0b, "AMD",
       "Retired SSE, AVX and MMX integer ops of all types."},
      {"fp_pack_ops_retired.fp128_add", PERF_TYPE_RAW, 0x10c, "AMD",
       "Retired 128-bit packed floating-point add ops."},
      {"fp_pack_ops_retired.fp128_sub", PERF_TYPE_RAW, 0x20c, "AMD",
       "Retired 128-bit packed floating-point subtract ops."},
      {"fp_pack_ops_retired.fp128_mul", PERF_TYPE_RAW, 0x30c, "AMD",
       "Retired 128-bit packed floating-point multiply ops."},
      {"fp_pack_ops_retired.fp128_mac", PERF_TYPE_RAW, 0x40c, "AMD",
       "Retired 128-bit packed floating-point multiply-accumulate ops."},
      {"fp_pack_ops_retired.fp128_div", PERF_TYPE_RAW, 0x50c, "AMD",
       "Retired 128-bit packed floating-point divide ops."},
      {"fp_pack_ops_retired.fp128_sqrt", PERF_TYPE_RAW, 0x60c, "AMD",
       "Retired 128-bit packed floating-point square root ops."},
      {"fp_pack_ops_retired.fp128_cmp", PERF_TYPE_RAW, 0x70c, "AMD",
       "Retired 128-bit packed floating-point compare ops."},
      {"fp_pack_ops_retired.fp128_cvt", PERF_TYPE_RAW, 0x80c, "AMD",
       "Retired 128-bit packed floating-point convert ops."},
      {"fp_pack_ops_retired.fp128_blend", PERF_TYPE_RAW, 0x90c, "AMD",
       "Retired 128-bit packed floating-point blend ops."},
      {"fp_pack_ops_retired.fp128_shuffle", PERF_TYPE_RAW, 0xb0c, "AMD",
       "Retired 128-bit packed floating-point shuffle ops (may include "
       "instructions not necessarily thought of as including shuffles e.g. "
       "horizontal add, dot product, and certain MOV instructions)."},
      {"fp_pack_ops_retired.fp128_logical", PERF_TYPE_RAW, 0xd0c, "AMD",
       "Retired 128-bit packed floating-point logical ops."},
      {"fp_pack_ops_retired.fp128_other", PERF_TYPE_RAW, 0xe0c, "AMD",
       "Retired 128-bit packed floating-point ops of other types."},
      {"fp_pack_ops_retired.fp128_all", PERF_TYPE_RAW, 0xf0c, "AMD",
       "Retired 128-bit packed floating-point ops of all types."},
      {"fp_pack_ops_retired.fp256_add", PERF_TYPE_RAW, 0x100c, "AMD",
       "Retired 256-bit packed floating-point add ops."},
      {"fp_pack_ops_retired.fp256_sub", PERF_TYPE_RAW, 0x200c, "AMD",
       "Retired 256-bit packed floating-point subtract ops."},
      {"fp_pack_ops_retired.fp256_mul", PERF_TYPE_RAW, 0x300c, "AMD",
       "Retired 256-bit packed floating-point multiply ops."},
      {"fp_pack_ops_retired.fp256_mac", PERF_TYPE_RAW, 0x400c, "AMD",
       "Retired 256-bit packed floating-point multiply-accumulate ops."},
      {"fp_pack_ops_retired.fp256_div", PERF_TYPE_RAW, 0x500c, "AMD",
       "Retired 256-bit packed floating-point divide ops."},
      {"fp_pack_ops_retired.fp256_sqrt", PERF_TYPE_RAW, 0x600c, "AMD",
       "Retired 256-bit packed floating-point square root ops."},
      {"fp_pack_ops_retired.fp256_cmp", PERF_TYPE_RAW, 0x700c, "AMD",
       "Retired 256-bit packed floating-point compare ops."},
      {"fp_pack_ops_retired.fp256_cvt", PERF_TYPE_RAW, 0x800c, "AMD",
       "Retired 256-bit packed floating-point convert ops."},
      {"fp_pack_ops_retired.fp256_blend", PERF_TYPE_RAW, 0x900c, "AMD",
       "Retired 256-bit packed floating-point blend ops."},
      {"fp_pack_ops_retired.fp256_shuffle", PERF_TYPE_RAW, 0xb00c, "AMD",
       "Retired 256-bit packed floating-point shuffle ops (may include "
       "instructions not necessarily thought of as including shuffles e.g. "
       "horizontal add, dot product, and certain MOV instructions)."},
      {"fp_pack_ops_retired.fp256_logical", PERF_TYPE_RAW, 0xd00c, "AMD",
       "Retired 256-bit packed floating-point logical ops."},
      {"fp_pack_ops_retired.fp256_other", PERF_TYPE_RAW, 0xe00c, "AMD",
       "Retired 256-bit packed floating-point ops of other types."},
      {"fp_pack_ops_retired.fp256_all", PERF_TYPE_RAW, 0xf00c, "AMD",
       "Retired 256-bit packed floating-point ops of all types."},
      {"fp_pack_ops_retired.all", PERF_TYPE_RAW, 0xff0c, "AMD",
       "Retired packed floating-point ops of all types."},
      {"packed_int_op_type.int128_add", PERF_TYPE_RAW, 0x10d, "AMD",
       "Retired 128-bit packed integer add ops."},
      {"packed_int_op_type.int128_sub", PERF_TYPE_RAW, 0x20d, "AMD",
       "Retired 128-bit packed integer subtract ops."},
      {"packed_int_op_type.int128_mul", PERF_TYPE_RAW, 0x30d, "AMD",
       "Retired 128-bit packed integer multiply ops."},
      {"packed_int_op_type.int128_mac", PERF_TYPE_RAW, 0x40d, "AMD",
       "Retired 128-bit packed integer multiply-accumulate ops."},
      {"packed_int_op_type.int128_aes", PERF_TYPE_RAW, 0x50d, "AMD",
       "Retired 128-bit packed integer AES ops."},
      {"packed_int_op_type.int128_sha", PERF_TYPE_RAW, 0x60d, "AMD",
       "Retired 128-bit packed integer SHA ops."},
      {"packed_int_op_type.int128_cmp", PERF_TYPE_RAW, 0x70d, "AMD",
       "Retired 128-bit packed integer compare ops."},
      {"packed_int_op_type.int128_clm", PERF_TYPE_RAW, 0x80d, "AMD",
       "Retired 128-bit packed integer CLM ops."},
      {"packed_int_op_type.int128_shift", PERF_TYPE_RAW, 0x90d, "AMD",
       "Retired 128-bit packed integer shift ops."},
      {"packed_int_op_type.int128_mov", PERF_TYPE_RAW, 0xa0d, "AMD",
       "Retired 128-bit packed integer MOV ops."},
      {"packed_int_op_type.int128_shuffle", PERF_TYPE_RAW, 0xb0d, "AMD",
       "Retired 128-bit packed integer shuffle ops (may include instructions "
       "not necessarily thought of as including shuffles e.g. horizontal add, "
       "dot "
       "product, and certain MOV instructions)."},
      {"packed_int_op_type.int128_pack", PERF_TYPE_RAW, 0xc0d, "AMD",
       "Retired 128-bit packed integer pack ops."},
      {"packed_int_op_type.int128_logical", PERF_TYPE_RAW, 0xd0d, "AMD",
       "Retired 128-bit packed integer logical ops."},
      {"packed_int_op_type.int128_other", PERF_TYPE_RAW, 0xe0d, "AMD",
       "Retired 128-bit packed integer ops of other types."},
      {"packed_int_op_type.int128_all", PERF_TYPE_RAW, 0xf0d, "AMD",
       "Retired 128-bit packed integer ops of all types."},
      {"packed_int_op_type.int256_add", PERF_TYPE_RAW, 0x100d, "AMD",
       "Retired 256-bit packed integer add ops."},
      {"packed_int_op_type.int256_sub", PERF_TYPE_RAW, 0x200d, "AMD",
       "Retired 256-bit packed integer subtract ops."},
      {"packed_int_op_type.int256_mul", PERF_TYPE_RAW, 0x300d, "AMD",
       "Retired 256-bit packed integer multiply ops."},
      {"packed_int_op_type.int256_mac", PERF_TYPE_RAW, 0x400d, "AMD",
       "Retired 256-bit packed integer multiply-accumulate ops."},
      {"packed_int_op_type.int256_cmp", PERF_TYPE_RAW, 0x700d, "AMD",
       "Retired 256-bit packed integer compare ops."},
      {"packed_int_op_type.int256_shift", PERF_TYPE_RAW, 0x900d, "AMD",
       "Retired 256-bit packed integer shift ops."},
      {"packed_int_op_type.int256_mov", PERF_TYPE_RAW, 0xa00d, "AMD",
       "Retired 256-bit packed integer MOV ops."},
      {"packed_int_op_type.int256_shuffle", PERF_TYPE_RAW, 0xb00d, "AMD",
       "Retired 256-bit packed integer shuffle ops (may include instructions "
       "not necessarily thought of as including shuffles e.g. horizontal add, "
       "dot "
       "product, and certain MOV instructions)."},
      {"packed_int_op_type.int256_pack", PERF_TYPE_RAW, 0xc00d, "AMD",
       "Retired 256-bit packed integer pack ops."},
      {"packed_int_op_type.int256_logical", PERF_TYPE_RAW, 0xd00d, "AMD",
       "Retired 256-bit packed integer logical ops."},
      {"packed_int_op_type.int256_other", PERF_TYPE_RAW, 0xe00d, "AMD",
       "Retired 256-bit packed integer ops of other types."},
      {"packed_int_op_type.int256_all", PERF_TYPE_RAW, 0xf00d, "AMD",
       "Retired 256-bit packed integer ops of all types."},
      {"packed_int_op_type.all", PERF_TYPE_RAW, 0xff0d, "AMD",
       "Retired packed integer ops of all types."},
      {"fp_disp_faults.x87_fill_fault", PERF_TYPE_RAW, 0x10e, "AMD",
       "Floating-point dispatch faults for x87 fills."},
      {"fp_disp_faults.xmm_fill_fault", PERF_TYPE_RAW, 0x20e, "AMD",
       "Floating-point dispatch faults for XMM fills."},
      {"fp_disp_faults.ymm_fill_fault", PERF_TYPE_RAW, 0x40e, "AMD",
       "Floating-point dispatch faults for YMM fills."},
      {"fp_disp_faults.ymm_spill_fault", PERF_TYPE_RAW, 0x80e, "AMD",
       "Floating-point dispatch faults for YMM spills."},
      {"fp_disp_faults.sse_avx_all", PERF_TYPE_RAW, 0xe0e, "AMD",
       "Floating-point dispatch faults of all types for SSE and AVX ops."},
      {"fp_disp_faults.all", PERF_TYPE_RAW, 0xf0e, "AMD",
       "Floating-point dispatch faults of all types."},
  };
  return sAllPerfEventTypes;
}
