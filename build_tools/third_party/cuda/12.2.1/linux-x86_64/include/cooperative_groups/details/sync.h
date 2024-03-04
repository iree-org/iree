 /* Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */

#ifndef _CG_GRID_H
#define _CG_GRID_H

#include "info.h"

_CG_BEGIN_NAMESPACE

namespace details
{

typedef unsigned int barrier_t;

_CG_STATIC_QUALIFIER bool bar_has_flipped(unsigned int old_arrive, unsigned int current_arrive) {
    return (((old_arrive ^ current_arrive) & 0x80000000) != 0);
}

_CG_STATIC_QUALIFIER bool is_cta_master() {
    return (threadIdx.x + threadIdx.y + threadIdx.z == 0);
}

_CG_STATIC_QUALIFIER void sync_grids(volatile barrier_t *arrived) {
    unsigned int oldArrive = 0;

    __barrier_sync(0);

    if (is_cta_master()) {
        unsigned int expected = gridDim.x * gridDim.y * gridDim.z;
        bool gpu_master = (blockIdx.x + blockIdx.y + blockIdx.z == 0);
        unsigned int nb = 1;

        if (gpu_master) {
            nb = 0x80000000 - (expected - 1);
        }

#if __CUDA_ARCH__ < 700
        // Fence; barrier update; volatile polling; fence
        __threadfence();

        oldArrive = atomicAdd((unsigned int*)arrived, nb);

        while (!bar_has_flipped(oldArrive, *arrived));

        __threadfence();
#else
        // Barrier update with release; polling with acquire
        asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;" : "=r"(oldArrive) : _CG_ASM_PTR_CONSTRAINT((unsigned int*)arrived), "r"(nb) : "memory");

        unsigned int current_arrive;
        do {
            asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(current_arrive) : _CG_ASM_PTR_CONSTRAINT((unsigned int *)arrived) : "memory");
        } while (!bar_has_flipped(oldArrive, current_arrive));
#endif
    }

    __barrier_sync(0);
}

_CG_STATIC_QUALIFIER unsigned int sync_grids_arrive(volatile barrier_t *arrived) {
    unsigned int oldArrive = 0;

    __barrier_sync(0);

    if (is_cta_master()) {
        unsigned int expected = gridDim.x * gridDim.y * gridDim.z;
        bool gpu_master = (blockIdx.x + blockIdx.y + blockIdx.z == 0);
        unsigned int nb = 1;

        if (gpu_master) {
            nb = 0x80000000 - (expected - 1);
        }

#if __CUDA_ARCH__ < 700
        // Fence; barrier update; volatile polling; fence
        __threadfence();

        oldArrive = atomicAdd((unsigned int*)arrived, nb);
#else
        // Barrier update with release; polling with acquire
        asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;" : "=r"(oldArrive) : _CG_ASM_PTR_CONSTRAINT((unsigned int*)arrived), "r"(nb) : "memory");
#endif
    }

    return oldArrive;
}


_CG_STATIC_QUALIFIER void sync_grids_wait(unsigned int oldArrive, volatile barrier_t *arrived) {
    if (is_cta_master()) {
#if __CUDA_ARCH__ < 700
        while (!bar_has_flipped(oldArrive, *arrived));

        __threadfence();

#else
        unsigned int current_arrive;
        do {
            asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(current_arrive) : _CG_ASM_PTR_CONSTRAINT((unsigned int *)arrived) : "memory");
        } while (!bar_has_flipped(oldArrive, current_arrive));
#endif
    }

    __barrier_sync(0);
}

/* - Multi warp groups synchronization routines - */

// Need both acquire and release for the last warp, since it won't be able to acquire with red.and
_CG_STATIC_QUALIFIER unsigned int atom_or_acq_rel_cta(unsigned int *addr, unsigned int val) {
    unsigned int old;
#if __CUDA_ARCH__ < 700
    __threadfence_block();
    old = atomicOr(addr, val);
#else
    asm volatile("atom.or.acq_rel.cta.b32 %0,[%1],%2;" : "=r"(old) : _CG_ASM_PTR_CONSTRAINT(addr), "r"(val) : "memory");
#endif
    return old;
}

// Special case where barrier is arrived, but not waited on
_CG_STATIC_QUALIFIER void red_or_release_cta(unsigned int *addr, unsigned int val) {
#if __CUDA_ARCH__ < 700
    __threadfence_block();
    atomicOr(addr, val);
#else
    asm volatile("red.or.release.cta.b32 [%0],%1;" :: _CG_ASM_PTR_CONSTRAINT(addr), "r"(val) : "memory");
#endif
}

// Usually called by last arriving warp to released other warps, can be relaxed, since or was already acq_rel
_CG_STATIC_QUALIFIER void red_and_relaxed_cta(unsigned int *addr, unsigned int val) {
#if __CUDA_ARCH__ < 700
    atomicAnd(addr, val);
#else
    asm volatile("red.and.relaxed.cta.b32 [%0],%1;" :: _CG_ASM_PTR_CONSTRAINT(addr), "r"(val) : "memory");
#endif
}

// Special case of release, where last warp was doing extra work before releasing others, need to be release
//  to ensure that extra work is visible
_CG_STATIC_QUALIFIER void red_and_release_cta(unsigned int *addr, unsigned int val) {
#if __CUDA_ARCH__ < 700
    __threadfence_block();
    atomicAnd(addr, val);
#else
    asm volatile("red.and.release.cta.b32 [%0],%1;" :: _CG_ASM_PTR_CONSTRAINT(addr), "r"(val) : "memory");
#endif
}

// Read the barrier, acquire to ensure all memory operations following the sync are correctly performed after it is released
_CG_STATIC_QUALIFIER unsigned int ld_acquire_cta(unsigned int *addr) {
    unsigned int val;
#if __CUDA_ARCH__ < 700
    val = *((volatile unsigned int*) addr);
    __threadfence_block();
#else
    asm volatile("ld.acquire.cta.u32 %0,[%1];" : "=r"(val) : _CG_ASM_PTR_CONSTRAINT(addr) : "memory");
#endif
    return val;
}

// Get synchronization bit mask of my thread_block_tile of size num_warps. Thread ranks 0..31 have the first bit assigned to them,
// thread ranks 32..63 second etc 
// Bit masks are unique for each group, groups of the same size will have the same number of bits set, but on different positions 
_CG_STATIC_QUALIFIER unsigned int get_group_mask(unsigned int thread_rank, unsigned int num_warps) {
    return num_warps == 32 ? ~0 : ((1 << num_warps) - 1) << (num_warps * (thread_rank / (num_warps * 32)));
}

_CG_STATIC_QUALIFIER void barrier_wait(barrier_t *arrived, unsigned int warp_bit) {
    while(ld_acquire_cta(arrived) & warp_bit);
}

// Default blocking sync.
_CG_STATIC_QUALIFIER void sync_warps(barrier_t *arrived, unsigned int thread_rank, unsigned int num_warps) {
    unsigned int warp_id = thread_rank / 32;
    bool warp_master = (thread_rank % 32 == 0);
    unsigned int warp_bit = 1 << warp_id;
    unsigned int group_mask = get_group_mask(thread_rank, num_warps);

    __syncwarp(0xFFFFFFFF);

    if (warp_master) {
        unsigned int old = atom_or_acq_rel_cta(arrived, warp_bit);
        if (((old | warp_bit) & group_mask) == group_mask) {
            red_and_relaxed_cta(arrived, ~group_mask);
        }
        else {
            barrier_wait(arrived, warp_bit);
        }
    }

    __syncwarp(0xFFFFFFFF);
}

// Blocking sync, except the last arriving warp, that releases other warps, returns to do other stuff first.
// Warp returning true from this function needs to call sync_warps_release.
_CG_STATIC_QUALIFIER bool sync_warps_last_releases(barrier_t *arrived, unsigned int thread_rank, unsigned int num_warps) {
    unsigned int warp_id = thread_rank / 32;
    bool warp_master = (thread_rank % 32 == 0);
    unsigned int warp_bit = 1 << warp_id;
    unsigned int group_mask = get_group_mask(thread_rank, num_warps);

    __syncwarp(0xFFFFFFFF);

    unsigned int old = 0;
    if (warp_master) {
        old = atom_or_acq_rel_cta(arrived, warp_bit);
    }
    old = __shfl_sync(0xFFFFFFFF, old, 0);
    if (((old | warp_bit) & group_mask) == group_mask) {
        return true;
    }
    barrier_wait(arrived, warp_bit);

    return false;
}

// Release my group from the barrier.
_CG_STATIC_QUALIFIER void sync_warps_release(barrier_t *arrived, bool is_master, unsigned int thread_rank, unsigned int num_warps) {
    unsigned int group_mask = get_group_mask(thread_rank, num_warps);
    if (is_master) {
        red_and_release_cta(arrived, ~group_mask);
    }
}

// Arrive at my group barrier, but don't block or release the barrier, even if every one arrives.
// sync_warps_release needs to be called by some warp after this one to reset the barrier.
_CG_STATIC_QUALIFIER void sync_warps_arrive(barrier_t *arrived, unsigned int thread_rank, unsigned int num_warps) {
    unsigned int warp_id = thread_rank / 32;
    bool warp_master = (thread_rank % 32 == 0);
    unsigned int warp_bit = 1 << warp_id;
    unsigned int group_mask = get_group_mask(thread_rank, num_warps);

    __syncwarp(0xFFFFFFFF);

    if (warp_master) {
        red_or_release_cta(arrived, warp_bit);
    }
}

// Wait for my warp to be released from the barrier. Warp must have arrived first.
_CG_STATIC_QUALIFIER void sync_warps_wait(barrier_t *arrived, unsigned int thread_rank) {
    unsigned int warp_id = thread_rank / 32;
    unsigned int warp_bit = 1 << warp_id;

    barrier_wait(arrived, warp_bit);
}

// Wait for specific warp to arrive at the barrier
_CG_QUALIFIER void sync_warps_wait_for_specific_warp(barrier_t *arrived, unsigned int wait_warp_id) {
    unsigned int wait_mask = 1 << wait_warp_id;
    while((ld_acquire_cta(arrived) & wait_mask) != wait_mask);
}

// Initialize the bit corresponding to my warp in the barrier
_CG_QUALIFIER void sync_warps_reset(barrier_t *arrived, unsigned int thread_rank) {
    unsigned int warp_id = thread_rank / 32;
    unsigned int warp_bit = 1 << warp_id;

    __syncwarp(0xFFFFFFFF);

    if (thread_rank % 32 == 0) {
        red_and_release_cta(arrived, ~warp_bit);
    }
    // No need to sync after the atomic, there will be a sync of the group that is being partitioned right after this.
}

} // details

_CG_END_NAMESPACE

#endif // _CG_GRID_H
