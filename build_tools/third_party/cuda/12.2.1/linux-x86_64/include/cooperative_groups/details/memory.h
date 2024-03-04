/* Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
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

#ifndef _COOPERATIVE_GROUPS_MEMORY_H_
# define _COOPERATIVE_GROUPS_MEMORY_H_

#include "info.h"

_CG_BEGIN_NAMESPACE

#if defined(_CG_CPP11_FEATURES)
namespace details {
    _CG_STATIC_CONST_DECL int scratch_num_reserved_bytes = 12;

#if defined(_CG_HAS_RESERVED_SHARED)
    _CG_STATIC_QUALIFIER void* reserved_shared_ptr()
    {
        void *ptr;
        asm ("{\n\t"
             " .reg .u32 start;\n\t"
             " .reg .u64 extended;\n\t"
             " mov.u32 start, %%reserved_smem_offset_1;\n\t"
             " cvt.u64.u32 extended, start;\n\t"
             " cvta.shared.u64 %0, extended;\n\t"
             "}"
             : "=" _CG_ASM_PTR_CONSTRAINT(ptr));
        return ptr;
    }
#endif

    struct multi_warp_scratch {
        // One barrier per possible size of the group.
        _CG_STATIC_CONST_DECL unsigned int memory_barriers_count = 5;
        _CG_STATIC_CONST_DECL size_t sync_memory_size = memory_barriers_count * sizeof(barrier_t);

        using communication_type = unsigned long long;
        _CG_STATIC_CONST_DECL size_t communication_size = sizeof(communication_type);

        // Layout of the scratch space:
        barrier_t barriers[memory_barriers_count];
        char reserved[scratch_num_reserved_bytes]; // Reserve 12 bytes for future use
        communication_type communication_memory[default_max_block_size / 32];

        _CG_STATIC_CONSTEXPR_QUALIFIER unsigned int scratch_size_needed(unsigned int max_block_size) {
            // One slot of collectives memory per warp.
            return scratch_num_reserved_bytes + sync_memory_size + max_block_size / 32 * communication_size;
        }

        _CG_QUALIFIER void init_barriers(unsigned int thread_rank) {
            if (thread_rank < memory_barriers_count) {
                barriers[thread_rank] = 0;
            }
        }
    };

#if defined(_CG_HAS_RESERVED_SHARED)
    // CG can expect at least 288 bytes available in reserved shared
    static_assert(sizeof(multi_warp_scratch) <= 288, "multi-warp scratch size is too large");
#endif

    // Make sure the structure can fit into the user provided memory
    static_assert(sizeof(multi_warp_scratch) <= multi_warp_scratch::scratch_size_needed(default_max_block_size),
                  "multi-warp scratch size is too large");


    _CG_QUALIFIER multi_warp_scratch* get_scratch_ptr(void* user_scratch) {
        void *ptr;
#if defined(_CG_HAS_RESERVED_SHARED)
        ptr = reserved_shared_ptr();
#else
        ptr = user_scratch;
#endif
        return static_cast<multi_warp_scratch*>(ptr);

    }

}

template <unsigned int MaxBlockSize = details::default_max_block_size>
struct __align__(details::multi_warp_scratch::communication_size) block_tile_memory {
private:
#if !defined(_CG_HAS_RESERVED_SHARED)
    char scratch[details::multi_warp_scratch::scratch_size_needed(MaxBlockSize)];
#endif
};
#endif

_CG_END_NAMESPACE

#endif /* !_COOPERATIVE_GROUPS_MEMORY_H_ */
