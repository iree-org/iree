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

#ifndef _CG_ASYNC_H
#define _CG_ASYNC_H

#include "helpers.h"
#include "info.h"

#include <cuda_pipeline.h>

_CG_BEGIN_NAMESPACE

namespace details {
// Groups supported by memcpy_async
template <class TyGroup>
struct _async_copy_group_supported : public _CG_STL_NAMESPACE::false_type {};

template <unsigned int Sz, typename TyPar>
struct _async_copy_group_supported<cooperative_groups::thread_block_tile<Sz, TyPar>>
    : public _CG_STL_NAMESPACE::true_type {};
template <>
struct _async_copy_group_supported<cooperative_groups::coalesced_group> : public _CG_STL_NAMESPACE::true_type {};
template <>
struct _async_copy_group_supported<cooperative_groups::thread_block> : public _CG_STL_NAMESPACE::true_type {};

template <class TyGroup>
using async_copy_group_supported = _async_copy_group_supported<details::remove_qual<TyGroup>>;

// Groups that require optimization
template <class TyGroup>
struct _async_copy_optimize_tile : public _CG_STL_NAMESPACE::false_type {};

template <typename TyPar>
struct _async_copy_optimize_tile<cooperative_groups::thread_block_tile<1, TyPar>>
    : public _CG_STL_NAMESPACE::false_type {};

template <unsigned int Sz, typename TyPar>
struct _async_copy_optimize_tile<cooperative_groups::thread_block_tile<Sz, TyPar>>
    : public _CG_STL_NAMESPACE::true_type {};

template <class TyGroup>
using async_copy_optimize_tile = _async_copy_optimize_tile<details::remove_qual<TyGroup>>;

// SFINAE helpers for tile optimizations
template <class TyGroup>
using enable_tile_optimization =
    typename _CG_STL_NAMESPACE::enable_if<async_copy_optimize_tile<TyGroup>::value, void *>::type;

template <class TyGroup>
using disable_tile_optimization =
    typename _CG_STL_NAMESPACE::enable_if<!async_copy_optimize_tile<TyGroup>::value, void *>::type;

// Segment for punning to aligned types
template <unsigned int N>
struct _Segment {
    int _seg[N];
};

// Trivial layout guaranteed-aligned copy-async compatible segments
template <unsigned int N>
struct Segment;
template <>
struct __align__(4) Segment<1> : public _Segment<1>{};
template <>
struct __align__(8) Segment<2> : public _Segment<2>{};
template <>
struct __align__(16) Segment<4> : public _Segment<4>{};

// Interleaved element by element copies from source to dest
template <typename TyGroup, typename TyElem>
_CG_STATIC_QUALIFIER void inline_copy(TyGroup &group, TyElem *__restrict__ dst, const TyElem *__restrict__ src,
                                      size_t count) {
    const unsigned int rank = group.thread_rank();
    const unsigned int stride = group.size();

    for (size_t idx = rank; idx < count; idx += stride) {
        dst[idx] = src[idx];
    }
}

template <typename TyGroup, typename TyElem, enable_tile_optimization<TyGroup> = nullptr>
_CG_STATIC_QUALIFIER void accelerated_async_copy(TyGroup &group, TyElem *__restrict__ dst,
                                                 const TyElem *__restrict__ src, size_t count) {
    static_assert(async_copy_group_supported<TyGroup>::value,
                  "Async copy is only supported for groups that represent private shared memory");

    if (count == 0) {
        return;
    }

    const bool dstIsNotShared = !__isShared(dst);
    const bool srcIsNotGlobal = !__isGlobal(src);

    if (dstIsNotShared || srcIsNotGlobal) {
        inline_copy(group, dst, src, count);
        return;
    }

    const unsigned int stride = group.size();
    const unsigned int rank = group.thread_rank();
    // Efficient copies require warps to operate on the same amount of work at each step.
    // remainders are handled in a separate stage to prevent branching
    const unsigned int subWarpMask = (stride - 1);
    const unsigned int subwarpCopies = (subWarpMask & (unsigned int)count);
    const unsigned int maxSubwarpRank = min(rank, subwarpCopies - 1);

    const size_t warpCopies = (count & (~subWarpMask));

    for (size_t idx = 0; idx < warpCopies; idx += stride) {
        size_t _srcIdx = rank + idx;
        size_t _dstIdx = rank + idx;
        __pipeline_memcpy_async(dst + _dstIdx, src + _srcIdx, sizeof(TyElem));
    }

    if (subwarpCopies) {
        size_t _srcIdx = warpCopies + maxSubwarpRank;
        size_t _dstIdx = warpCopies + maxSubwarpRank;
        __pipeline_memcpy_async(dst + _dstIdx, src + _srcIdx, sizeof(TyElem));
    }
}

template <typename TyGroup, typename TyElem, disable_tile_optimization<TyGroup> = nullptr>
_CG_STATIC_QUALIFIER void accelerated_async_copy(TyGroup &group, TyElem *__restrict__ dst,
                                                 const TyElem *__restrict__ src, size_t count) {
    static_assert(async_copy_group_supported<TyGroup>::value,
                  "Async copy is only supported for groups that represent private shared memory");

    const bool dstIsNotShared = !__isShared(dst);
    const bool srcIsNotGlobal = !__isGlobal(src);

    if (dstIsNotShared || srcIsNotGlobal) {
        inline_copy(group, dst, src, count);
        return;
    }

    unsigned int stride = group.size();
    unsigned int rank = group.thread_rank();

    for (size_t idx = rank; idx < count; idx += stride) {
        size_t _srcIdx = idx;
        size_t _dstIdx = idx;
        __pipeline_memcpy_async(dst + _dstIdx, src + _srcIdx, sizeof(TyElem));
    }
}

// Determine best possible alignment given an input and initial conditions
// Attempts to generate as little code as possible, most likely should only be used with 1 and 2 byte alignments
template <unsigned int MinAlignment, unsigned int MaxAlignment>
_CG_STATIC_QUALIFIER uint32_t find_best_alignment(void *__restrict__ dst, const void *__restrict__ src) {
    // Narrowing conversion intentional
    uint32_t base1 = (uint32_t) reinterpret_cast<uintptr_t>(src);
    uint32_t base2 = (uint32_t) reinterpret_cast<uintptr_t>(dst);

    uint32_t diff = ((base1) ^ (base2)) & (MaxAlignment - 1);

    // range [MaxAlignment, alignof(elem)], step: x >> 1
    // over range of possible alignments, choose best available out of range
    uint32_t out = MaxAlignment;
#pragma unroll
    for (uint32_t alignment = (MaxAlignment >> 1); alignment >= MinAlignment; alignment >>= 1) {
        if (alignment & diff)
            out = alignment;
    }

    return out;
}

// Determine best possible alignment given an input and initial conditions
// Attempts to generate as little code as possible, most likely should only be used with 1 and 2 byte alignments
template <typename TyType, typename TyGroup>
_CG_STATIC_QUALIFIER void copy_like(const TyGroup &group, void *__restrict__ _dst, const void *__restrict__ _src,
                                    size_t count) {
    const char *src = reinterpret_cast<const char *>(_src);
    char *dst = reinterpret_cast<char *>(_dst);

    constexpr uint32_t targetAlignment = (uint32_t)alignof(TyType);

    uint32_t base = (uint32_t) reinterpret_cast<uintptr_t>(src);
    uint32_t alignOffset = ((~base) + 1) & (targetAlignment - 1);

    inline_copy(group, dst, src, alignOffset);
    count -= alignOffset;
    src += alignOffset;
    dst += alignOffset;

    // Copy using the best available alignment, async_copy expects n-datums, not bytes
    size_t asyncCount = count / sizeof(TyType);
    accelerated_async_copy(group, reinterpret_cast<TyType *>(dst), reinterpret_cast<const TyType *>(src), asyncCount);
    asyncCount *= sizeof(TyType);

    count -= asyncCount;
    src += asyncCount;
    dst += asyncCount;
    inline_copy(group, dst, src, count);
}

// We must determine alignment and manually align src/dst ourselves
template <size_t AlignHint>
struct _memcpy_async_align_dispatch {
    template <typename TyGroup>
    _CG_STATIC_QUALIFIER void copy(TyGroup &group, void *__restrict__ dst, const void *__restrict__ src, size_t count) {
        uint32_t alignment = find_best_alignment<AlignHint, 16>(dst, src);

        // Avoid copying the extra bytes if desired copy count is smaller
        alignment = count < alignment ? AlignHint : alignment;

        switch (alignment) {
        default:
        case 1:
            inline_copy(group, reinterpret_cast<char *>(dst), reinterpret_cast<const char *>(src), count);
            break;
        case 2:
            inline_copy(group, reinterpret_cast<short *>(dst), reinterpret_cast<const short *>(src), count >> 1);
            break;
        case 4:
            copy_like<Segment<1>>(group, dst, src, count);
            break;
        case 8:
            copy_like<Segment<2>>(group, dst, src, count);
            break;
        case 16:
            copy_like<Segment<4>>(group, dst, src, count);
            break;
        }
    }
};

// Specialization for 4 byte alignments
template <>
struct _memcpy_async_align_dispatch<4> {
    template <typename TyGroup>
    _CG_STATIC_QUALIFIER void copy(TyGroup &group, void *__restrict__ _dst, const void *__restrict__ _src,
                                   size_t count) {
        const Segment<1> *src = reinterpret_cast<const Segment<1> *>(_src);
        Segment<1> *dst = reinterpret_cast<Segment<1> *>(_dst);

        // Dispatch straight to aligned LDGSTS calls
        accelerated_async_copy(group, dst, src, count / sizeof(*dst));
    }
};

// Specialization for 8 byte alignments
template <>
struct _memcpy_async_align_dispatch<8> {
    template <typename TyGroup>
    _CG_STATIC_QUALIFIER void copy(TyGroup &group, void *__restrict__ _dst, const void *__restrict__ _src,
                                   size_t count) {
        const Segment<2> *src = reinterpret_cast<const Segment<2> *>(_src);
        Segment<2> *dst = reinterpret_cast<Segment<2> *>(_dst);

        // Dispatch straight to aligned LDGSTS calls
        accelerated_async_copy(group, dst, src, count / sizeof(*dst));
    }
};

// Alignments over 16 are truncated to 16 and bypass alignment
// This is the highest performing memcpy available
template <>
struct _memcpy_async_align_dispatch<16> {
    template <typename TyGroup>
    _CG_STATIC_QUALIFIER void copy(TyGroup &group, void *__restrict__ _dst, const void *__restrict__ _src,
                                   size_t count) {
        const Segment<4> *src = reinterpret_cast<const Segment<4> *>(_src);
        Segment<4> *dst = reinterpret_cast<Segment<4> *>(_dst);

        // Dispatch straight to aligned LDGSTS calls
        accelerated_async_copy(group, dst, src, count / sizeof(*dst));
    }
};

// byte-wide API
template <size_t Alignment, class TyGroup>
_CG_STATIC_QUALIFIER void _memcpy_async_dispatch_to_aligned_copy(const TyGroup &group, void *__restrict__ _dst,
                                                                 const void *__restrict__ _src, size_t count) {
    static_assert(!(Alignment & (Alignment - 1)), "Known static alignment dispatch must be a power of 2");
    details::_memcpy_async_align_dispatch<Alignment>::copy(group, _dst, _src, count);
}

// Internal dispatch APIs
// These deduce the alignments and sizes necessary to invoke the underlying copy engine
template <typename Ty>
using is_void = _CG_STL_NAMESPACE::is_same<Ty, void>;

template <typename Ty>
using enable_if_not_void = typename _CG_STL_NAMESPACE::enable_if<!is_void<Ty>::value, void *>::type;

template <typename Ty>
using enable_if_void = typename _CG_STL_NAMESPACE::enable_if<is_void<Ty>::value, void *>::type;

template <typename Ty>
using enable_if_integral =
    typename _CG_STL_NAMESPACE::enable_if<_CG_STL_NAMESPACE::is_integral<Ty>::value, void *>::type;

// byte-wide API using aligned_sized_t
template <class TyGroup, template <size_t> typename Alignment, size_t Hint>
_CG_STATIC_QUALIFIER void _memcpy_async_bytes(const TyGroup &group, void *__restrict__ _dst,
                                              const void *__restrict__ _src, const Alignment<Hint> &count) {
    constexpr size_t _align = (Hint > 16) ? 16 : Hint;

    details::_memcpy_async_dispatch_to_aligned_copy<_align>(group, _dst, _src, (size_t)count);
}

// byte-wide API using type for aligment
template <class TyGroup, typename TyElem, typename TySize, size_t Hint = alignof(TyElem),
          enable_if_not_void<TyElem> = nullptr, enable_if_integral<TySize> = nullptr>
_CG_STATIC_QUALIFIER void _memcpy_async_bytes(const TyGroup &group, TyElem *__restrict__ _dst,
                                              const TyElem *__restrict__ _src, const TySize& count) {
    constexpr size_t _align = (Hint > 16) ? 16 : Hint;

    details::_memcpy_async_dispatch_to_aligned_copy<_align>(group, _dst, _src, count);
}

// byte-wide API with full alignment deduction required
template <class TyGroup, typename TyElem, typename TySize, enable_if_void<TyElem> = nullptr,
          enable_if_integral<TySize> = nullptr>
_CG_STATIC_QUALIFIER void _memcpy_async_bytes(const TyGroup &group, TyElem *__restrict__ _dst,
                                              const TyElem *__restrict__ _src, const TySize& count) {
    details::_memcpy_async_dispatch_to_aligned_copy<1>(group, _dst, _src, count);
}

// 1d-datum API
template <class TyGroup, typename TyElem, size_t Hint = alignof(TyElem)>
_CG_STATIC_QUALIFIER void _memcpy_async_datum(const TyGroup &group, TyElem *__restrict__ dst, const size_t dstCount,
                                              const TyElem *__restrict__ src, const size_t srcCount) {
    constexpr unsigned int _align = Hint;
    const size_t totalCount = min(dstCount, srcCount) * sizeof(TyElem);

    details::_memcpy_async_dispatch_to_aligned_copy<_align>(group, dst, src, totalCount);
}

// 1d-datum API using aligned_size_t
template <class TyGroup, typename TyElem, template <size_t> typename Alignment, size_t Hint>
_CG_STATIC_QUALIFIER void _memcpy_async_datum(const TyGroup &group, TyElem *__restrict__ dst, const Alignment<Hint> &dstCount,
                                              const TyElem *__restrict__ src, const Alignment<Hint> &srcCount) {
    constexpr unsigned int _align = Hint;
    const size_t totalCount = min((size_t)dstCount, (size_t)srcCount) * sizeof(TyElem);

    details::_memcpy_async_dispatch_to_aligned_copy<_align>(group, dst, src, totalCount);
}

} // namespace details

/*
 * Group submit batch of async-copy to cover contiguous 1D array
 * and commit that batch to eventually wait for completion.
 */
template <class TyGroup, typename TyElem, typename TySizeT>
_CG_STATIC_QUALIFIER void memcpy_async(const TyGroup &group, TyElem *__restrict__ _dst, const TyElem *__restrict__ _src,
                                       const TySizeT &count) {
    details::_memcpy_async_bytes(group, _dst, _src, count);
    __pipeline_commit();
}

/*
 * Group submit batch of async-copy to cover contiguous 1D array
 * and commit that batch to eventually wait for completion.
 * Object counts are in datum sized chunks, not bytes.
 */
template <class TyGroup, class TyElem, typename DstLayout, typename SrcLayout>
_CG_STATIC_QUALIFIER void memcpy_async(const TyGroup &group, TyElem *__restrict__ dst, const DstLayout &dstLayout,
                                       const TyElem *__restrict__ src, const SrcLayout &srcLayout) {
    details::_memcpy_async_datum(group, dst, dstLayout, src, srcLayout);
    __pipeline_commit();
}

/* Group wait for prior Nth stage of memcpy_async to complete. */
template <unsigned int Stage, class TyGroup>
_CG_STATIC_QUALIFIER void wait_prior(const TyGroup &group) {
    __pipeline_wait_prior(Stage);
    group.sync();
}

/* Group wait all previously submitted memcpy_async to complete. */
template <class TyGroup>
_CG_STATIC_QUALIFIER void wait(const TyGroup &group) {
    __pipeline_wait_prior(0);
    group.sync();
}

/***************** CG APIs including pipeline are deprecated *****************/

/* Group submit batch of async-copy to cover of contiguous 1D array
   to a pipeline and commit the batch*/
template <class TyGroup, class TyElem>
_CG_DEPRECATED _CG_STATIC_QUALIFIER void memcpy_async(TyGroup &group, TyElem *dst, size_t dstCount, const TyElem *src, size_t srcCount,
                                       nvcuda::experimental::pipeline &pipe) {
    details::_memcpy_async_datum(group, dst, dstCount, src, srcCount);
    pipe.commit();
}

/* Group wait for prior Nth stage of memcpy_async to complete. */
template <unsigned int Stage, class TyGroup>
_CG_DEPRECATED _CG_STATIC_QUALIFIER void wait_prior(TyGroup &group, nvcuda::experimental::pipeline &pipe) {
    pipe.wait_prior<Stage>();
    group.sync();
}

/* Group wait for stage-S of memcpy_async to complete. */
template <class TyGroup>
_CG_DEPRECATED _CG_STATIC_QUALIFIER void wait(TyGroup &group, nvcuda::experimental::pipeline &pipe, size_t stage) {
    pipe.wait(stage);
    group.sync();
}
_CG_END_NAMESPACE

#endif // _CG_ASYNC_H
