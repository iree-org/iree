// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

CUPTI_PFN_DECL(cuptiActivityEnable, CUpti_ActivityKind)
CUPTI_PFN_DECL(cuptiActivityFlushAll, uint32_t)
CUPTI_PFN_DECL(cuptiActivityGetAttribute, CUpti_ActivityAttribute, size_t*,
               void*)
CUPTI_PFN_DECL(cuptiActivityGetNextRecord, uint8_t*, size_t, CUpti_Activity**)
CUPTI_PFN_DECL(cuptiActivityGetNumDroppedRecords, CUcontext, uint32_t, size_t*)
CUPTI_PFN_DECL(cuptiActivitySetAttribute, CUpti_ActivityAttribute, size_t*,
               void*)
CUPTI_PFN_DECL(cuptiActivityRegisterCallbacks, CUpti_BuffersCallbackRequestFunc,
               CUpti_BuffersCallbackCompleteFunc)
CUPTI_PFN_DECL(cuptiEnableDomain, uint32_t, CUpti_SubscriberHandle,
               CUpti_CallbackDomain)
CUPTI_PFN_DECL(cuptiGetResultString, CUptiResult result, const char** str)
CUPTI_PFN_DECL(cuptiGetTimestamp, uint64_t*)
CUPTI_PFN_DECL(cuptiSubscribe, CUpti_SubscriberHandle*, CUpti_CallbackFunc,
               void*)
