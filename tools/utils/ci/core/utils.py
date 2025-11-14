# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared utility functions for CI tools."""


def is_meta_job(job_name: str) -> bool:
    """Check if a job is a meta/summary job.

    Meta-jobs aggregate results from other jobs and should not be analyzed
    as individual failures. They include summary jobs, aggregation jobs, etc.

    Args:
        job_name: Job name string

    Returns:
        True if job is a meta-job (aggregate, summary, etc.)

    Examples:
        >>> is_meta_job("pkgci_summary / summary")
        True
        >>> is_meta_job("ci_summary / summary")
        True
        >>> is_meta_job("Test Torch / torch_models tests")
        False
    """
    meta_keywords = ["summary", "aggregate", "pkgci_summary", "job-summary"]
    job_name_lower = job_name.lower()
    return any(keyword in job_name_lower for keyword in meta_keywords)
