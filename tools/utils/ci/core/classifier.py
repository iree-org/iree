# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Classification system for CI failures.

Uses TriageEngine library directly to extract issues from corpus logs.
Results are cached to avoid re-analyzing unchanged logs.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from common import console
from common.extractors.infrastructure_flake import InfrastructureFlakeExtractor
from common.log_buffer import LogBuffer

from ci.core.corpus import Corpus
from ci.iree_ci_triage import create_ci_triage_engine


@dataclass
class ClassificationResult:
    """Results from classifying a single log."""

    run_id: str
    job_id: str
    extracted_issues: list[dict[str, Any]]
    categories: list[str]
    recognized: bool  # True if at least one issue extracted.
    cached: bool


@dataclass
class ClassificationReport:
    """Report from classifying multiple logs."""

    total_logs: int
    classified: int
    recognized: int
    unrecognized: int
    recognition_rate: float
    categories: dict[str, int]  # Category name -> count.
    errors: list[str]
    results: list[ClassificationResult]


class Classifier:
    """Classifies CI failures using TriageEngine library."""

    def __init__(self, corpus: Corpus) -> None:
        """Initialize classifier.

        Args:
            corpus: Corpus instance
        """
        self.corpus = corpus
        self.cache_dir = corpus.classification_dir / "cache"
        self.triage_engine = create_ci_triage_engine()

    def classify_run(
        self, run_id: str, force_reclassify: bool = False
    ) -> list[ClassificationResult]:
        """Classify all logs in a run.

        Args:
            run_id: Run ID to classify
            force_reclassify: If True, ignore cache and reclassify

        Returns:
            List of ClassificationResult for each job log
        """
        # Find all logs for this run.
        run_dir = self.corpus.logs_dir / run_id
        if not run_dir.exists():
            return []

        log_files = list(run_dir.glob("*.log"))

        # Parallelize log classification (embarrassingly parallel workload).
        # Thread-safe: extractors are stateless, file reads independent.
        # max_workers=None uses all available cores.
        with ThreadPoolExecutor(max_workers=None) as executor:
            futures = [
                executor.submit(
                    self.classify_log, run_id, log_file.stem, force_reclassify
                )
                for log_file in log_files
            ]
            results = [f.result() for f in futures]

        # Filter out None results.
        results = [r for r in results if r]

        # Update run metadata to mark as classified.
        run_meta = self.corpus.get_run_metadata(run_id)
        if run_meta:
            run_meta["classified"] = True
            run_meta["classified_at"] = datetime.now().isoformat()
            self.corpus.save_run_metadata(run_id, run_meta)

        return results

    def classify_log(
        self, run_id: str, job_id: str, force_reclassify: bool = False
    ) -> ClassificationResult | None:
        """Classify a single log file.

        Args:
            run_id: Run ID
            job_id: Job ID
            force_reclassify: If True, ignore cache

        Returns:
            ClassificationResult, or None if log doesn't exist
        """
        log_path = self.corpus.get_log_path(run_id, job_id)
        if not log_path:
            return None

        # Check cache first.
        cache_path = self.cache_dir / f"{run_id}_{job_id}.json"
        if not force_reclassify and cache_path.exists():
            # Check if cache is still valid (TTL check).
            cache_age = datetime.now() - datetime.fromtimestamp(
                cache_path.stat().st_mtime
            )
            config = json.loads(self.corpus.config_path.read_text())
            ttl_days = config.get("classification_settings", {}).get(
                "cache_ttl_days", 30
            )

            if cache_age < timedelta(days=ttl_days):
                # Load from cache.
                cached_data = json.loads(cache_path.read_text())
                extracted_issues = cached_data.get("extracted_issues", [])
                return ClassificationResult(
                    run_id=run_id,
                    job_id=job_id,
                    extracted_issues=extracted_issues,
                    categories=cached_data.get("categories", []),
                    recognized=len(extracted_issues) > 0,
                    cached=True,
                )

        # Run triage engine directly.
        try:
            # Check for annotations first.
            annotation_issues = []
            annotations = self.corpus.get_annotations(run_id, job_id)
            if annotations:
                # Extract issues from annotations using infrastructure_flake extractor.
                infra_extractor = InfrastructureFlakeExtractor()
                annotation_issues = infra_extractor.extract_from_annotations(
                    annotations, run_id, job_id
                )

            # Read log content (only if it exists).
            log_content = log_path.read_text()

            # Create log buffer and analyze.
            log_buffer = LogBuffer(log_content, auto_detect_format=True)
            triage_result = self.triage_engine.analyze(log_buffer)

            # Combine issues from annotations and log analysis.
            all_issues = annotation_issues + triage_result.issues

            # Serialize issues to dictionaries.
            extracted_issues = [
                {
                    "title": issue.message[:100],
                    "message": issue.message,
                    "severity": issue.severity.name,
                    "extractor": issue.source_extractor,
                }
                for issue in all_issues
            ]

            # Extract unique categories (extractor names).
            categories = list(set(issue["extractor"] for issue in extracted_issues))

            # Determine extraction status.
            extraction_status = "success" if extracted_issues else "incomplete"

            # Cache the result.
            cache_data = {
                "run_id": run_id,
                "job_id": job_id,
                "extracted_issues": extracted_issues,
                "categories": categories,
                "extraction_status": extraction_status,
                "classified_at": datetime.now().isoformat(),
            }
            cache_path.write_text(json.dumps(cache_data, indent=2))

            return ClassificationResult(
                run_id=run_id,
                job_id=job_id,
                extracted_issues=extracted_issues,
                categories=categories,
                recognized=len(extracted_issues) > 0,
                cached=False,
            )

        except Exception as e:  # noqa: BLE001
            # Classification failed - catch all exceptions to prevent single log from blocking entire corpus.
            console.warn(f"Triage failed for {run_id}/{job_id}: {e}")
            return ClassificationResult(
                run_id=run_id,
                job_id=job_id,
                extracted_issues=[],
                categories=["error"],
                recognized=False,
                cached=False,
            )

    def _classify_runs(
        self, runs: list, force_reclassify: bool, action: str = "Classifying"
    ) -> ClassificationReport:
        """Classify a list of runs.

        Args:
            runs: List of run metadata dicts to classify
            force_reclassify: If True, ignore cache and reclassify
            action: Action verb for progress messages (e.g., "Classifying", "Reclassifying")

        Returns:
            ClassificationReport with statistics
        """
        report = ClassificationReport(
            total_logs=0,
            classified=0,
            recognized=0,
            unrecognized=0,
            recognition_rate=0.0,
            categories={},
            errors=[],
            results=[],
        )

        # Gather ALL logs across ALL runs first.
        all_log_tasks = []
        for run in runs:
            run_id = run["run_id"]
            run_dir = self.corpus.logs_dir / run_id
            if not run_dir.exists():
                continue

            for log_file in run_dir.glob("*.log"):
                all_log_tasks.append((run_id, log_file.stem))

        total_logs = len(all_log_tasks)
        console.note(f"{action} {total_logs} logs across {len(runs)} runs...")

        # Parallelize ALL logs at once (not per-run).
        # Thread-safe: extractors are stateless, file reads independent.
        # max_workers=None uses all available cores.
        results = []
        with ThreadPoolExecutor(max_workers=None) as executor:
            # Submit all tasks.
            future_to_task = {
                executor.submit(self.classify_log, run_id, job_id, force_reclassify): (
                    run_id,
                    job_id,
                )
                for (run_id, job_id) in all_log_tasks
            }

            # Process results as they complete (for progress reporting).
            for completed, future in enumerate(as_completed(future_to_task), start=1):
                run_id, job_id = future_to_task[future]
                result = future.result()
                if result:
                    results.append(result)

                # Print progress with log file path.
                console.note(
                    f"  [{completed}/{total_logs}] processed log: {run_id}/{job_id}.log"
                )

        # Filter out None results and build report.
        results = [r for r in results if r]
        report.results = results

        for result in results:
            report.total_logs += 1
            report.classified += 1
            if result.recognized:
                report.recognized += 1
            else:
                report.unrecognized += 1

            # Count categories.
            for category in result.categories:
                report.categories[category] = report.categories.get(category, 0) + 1

        # Update run metadata for all processed runs.
        for run in runs:
            run_id = run["run_id"]
            run_meta = self.corpus.get_run_metadata(run_id)
            if run_meta:
                run_meta["classified"] = True
                run_meta["classified_at"] = datetime.now().isoformat()
                self.corpus.save_run_metadata(run_id, run_meta)

        # Calculate recognition rate.
        if report.total_logs > 0:
            report.recognition_rate = report.recognized / report.total_logs

        return report

    def classify_all_unclassified(self) -> ClassificationReport:
        """Classify all unclassified runs in corpus.

        Returns:
            ClassificationReport with statistics
        """
        unclassified_runs = list(self.corpus.get_unclassified())
        return self._classify_runs(
            unclassified_runs, force_reclassify=False, action="Classifying"
        )

    def reclassify_all(self) -> ClassificationReport:
        """Force reclassification of entire corpus.

        Returns:
            ClassificationReport with statistics
        """
        all_runs = list(self.corpus.get_runs())
        return self._classify_runs(
            all_runs, force_reclassify=True, action="Reclassifying"
        )
