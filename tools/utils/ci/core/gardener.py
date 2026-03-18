# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Gardening workflows for CI corpus maintenance.

Provides tools for:
- Finding unrecognized failures
- Generating TODO lists for pattern development
- Testing new patterns against the corpus
- Computing recognition rates and corpus health metrics
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ci.core.classifier import Classifier
from ci.core.corpus import Corpus


@dataclass
class UnrecognizedFailure:
    """An unrecognized failure needing investigation."""

    run_id: str
    job_id: str
    log_path: Path
    occurrence_count: int  # How many times this pattern appears.
    first_seen: str  # ISO timestamp.
    sample_lines: list[str]  # Sample error lines.


class Gardener:
    """Interactive gardening workflows for corpus maintenance."""

    # Exclude known false positives from unrecognized/TODO.md.
    # These are warnings/errors that appear in logs but are not root causes.
    EXCLUSION_PATTERNS = [
        # Pip dependency resolver warnings (appear during setup, not actual failures).
        "ERROR: pip's dependency resolver",
        "pip's dependency resolver does",
        # Add more false positive patterns here as they're discovered.
    ]

    def __init__(self, corpus: Corpus, classifier: Classifier) -> None:
        """Initialize gardener.

        Args:
            corpus: Corpus instance
            classifier: Classifier instance
        """
        self.corpus = corpus
        self.classifier = classifier

    def find_unrecognized(self) -> list[UnrecognizedFailure]:
        """Find all failures without root causes.

        Returns:
            List of UnrecognizedFailure instances
        """
        unrecognized = []

        # Load all cached classifications.
        cache_dir = self.corpus.classification_dir / "cache"
        if not cache_dir.exists():
            return unrecognized

        for cache_file in cache_dir.glob("*.json"):
            cache_data = json.loads(cache_file.read_text())
            extracted_issues = cache_data.get("extracted_issues", [])

            # Check if unrecognized.
            if len(extracted_issues) == 0:
                run_id = cache_data.get("run_id")
                job_id = cache_data.get("job_id")
                log_path = self.corpus.get_log_path(run_id, job_id)

                if log_path and log_path.exists():
                    # Extract sample error lines.
                    sample_lines = self._extract_error_samples(log_path)

                    unrecognized.append(
                        UnrecognizedFailure(
                            run_id=run_id,
                            job_id=job_id,
                            log_path=log_path,
                            occurrence_count=1,  # Will be updated by grouping.
                            first_seen=cache_data.get("classified_at", ""),
                            sample_lines=sample_lines,
                        )
                    )

        return unrecognized

    def _extract_error_samples(self, log_path: Path, max_lines: int = 10) -> list[str]:
        """Extract sample error lines from a log.

        Args:
            log_path: Path to log file
            max_lines: Maximum number of sample lines to extract

        Returns:
            List of sample error lines
        """
        samples = []
        error_keywords = ["error:", "Error:", "ERROR:", "FAILED", "fatal:", "Fatal:"]

        try:
            with open(log_path) as f:
                for line in f:
                    if any(keyword in line for keyword in error_keywords):
                        samples.append(line.strip())
                        if len(samples) >= max_lines:
                            break
        except (OSError, UnicodeDecodeError):
            pass

        return samples

    def generate_todo(self) -> str:
        """Generate unrecognized/TODO.md file.

        Returns:
            Markdown content for uncategorized failures list
        """
        unrecognized = self.find_unrecognized()

        # Group by similar error patterns (simple heuristic for now).
        pattern_groups = self._group_similar_failures(unrecognized)

        # Generate markdown.
        lines = [
            "# Uncategorized Failures",
            "",
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Recognition rate: {self.compute_recognition_rate():.1%}",
            "",
            "These failures were not recognized by any extractor. To categorize them:",
            "",
            "**Available Extractors:**",
            "- Infrastructure/network/hardware → `tools/utils/common/extractors/infrastructure_flake.py`",
            "- Build errors (C/C++ compilation) → `tools/utils/common/extractors/build_error.py`",
            "- CMake configuration errors → `tools/utils/common/extractors/cmake_error.py`",
            "- Bazel build errors → `tools/utils/common/extractors/bazel_error.py`",
            "- MLIR compiler errors → `tools/utils/common/extractors/mlir_compiler.py`",
            "- Sanitizer failures (ASan/TSan/etc) → `tools/utils/common/extractors/sanitizer.py`",
            "- Pre-commit hook failures → `tools/utils/common/extractors/precommit.py`",
            "- CodeQL analysis errors → `tools/utils/common/extractors/codeql_error.py`",
            "",
            "**After adding patterns:** Run `iree-ci-garden reclassify` to re-process corpus.",
            "",
            "## Failures by Priority",
            "",
        ]

        # Sort by occurrence count (descending).
        sorted_groups = sorted(
            pattern_groups.items(), key=lambda x: x[1]["count"], reverse=True
        )

        # High priority (3+ occurrences).
        high_priority = [g for g in sorted_groups if g[1]["count"] >= 3]
        if high_priority:
            lines.append("### High Priority - Seen 3+ times")
            for i, (pattern, data) in enumerate(high_priority, 1):
                lines.append(f"{i}. **{pattern}** ({data['count']} occurrences)")
                lines.append(f"   - First seen: {data['first_seen']}")
                lines.append(f"   - Logs: {', '.join(data['logs'][:3])}")
                lines.append("   - Action: Add pattern to appropriate extractor")
                lines.append("   - [ ] Categorized")
                lines.append("")

        # Medium priority (2 occurrences).
        med_priority = [g for g in sorted_groups if g[1]["count"] == 2]
        if med_priority:
            lines.append("### Medium Priority - Seen 2 times")
            for i, (pattern, data) in enumerate(med_priority, 1):
                lines.append(f"{i}. **{pattern}** ({data['count']} occurrences)")
                lines.append(f"   - Logs: {', '.join(data['logs'])}")
                lines.append("   - Action: Investigate pattern")
                lines.append("   - [ ] Investigated")
                lines.append("")

        # Low priority (1 occurrence).
        low_priority = [g for g in sorted_groups if g[1]["count"] == 1]
        if low_priority and len(low_priority) <= 20:  # Only show up to 20.
            lines.append("### Low Priority - Seen once")
            for i, (pattern, data) in enumerate(low_priority[:20], 1):
                lines.append(f"{i}. **{pattern}**")
                lines.append(f"   - Log: {data['logs'][0]}")
                lines.append("")

        return "\n".join(lines)

    def _group_similar_failures(
        self, failures: list[UnrecognizedFailure]
    ) -> dict[str, dict[str, Any]]:
        """Group similar failures by error pattern.

        Args:
            failures: List of unrecognized failures

        Returns:
            Dict mapping pattern name to {'count', 'logs', 'first_seen'}
        """
        groups = {}

        for failure in failures:
            # Simple heuristic: use first error line as pattern identifier.
            pattern = (
                failure.sample_lines[0] if failure.sample_lines else "Unknown error"
            )

            # Skip known false positives.
            if self._is_excluded(pattern):
                continue

            # Clean up pattern (remove timestamps, paths, etc.).
            pattern = self._clean_pattern(pattern)

            if pattern not in groups:
                groups[pattern] = {
                    "count": 0,
                    "logs": [],
                    "first_seen": failure.first_seen,
                }

            groups[pattern]["count"] += 1
            groups[pattern]["logs"].append(f"{failure.run_id}/{failure.job_id}.log")

        return groups

    def _is_excluded(self, pattern: str) -> bool:
        """Check if pattern matches exclusion list.

        Args:
            pattern: Error pattern to check

        Returns:
            True if pattern should be excluded
        """
        return any(exclusion in pattern for exclusion in self.EXCLUSION_PATTERNS)

    def _clean_pattern(self, pattern: str) -> str:
        """Clean up error pattern for grouping.

        Args:
            pattern: Raw error line

        Returns:
            Cleaned pattern string
        """
        # Remove timestamps (ISO 8601 format).
        pattern = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z", "", pattern)

        # Remove file paths (anything with /).
        pattern = re.sub(r"/[\w/.-]+", "<path>", pattern)

        # Remove numbers (line numbers, etc.).
        pattern = re.sub(r"\b\d+\b", "<num>", pattern)

        # Truncate if too long.
        if len(pattern) > 100:
            pattern = pattern[:97] + "..."

        return pattern.strip()

    def compute_recognition_rate(self) -> float:
        """Calculate % of logs with identified root causes.

        Returns:
            Recognition rate as float (0.0 to 1.0)
        """
        total = 0
        recognized = 0

        cache_dir = self.corpus.classification_dir / "cache"
        if not cache_dir.exists():
            return 0.0

        for cache_file in cache_dir.glob("*.json"):
            total += 1
            cache_data = json.loads(cache_file.read_text())
            if len(cache_data.get("extracted_issues", [])) > 0:
                recognized += 1

        if total == 0:
            return 0.0

        return recognized / total

    def get_category_breakdown(self) -> dict[str, int]:
        """Get count of logs per category.

        Returns:
            Dict mapping category name to count
        """
        categories = Counter()

        cache_dir = self.corpus.classification_dir / "cache"
        if not cache_dir.exists():
            return dict(categories)

        for cache_file in cache_dir.glob("*.json"):
            cache_data = json.loads(cache_file.read_text())
            for category in cache_data.get("categories", []):
                categories[category] += 1

        return dict(categories)

    def save_todo_list(self) -> None:
        """Generate and save unrecognized/TODO.md file."""
        todo_content = self.generate_todo()
        todo_path = self.corpus.unrecognized_dir / "TODO.md"
        todo_path.write_text(todo_content)

    def get_corpus_health(self) -> dict[str, Any]:
        """Get overall corpus health metrics.

        Returns:
            Dict with health metrics
        """
        stats = self.corpus.get_stats()
        recognition_rate = self.compute_recognition_rate()
        categories = self.get_category_breakdown()

        return {
            "stats": stats,
            "recognition_rate": recognition_rate,
            "categories": categories,
            "health_score": recognition_rate,  # Simple score for now.
        }
