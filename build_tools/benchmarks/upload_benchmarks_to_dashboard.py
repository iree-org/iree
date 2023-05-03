#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Upload benchmark results to IREE Benchmark Dashboards.

This script is meant to be used by Buildkite for automation.

Example usage:
  # Export necessary environment variables:
  export IREE_DASHBOARD_API_TOKEN=...
  # Then run the script:
  python3 upload_benchmarks.py /path/to/benchmark/json/file
"""

import argparse
import json
import os
import pathlib
import requests

from typing import Any, Dict, Optional, Union

from common.common_arguments import expand_and_check_file_paths
from common import benchmark_definition, benchmark_presentation, benchmark_thresholds

IREE_DASHBOARD_URL = "https://perf.iree.dev"
IREE_GITHUB_COMMIT_URL_PREFIX = 'https://github.com/openxla/iree/commit'
IREE_PROJECT_ID = 'IREE'
THIS_DIRECTORY = pathlib.Path(__file__).resolve().parent

COMMON_DESCRIPTION = """
<br>
For the graph, the x axis is the Git commit index, and the y axis is the
measured metrics. The unit for the numbers is shown in the "Unit" dropdown.
<br>
See <a href="https://github.com/openxla/iree/tree/main/benchmarks/dashboard.md">
https://github.com/openxla/iree/tree/main/benchmarks/dashboard.md
</a> for benchmark philosophy, specification, and definitions.
"""

# A non-exhaustive list of models and their source URLs.
# For models listed here we can provide a nicer description for them on
# webpage.
IREE_TF_MODEL_SOURCE_URL = {
    'MobileBertSquad':
        'https://github.com/google-research/google-research/tree/master/mobilebert',
    'MobileNetV2':
        'https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2',
    'MobileNetV3Small':
        'https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small',
}

IREE_TFLITE_MODEL_SOURCE_URL = {
    'DeepLabV3':
        'https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1',
    'MobileSSD':
        'https://www.tensorflow.org/lite/performance/gpu#demo_app_tutorials',
    'PoseNet':
        'https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1',
}


def get_model_description(model_name: str, model_source: str) -> Optional[str]:
  """Gets the model description for the given benchmark."""
  url = None
  if model_source == "TensorFlow":
    url = IREE_TF_MODEL_SOURCE_URL.get(model_name)
  elif model_source == "TFLite":
    url = IREE_TFLITE_MODEL_SOURCE_URL.get(model_name)
  if url is not None:
    description = f'{model_name} from <a href="{url}">{url}</a>.'
    return description
  return None


def get_git_commit_hash(commit: str, verbose: bool = False) -> str:
  """Gets the commit hash for the given commit."""
  return benchmark_definition.execute_cmd_and_get_stdout(
      ['git', 'rev-parse', commit], cwd=THIS_DIRECTORY, verbose=verbose)


def get_git_total_commit_count(commit: str, verbose: bool = False) -> int:
  """Gets the total commit count in history ending with the given commit."""
  count = benchmark_definition.execute_cmd_and_get_stdout(
      ['git', 'rev-list', '--count', commit],
      cwd=THIS_DIRECTORY,
      verbose=verbose)
  return int(count)


def get_git_commit_info(commit: str, verbose: bool = False) -> Dict[str, str]:
  """Gets commit information dictionary for the given commit."""
  cmd = [
      'git', 'show', '--format=%H:::%h:::%an:::%ae:::%s', '--no-patch', commit
  ]
  info = benchmark_definition.execute_cmd_and_get_stdout(cmd,
                                                         cwd=THIS_DIRECTORY,
                                                         verbose=verbose)
  segments = info.split(':::')
  return {
      'hash': segments[0],
      'abbrevHash': segments[1],
      'authorName': segments[2],
      'authorEmail': segments[3],
      'subject': segments[4],
  }


def compose_series_payload(project_id: str,
                           series_id: str,
                           series_unit: str,
                           series_name: Optional[str] = None,
                           series_description: Optional[str] = None,
                           average_range: Union[int, str] = '5%',
                           average_min_count: int = 3,
                           better_criterion: str = 'smaller',
                           override: bool = False) -> Dict[str, Any]:
  """Composes the payload dictionary for a series."""
  payload = {
      'projectId': project_id,
      'serieId': series_id,
      'serieUnit': series_unit,
      'serieName': series_name,
      'analyse': {
          'benchmark': {
              'range': average_range,
              'required': average_min_count,
              'trend': better_criterion,
          }
      },
      'override': override,
  }
  if series_description is not None:
    payload['description'] = series_description
  return payload


def compose_build_payload(project_id: str,
                          project_github_commit_url: str,
                          build_id: int,
                          commit: str,
                          override: bool = False) -> Dict[str, Any]:
  """Composes the payload dictionary for a build."""
  commit_info = get_git_commit_info(commit)
  commit_info['url'] = f'{project_github_commit_url}/{commit_info["hash"]}'
  return {
      'projectId': project_id,
      'build': {
          'buildId': build_id,
          'infos': commit_info,
      },
      'override': override,
  }


def compose_sample_payload(project_id: str,
                           series_id: str,
                           build_id: int,
                           sample_unit: str,
                           sample_value: int,
                           override: bool = False) -> Dict[str, Any]:
  """Composes the payload dictionary for a sample."""
  return {
      'projectId': project_id,
      'serieId': series_id,
      'sampleUnit': sample_unit,
      'sample': {
          'buildId': build_id,
          'value': sample_value
      },
      'override': override
  }


def get_required_env_var(var: str) -> str:
  """Gets the value for a required environment variable."""
  value = os.getenv(var)
  if value is None:
    raise RuntimeError(f'Missing environment variable "{var}"')
  return value


def post_to_dashboard(url: str,
                      payload: Dict[str, Any],
                      dry_run: bool = False,
                      verbose: bool = False):
  data = json.dumps(payload)

  if dry_run or verbose:
    print(f'API request payload: {data}')

  if dry_run:
    return

  api_token = get_required_env_var('IREE_DASHBOARD_API_TOKEN')
  headers = {
      'Content-type': 'application/json',
      'Authorization': f'Bearer {api_token}',
  }

  response = requests.post(url, data=data, headers=headers)
  code = response.status_code
  if code != 200:
    raise requests.RequestException(
        f'Failed to post to dashboard server with {code} - {response.text}')


def add_new_iree_series(series_id: str,
                        series_unit: str,
                        series_name: str,
                        series_description: Optional[str] = None,
                        average_range: Optional[Union[str, int]] = None,
                        override: bool = False,
                        dry_run: bool = False,
                        verbose: bool = False):
  """Posts a new series to the dashboard."""
  if average_range is None:
    raise ValueError(f"no matched threshold setting for benchmark: {series_id}")

  payload = compose_series_payload(IREE_PROJECT_ID,
                                   series_id,
                                   series_unit,
                                   series_name,
                                   series_description,
                                   average_range=average_range,
                                   override=override)
  post_to_dashboard(f'{IREE_DASHBOARD_URL}/apis/v2/addSerie',
                    payload,
                    dry_run=dry_run,
                    verbose=verbose)


def add_new_iree_build(build_id: int,
                       commit: str,
                       override: bool = False,
                       dry_run: bool = False,
                       verbose: bool = False):
  """Posts a new build to the dashboard."""
  payload = compose_build_payload(IREE_PROJECT_ID,
                                  IREE_GITHUB_COMMIT_URL_PREFIX, build_id,
                                  commit, override)
  post_to_dashboard(f'{IREE_DASHBOARD_URL}/apis/addBuild',
                    payload,
                    dry_run=dry_run,
                    verbose=verbose)


def add_new_sample(series_id: str,
                   build_id: int,
                   sample_unit: str,
                   sample_value: int,
                   override: bool = False,
                   dry_run: bool = False,
                   verbose: bool = False):
  """Posts a new sample to the dashboard."""
  payload = compose_sample_payload(IREE_PROJECT_ID, series_id, build_id,
                                   sample_unit, sample_value, override)
  post_to_dashboard(f'{IREE_DASHBOARD_URL}/apis/v2/addSample',
                    payload,
                    dry_run=dry_run,
                    verbose=verbose)


def parse_arguments():
  """Parses command-line options."""

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--benchmark_files',
      metavar='<benchmark-json-files>',
      default=[],
      nargs='+',
      help=("Paths to the JSON files containing benchmark results, "
            "accepts wildcards"))
  parser.add_argument(
      "--compile_stats_files",
      metavar="<compile-stats-json-files>",
      default=[],
      nargs="+",
      help=("Paths to the JSON files containing compilation statistics, "
            "accepts wildcards"))
  parser.add_argument("--dry-run",
                      action="store_true",
                      help="Print the comment instead of posting to dashboard")
  parser.add_argument('--verbose',
                      action='store_true',
                      help='Print internal information during execution')
  args = parser.parse_args()

  return args


def main(args):
  benchmark_files = expand_and_check_file_paths(args.benchmark_files)
  compile_stats_files = expand_and_check_file_paths(args.compile_stats_files)

  if len(benchmark_files) > 0:
    committish = benchmark_definition.BenchmarkResults.from_json_str(
        benchmark_files[0].read_text()).commit
  elif len(compile_stats_files) > 0:
    committish = benchmark_definition.CompilationResults.from_json_object(
        json.loads(compile_stats_files[0].read_text())).commit
  else:
    raise ValueError("No benchmark/compilation results.")

  # Register a new build for the current commit.
  commit_hash = get_git_commit_hash(committish, verbose=args.verbose)
  commit_count = get_git_total_commit_count(commit_hash, verbose=args.verbose)

  aggregate_results = benchmark_presentation.aggregate_all_benchmarks(
      benchmark_files=benchmark_files, expected_pr_commit=commit_hash)

  all_compilation_metrics = benchmark_presentation.collect_all_compilation_metrics(
      compile_stats_files=compile_stats_files, expected_pr_commit=commit_hash)

  # Allow override to support uploading data for the same build in
  # different batches.
  add_new_iree_build(commit_count,
                     commit_hash,
                     override=True,
                     dry_run=args.dry_run,
                     verbose=args.verbose)

  # Upload benchmark results to the dashboard.
  for series_id, benchmark_latency in aggregate_results.items():
    series_name = benchmark_latency.name
    benchmark_info = benchmark_latency.benchmark_info
    description = get_model_description(benchmark_info.model_name,
                                        benchmark_info.model_source)
    if description is None:
      description = ""
    description += COMMON_DESCRIPTION

    threshold = next(
        (threshold for threshold in benchmark_thresholds.BENCHMARK_THRESHOLDS
         if threshold.regex.match(series_name)), None)
    average_range = (threshold.get_threshold_str()
                     if threshold is not None else None)

    # Override by default to allow updates to the series.
    add_new_iree_series(series_id=series_id,
                        series_unit="ns",
                        series_name=benchmark_latency.name,
                        series_description=description,
                        average_range=average_range,
                        override=True,
                        dry_run=args.dry_run,
                        verbose=args.verbose)
    add_new_sample(series_id=series_id,
                   build_id=commit_count,
                   sample_unit="ns",
                   sample_value=benchmark_latency.mean_time,
                   dry_run=args.dry_run,
                   verbose=args.verbose)

  for target_id, compile_metrics in all_compilation_metrics.items():
    description = get_model_description(
        compile_metrics.compilation_info.model_name,
        compile_metrics.compilation_info.model_source)
    if description is None:
      description = ""
    description += COMMON_DESCRIPTION

    for mapper in benchmark_presentation.COMPILATION_METRICS_TO_TABLE_MAPPERS:
      sample_value, _ = mapper.get_current_and_base_value(compile_metrics)
      series_unit = mapper.get_unit()
      series_id = mapper.get_series_id(target_id)
      series_name = mapper.get_series_name(compile_metrics.name)

      threshold = next(
          (threshold for threshold in mapper.get_metric_thresholds()
           if threshold.regex.match(series_name)), None)
      average_range = (threshold.get_threshold_str()
                       if threshold is not None else None)

      # Override by default to allow updates to the series.
      add_new_iree_series(series_id=series_id,
                          series_unit=series_unit,
                          series_name=series_name,
                          series_description=description,
                          average_range=average_range,
                          override=True,
                          dry_run=args.dry_run,
                          verbose=args.verbose)
      add_new_sample(series_id=series_id,
                     build_id=commit_count,
                     sample_unit=series_unit,
                     sample_value=sample_value,
                     dry_run=args.dry_run,
                     verbose=args.verbose)


if __name__ == "__main__":
  main(parse_arguments())
