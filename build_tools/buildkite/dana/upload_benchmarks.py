#!/usr/bin/env python3
# Lint as: python3
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Upload benchmark results to IREE Benchmark Dashboards.

This script is meant to be used by Buildkite for automation.

Example usage:
  # Export necessary environment variables:
  export IREE_DASHBOARD_URL=...
  export IREE_DASHBOARD_API_TOKEN=...
  # Then run the script:
  python3 upload_benchmarks.py /path/to/benchmark/json/file
"""

import argparse
import json
import os
import re
import requests
import subprocess
import time

IREE_GITHUB_COMMIT_URL_PREFIX = 'https://github.com/google/iree/commit'
IREE_PROJECT_ID = 'IREE'

# Number of total retries when post to dashboard.
RETRY_COUNT = 5

# A non-exhaustive list of models and their descriptions.
# For models listed here we can provide a nicer description for them on
# webpage.
IREE_MODELS = {
    'MobileNetV2 (imagenet) [TensorFlow]':
        'MobileNet v2 from https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2',
    'MobileNetV3Small (imagenet) [TensorFlow]':
        'MobileNet v3 from https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small',
    'MobileBertSquad [TensorFlow]':
        'MobileBERT (SQuAD) from https://github.com/google-research/google-research/tree/master/mobilebert'
}

# A non-exhaustive list of phones and their descriptions.
# For phones listed here we can provide a nicer description for them on
# webpage.
IREE_PHONES = {
    'Pixel-4':
        'Google Pixel 4 chipset: Snapdragon 855 (https://en.wikichip.org/wiki/qualcomm/snapdragon_800/855)',
    'SM-G980F':
        'Samsung Galaxy S20 chipset: Exynos 990 (https://en.wikichip.org/wiki/samsung/exynos/990)',
}


def execute(args,
            cwd,
            capture_output=False,
            treat_io_as_text=True,
            verbose=False,
            **kwargs):
  """Executes a command."""
  if verbose:
    print(f"+{' '.join(args)}  [from {cwd}]")
  return subprocess.run(args,
                        cwd=cwd,
                        check=True,
                        capture_output=capture_output,
                        text=treat_io_as_text,
                        **kwargs)


def get_git_commit_hash(commit, verbose=False):
  """Gets the commit hash for the given commit."""
  return execute(['git', 'rev-parse', commit],
                 cwd=os.path.dirname(os.path.realpath(__file__)),
                 capture_output=True,
                 verbose=verbose).stdout.strip()


def get_git_total_commit_count(commit, verbose=False):
  """Gets the total commit count in history ending with the given commit."""
  count = execute(['git', 'rev-list', '--count', commit],
                  cwd=os.path.dirname(os.path.realpath(__file__)),
                  capture_output=True,
                  verbose=verbose).stdout.strip()
  return int(count)


def get_git_commit_info(commit, verbose=False):
  """Gets commit information dictory for the given commit."""
  info = execute([
      'git', 'show', '--format=%H:::%h:::%an:::%ae:::%s', '--no-patch', commit
  ],
                 cwd=os.path.dirname(os.path.realpath(__file__)),
                 capture_output=True,
                 verbose=verbose).stdout.strip()
  segments = info.split(':::')
  return {
      'hash': segments[0],
      'abbrevHash': segments[1],
      'authorName': segments[2],
      'authorEmail': segments[3],
      'subject': segments[4],
  }


def compose_serie_payload(project_id,
                          serie_id,
                          serie_description=None,
                          average_range='5%',
                          average_min_count=3,
                          better_criterion='smaller',
                          override=False):
  """Composes the payload dictionary for a serie."""
  payload = {
      'projectId': project_id,
      'serieId': serie_id,
      'analyse': {
          'benchmark': {
              'range': average_range,
              'required': average_min_count,
              'trend': better_criterion,
          }
      },
      'override': override,
  }
  if serie_description is not None:
    payload['description'] = serie_description
  return payload


def compose_build_payload(project_id,
                          project_github_comit_url,
                          build_id,
                          commit,
                          override=False):
  """Composes the payload dictionary for a build."""
  commit_info = get_git_commit_info(commit)
  commit_info['url'] = f'{project_github_comit_url}/{commit_info["hash"]}'
  return {
      'projectId': project_id,
      'build': {
          'buildId': build_id,
          'infos': commit_info,
      },
      'override': override,
  }


def compose_sample_payload(project_id,
                           serie_id,
                           build_id,
                           sample_value,
                           override=False):
  """Composes the payload dictionary for a sample."""
  return {
      'projectId': project_id,
      'serieId': serie_id,
      'sample': {
          'buildId': build_id,
          'value': sample_value
      },
      'override': override
  }


def get_env_var(var):
  """Gets the value for an environment variable."""
  value = os.getenv(var, None)
  if value is None:
    raise RuntimeError(f'Missing environment variable "{var}"')
  return value


def post_to_dashboard(url, payload, verbose=False):
  api_token = get_env_var('IREE_DASHBOARD_API_TOKEN')
  headers = {
      'Content-type': 'application/json',
      'Authorization': f'Bearer {api_token}',
  }
  data = json.dumps(payload)

  if verbose:
    print(f'-api request payload: {data}')

  for attempt in range(RETRY_COUNT):
    request = requests.post(url, data=data, headers=headers)
    if verbose:
      print(f'-api request status code: {request.status_code}')
    # Return if it succeeded or the content already exists.
    if request.status_code == 200 or request.status_code == 500:
      return
    # Otherwise pause a bit and then retry.
    time.sleep(1)

  raise requests.RequestException('Failed to post to dashboard server')


def add_new_iree_serie(serie_id,
                       serie_description=None,
                       override=False,
                       verbose=False):
  """Posts a new serie to the dashboard."""
  url = get_env_var('IREE_DASHBOARD_URL')
  payload = compose_serie_payload(IREE_PROJECT_ID, serie_id, serie_description,
                                  override)
  post_to_dashboard(f'{url}/apis/addSerie', payload, verbose=verbose)


def add_new_iree_build(build_id, commit, override=False, verbose=False):
  """Posts a new build to the dashboard."""
  url = get_env_var('IREE_DASHBOARD_URL')
  payload = compose_build_payload(IREE_PROJECT_ID,
                                  IREE_GITHUB_COMMIT_URL_PREFIX, build_id,
                                  commit, override)
  post_to_dashboard(f'{url}/apis/addBuild', payload, verbose=verbose)


def add_new_sample(serie_id,
                   build_id,
                   sample_value,
                   override=False,
                   verbose=False):
  """Posts a new sample to the dashboard."""
  url = get_env_var('IREE_DASHBOARD_URL')
  payload = compose_sample_payload(IREE_PROJECT_ID, serie_id, build_id,
                                   sample_value, override)
  post_to_dashboard(f'{url}/apis/addSample', payload, verbose=verbose)


def parse_arguments():
  """Parses command-line options."""

  def check_file_path(path):
    if os.path.isfile(path):
      return path
    else:
      raise ValueError(path)

  parser = argparse.ArgumentParser()
  parser.add_argument("benchmark_file",
                      metavar="<benchmark-json-file>",
                      type=check_file_path,
                      help="Path to the JSON file containing benchmark results")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")
  args = parser.parse_args()

  return args


def main(args):
  with open(args.benchmark_file) as f:
    benchmarks = json.load(f)

  commit_hash = get_git_commit_hash(benchmarks['commit'], verbose=args.verbose)
  commit_count = get_git_total_commit_count(commit_hash, verbose=args.verbose)
  add_new_iree_build(commit_count, commit_hash, verbose=args.verbose)

  for serie_id, sample_value in benchmarks['benchmarks'].items():
    model_description = ''
    for model_name, model_link in IREE_MODELS.items():
      if model_name in serie_id:
        model_description = model_link
        break

    phone_description = ''
    for phone_name, phone_link in IREE_PHONES.items():
      if phone_name in serie_id:
        phone_description = phone_link
        break

    description = ''
    if model_description or phone_description:
      description = f'{model_description}<br>{phone_description}'

    # Override by default to allow updates to the serie.
    add_new_iree_serie(serie_id,
                       description,
                       override=True,
                       verbose=args.verbose)
    add_new_sample(serie_id, commit_count, sample_value, verbose=args.verbose)


if __name__ == "__main__":
  main(parse_arguments())
