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
"""

import json
import os
import re
import requests
import subprocess
import time


IREE_DASHBOARD_URL='https://lei.ooo'
#IREE_DASHBOARD_URL='http://127.0.0.1:7000'
IREE_GITHUB_COMMIT_URL_PREFIX='https://github.com/google/iree/commit'
IREE_PROJECT_ID='IREE'


IREE_MODELS = {
    'MobileNet v2': 'MobileNet v2 from https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2',
    'MobileNet v3': 'MobileNet v3 from https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small',
    'MobileBERT': 'MobileBERT from https://github.com/google-research/google-research/tree/master/mobilebert'
}

IREE_PHONES = {
    'Pixel 4': {
        'description': 'Pixel 4 chipset Snapdragon 855: https://en.wikichip.org/wiki/qualcomm/snapdragon_800/855',
        'CPU': 'ARM A55/A76',
        'GPU': 'Adreno 640'
    },
    'Galaxy S20': {
        'description': 'Galaxy S20 chipset Exynos 990: https://en.wikichip.org/wiki/samsung/exynos/990',
        'CPU': 'ARM A55/A76',
        'GPU': 'Mali G77'
    }
}


def execute(args, cwd, capture_output=False, text=True, silent=True, **kwargs):
  if not silent:
    print(f"+{' '.join(args)}  [from {cwd}]")
  return subprocess.run(args, cwd=cwd, capture_output=capture_output,
                        check=True, text=text, **kwargs)

def get_serie_id(model_name, bench_mode, phone_name, target_name):
    whitespace = re.compile(r'\W+')
    model_name = whitespace.sub('-', model_name)
    bench_mode = whitespace.sub('-', bench_mode)
    phone_name = whitespace.sub('-', phone_name)
    target_name = whitespace.sub('-', target_name)
    return f'{model_name} {bench_mode} @ {phone_name} ({target_name})'

def get_git_commit_hash(commit):
  return execute(['git', 'rev-parse', commit],
                 cwd=os.path.dirname(os.path.realpath(__file__)),
                 capture_output=True).stdout.strip()


def get_git_total_commit_count(commit):
  count = execute(['git', 'rev-list', '--count', commit],
                  cwd=os.path.dirname(os.path.realpath(__file__)),
                  capture_output=True).stdout.strip()
  return int(count)


def get_git_commit_info(commit):
  info = execute(['git', 'show', '--format=%H:::%h:::%an:::%ae:::%s',
                  '--no-patch', commit],
                 cwd=os.path.dirname(os.path.realpath(__file__)),
                 capture_output=True).stdout.strip()
  segments = info.split(':::')
  return {'hash': segments[0],
          'abbrevHash': segments[1],
          'authorName': segments[2],
          'authorEmail': segments[3],
          'subject': segments[4]}


def compose_serie_payload(project_id, serie_id, serie_description=None,
                          override = False, average_range='5%',
                          average_min_count=3, better_criterion='smaller'):
  payload = {
    'projectId': project_id,
    'serieId': serie_id,
    'analyse': {
        'benchmark': {
            'range': average_range,
            'required': average_min_count,
            'trend': better_criterion
        }
    },
    'override': override
  }
  if serie_description is not None:
      payload['description'] = serie_description
  return payload


def compose_build_payload(project_id, project_github_comit_url,
                          build_id, commit, override=False):
  commit_info = get_git_commit_info(commit)
  commit_info['url'] = f'{project_github_comit_url}/{commit_info["hash"]}'
  return {
    'projectId': project_id,
    'build': {
        'buildId': build_id,
        'infos': commit_info
    },
    'override': override
  }

def compose_sample_payload(project_id, serie_id, build_id, sample_value, override=False):
  return {
      'projectId': project_id,
      'serieId': serie_id,
      'sample': {
          'buildId': build_id,
          'value': sample_value
      },
      'override': override
  }


def post_to_dashboard(url, payload, silent=True):
    api_token = os.getenv('IREE_DASHBOARD_API_TOKEN', None)
    if api_token is None:
        raise RuntimeError('Missing IREE_DASHBOARD_API_TOKEN')
    headers = {
        'Content-type': 'application/json',
        'Authorization': f'Bearer {api_token}'
    }
    data = json.dumps(payload)
    if not silent:
        print(f'-api request payload: {data}')
    for attempt in range(5):
        request = requests.post(url, data=data, headers=headers)
        if not silent:
            print(f'-api request status code: {request.status_code}')
        if request.status_code == 200 or request.status_code == 500:
            return
        time.sleep(1)
    raise requests.RequestException('Failed to post to dashboard server')


def add_new_iree_serie(serie_id, serie_description=None, override=False):
    payload = compose_serie_payload(IREE_PROJECT_ID, serie_id,
                                    serie_description, override)
    post_to_dashboard(f'{IREE_DASHBOARD_URL}/apis/addSerie', payload)


def add_new_iree_build(build_id, commit, override=False):
    payload = compose_build_payload(IREE_PROJECT_ID,
                                    IREE_GITHUB_COMMIT_URL_PREFIX,
                                    build_id, commit, override)
    post_to_dashboard(f'{IREE_DASHBOARD_URL}/apis/addBuild', payload)


def add_new_sample(serie_id, build_id, sample_value, override=False):
    payload = compose_sample_payload(IREE_PROJECT_ID, serie_id,
                                     build_id, sample_value, override)
    post_to_dashboard(f'{IREE_DASHBOARD_URL}/apis/addSample', payload)


if __name__ == '__main__':
    import random

    bench_mode = ['whole inference']

    series = []
    for model in IREE_MODELS.keys():
        for mode in bench_mode:
            for phone in IREE_PHONES.items():
                cpu, gpu = phone[1]['CPU'], phone[1]['GPU']
                series.append((model, phone[0], get_serie_id(model, mode, phone[0], f'CPU {cpu}')))
                series.append((model, phone[0], get_serie_id(model, mode, phone[0], f'GPU {gpu}')))


    for serie in series:
        model_description = IREE_MODELS[serie[0]]
        phone_description = IREE_PHONES[serie[1]]['description']
        add_new_iree_serie(
            serie[2], f'{model_description}<br>{phone_description}', True)

    random.seed(1970)
    for i in reversed(range(100)):
        commit = get_git_commit_hash(f'HEAD~{i}')
        commit_count = get_git_total_commit_count(commit)
        add_new_iree_build(commit_count, commit, True)
        for serie in series:
            add_new_sample(serie[2], commit_count, 1000 + random.randrange(500), True)
