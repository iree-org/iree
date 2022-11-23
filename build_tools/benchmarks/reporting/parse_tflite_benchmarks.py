# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Script to generate a HTML summary comparing IREE and TFLite latencies and memory usage.

Example usage:

python parse_tflite_benchmarks.py \
  --iree_version=20220924.276 \
  --tflite_version=20220924.162 \
  --platform=server \
  --input_csv=server_results.csv \
  --output_path=/tmp/server_summary.html

"""

import argparse
import pandas as pd
import pathlib
import sys

from datetime import date

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent / ".." / ".." / "python"))
from reporting.common import html_utils

# Supported platforms.
_PLATFORM_SERVER = "server"
_PLATFORM_MOBILE = "mobile"

# A map of model name to data type.
_MODEL_TO_DATA_TYPE = {
    "albert_lite_base_squadv1_1": "fp32",
    "albert_lite_base_squadv1_1_fp16": "fp16",
    "deeplabv3": "fp32",
    "deeplabv3_fp16": "fp16",
    "efficientnet_lite0_fp32_2": "fp32",
    "efficientnet_lite0_fp32_2_fp16": "fp16",
    "efficientnet_lite0_int8_2": "int8",
    "inception_v4_299_fp32": "fp32",
    "inception_v4_299_fp32_fp16": "fp16",
    "inception_v4_299_uint8": "uint8",
    "mobilebert-baseline-tf2-quant": "int8",
    "mobilebert_float_384_gpu": "fp32",
    "mobilebert_float_384_gpu_fp16": "fp16",
    "mobilenet_v2_1.0_224": "fp32",
    "mobilenet_v2_1.0_224_fp16": "fp16",
    "mobilenet_v2_224_1.0_uint8": "uint8",
    "person_detect": "int8",
    "resnet_v2_101_1_default_1": "fp32",
    "resnet_v2_101_1_default_1_fp16": "fp16",
    "ssd_mobilenet_v2_static_1.0_int8": "int8",
    "ssd_mobilenet_v2_fpnlite_fp32": "fp32",
    "ssd_mobilenet_v2_fpnlite_fp32_fp16": "fp16",
    "ssd_mobilenet_v2_fpnlite_uint8": 'uint8',
}

# Column headers.
_MODEL = "model"
_DATA_TYPE = "data type"
_RUNTIME = "runtime"
_LATENCY = "latency (ms)"
_TASKSET = "taskset"
_MEMORY = "vmhwm (KB)"
_THREADS = "threads"
_CONFIG = "config"
_DRIVER = "driver/delegate"
_TFLITE_CONFIG = "TFLite config"
_IREE_CONFIG = "IREE config"
_IREE_LATENCY = "IREE latency (ms)"
_TFLITE_LATENCY = "TFLite latency (ms)"
_IREE_MEMORY = "IREE vmhwm (kb)"
_TFLITE_MEMORY = "TFLite vmhwm (kb)"

_IREE_VS_TFLITE_LATENCY = "IREE vs TFLite latency"
_IREE_VS_TFLITE_MEMORY = "IREE vs TFLite memory"

_PERF_COLUMNS = [_IREE_VS_TFLITE_LATENCY, _IREE_VS_TFLITE_MEMORY]
_NUMBER_COLUMNS = [_IREE_LATENCY, _TFLITE_LATENCY, _IREE_MEMORY, _TFLITE_MEMORY]
_CONFIG_COLUMNS = [_TFLITE_CONFIG, _IREE_CONFIG]


def get_tflite_model_list(df):
  """Retrieves the list of TFLite models, filtering out duplicates.

  The .csv file includes multiple entries of the same model but under a
  different configuration (e.g. XNNPack enabled, XNNPack disabled).
  """
  df = df.loc[df.runtime == "tflite"]
  # Remove rows where the model name ends with `noxnn` since this is a duplicate.
  df = df[~df.model.str.endswith("noxnn")]
  return df.model.unique()


def get_fastest_result(model, df):
  """Retrieves the lowest latency result from multiple configurations.

  Benchmarks are run under different configurations (e.g. number of threads,
  Big core, LITTLE core, etc). This method retrieves the fastest configuration
  whilst ensuring apples to apples comparisons (e.g. FP16 results are not
  considered when the model is FP32).

  Args:
    model: The model name.
    df: The dataframe to filter through.

  Returns:
    A dataframe containing the lowest latency.
  """
  df = df[df.model.str.startswith(model)]
  if not model.endswith("fp16"):
    df = df[~df[_MODEL].str.endswith("fp16")]
  df = df[df[_LATENCY] != 0]
  df = df[df[_LATENCY] == df[_LATENCY].min()]
  return df.head(1)


def get_tflite_config(model, df):
  """Generates a configuration string from TFLite config variables."""
  config = []
  if _TASKSET in df.columns:
    taskset = df.taskset.iloc[0]
    config.append(f"taskset {taskset}")
  threads = df.threads.iloc[0]
  config.append(f"{threads} threads" if threads > 1 else f"{threads} thread")
  config.append("no xnnpack" if model.endswith("noxnn") else "xnnpack")
  return ", ".join(config)


def generate_tflite_summary(dataframe):
  """Generates a dataframe containing the fastest TFLite result for each model."""
  summary = pd.DataFrame(columns=[_MODEL, _LATENCY, _MEMORY, _CONFIG])
  tflite_df = dataframe[dataframe.runtime == "tflite"]
  model_list = get_tflite_model_list(dataframe)
  for model in model_list:
    df = get_fastest_result(model, tflite_df)
    if df.empty:
      print(f"Warning: TFLite results invalid for {model}.")
      continue
    latency = df[_LATENCY].iloc[0]
    full_model_name = df.model.iloc[0]
    memory = df[_MEMORY].iloc[0]
    config = get_tflite_config(full_model_name, df)
    summary.loc[len(summary)] = [model, latency, memory, config]
  return summary


def get_iree_model_list(df):
  """Retrieves the list of IREE models, filtering out duplicates.

  The .csv file includes multiple entries of the same model but under a
  different configuration (e.g. mmt4d).
  """
  df = df.loc[df.runtime == "iree"]
  df = df[~df.model.str.endswith("mmt4d")]
  df = df[~df.model.str.endswith("padfuse")]
  return df.model.unique()


def get_iree_config(model, df):
  """Generates a configuration string from IREE config variables.

  The configuration is embedded in the model name.
  """
  config = []
  if _TASKSET in df.columns:
    taskset = df.taskset.iloc[0]
    config.append(f"taskset {taskset}")
  threads = df.threads.iloc[0]
  config.append(f"{threads} threads" if threads > 1 else f"{threads} thread")
  if model.endswith("im2col_mmt4d"):
    config.append("im2col")
    config.append("mmt4d")
  elif model.endswith("mmt4d"):
    config.append("mmt4d")
  elif model.endswith("padfuse"):
    config.append("fused pad")
  return ", ".join(config)


def generate_iree_summary(dataframe):
  """Generates a dataframe containing the fastest IREE result for each model."""
  summary = pd.DataFrame(columns=[_MODEL, _LATENCY, _MEMORY, _CONFIG])
  iree_df = dataframe[dataframe.runtime == "iree"]
  model_list = get_iree_model_list(dataframe)
  for model in model_list:
    df = get_fastest_result(model, iree_df)
    if df.empty:
      print(f"Warning: IREE results invalid for {model}.")
      continue
    latency = df[_LATENCY].iloc[0]
    full_model_name = df.model.iloc[0]
    memory = df[_MEMORY].iloc[0]
    config = get_iree_config(full_model_name, df)
    summary.loc[len(summary)] = [model, latency, memory, config]
  return summary


def get_common_html_style(df, title):
  """Returns HTML style attributes common to both server and mobile."""
  st = df.style.set_table_styles(html_utils.get_table_css())
  st = st.hide(axis="index")
  st = st.set_caption(title)
  st = st.set_properties(subset=[_MODEL],
                         **{
                             "width": "300px",
                             "text-align": "left",
                         })
  st = st.set_properties(subset=[_DATA_TYPE],
                         **{
                             "width": "100",
                             "text-align": "center",
                         })
  st = st.set_properties(subset=_NUMBER_COLUMNS,
                         **{
                             "width": "100",
                             "text-align": "right",
                         })
  st = st.set_properties(subset=_PERF_COLUMNS,
                         **{
                             "width": "150px",
                             "text-align": "right",
                             "color": "#ffffff"
                         })
  st = st.applymap(html_utils.style_latency, subset=[_IREE_VS_TFLITE_LATENCY])
  st = st.applymap(html_utils.style_memory, subset=[_IREE_VS_TFLITE_MEMORY])
  return st


def generate_summary(dataframe, title):
  """Generates a table comparing latencies and memory usage between IREE and TFLite.

  For each model, retrieves the lowest latency configuration from both IREE and TFLite.

  Args:
    dataframe: The raw data to summarize.
    title: The title of the table.

  Returns:
    An HTML string containing the summarized report.
  """
  summary = pd.DataFrame(columns=[
      _MODEL, _DATA_TYPE, _TFLITE_CONFIG, _IREE_CONFIG, _TFLITE_LATENCY,
      _IREE_LATENCY, _IREE_VS_TFLITE_LATENCY, _TFLITE_MEMORY, _IREE_MEMORY,
      _IREE_VS_TFLITE_MEMORY
  ])

  tflite_df = generate_tflite_summary(dataframe)
  iree_df = generate_iree_summary(dataframe)
  model_list = tflite_df[_MODEL].unique()

  for model in model_list:
    tflite_results = tflite_df[tflite_df.model == model]
    iree_results = iree_df[iree_df.model == model]

    if tflite_results.empty:
      print(f"Warning: No TFLite results found for model {model}")
      continue
    if iree_results.empty:
      print(f"Warning: No IREE results found for model {model}")
      continue

    iree_latency = iree_results[_LATENCY].iloc[0]
    tflite_latency = tflite_results[_LATENCY].iloc[0]
    latency_comparison = html_utils.format_latency_comparison(
        iree_latency, tflite_latency)

    iree_memory = iree_results[_MEMORY].iloc[0]
    tflite_memory = tflite_results[_MEMORY].iloc[0]
    memory_comparison = html_utils.format_memory_comparison(
        iree_memory, tflite_memory)

    iree_config = iree_results.config.iloc[0]
    tflite_config = tflite_results.config.iloc[0]
    summary.loc[len(summary)] = [
        model,
        _MODEL_TO_DATA_TYPE[model],
        tflite_config,
        iree_config,
        f"{tflite_latency:.1f}",
        f"{iree_latency:.1f}",
        latency_comparison,
        f"{tflite_memory:,.0f}",
        f"{iree_memory:,.0f}",
        memory_comparison,
    ]

  summary = summary.round(2)
  st = get_common_html_style(summary, title)
  st = st.set_properties(subset=_CONFIG_COLUMNS,
                         **{
                             "width": "300px",
                             "text-align": "left",
                         })
  return st.to_html().replace("\\n", "<br>") + "<br/>"


def generate_detail(dataframe, title, platform):
  """Generates a table comparing latencies and memory usage between IREE and TFLite.

  The table generated is more detailed than `generate_summary`. It lists latencies
  of all IREE configurations, using the fastest TFLite configuration as baseline.

  Args:
    dataframe: The raw data to summarize.
    title: The title of the table.
    platform: Either `server` or `mobile`.

  Returns:
    An HTML string containing the detailed report.
  """
  summary = pd.DataFrame(columns=[
      _MODEL, _DATA_TYPE, _TFLITE_CONFIG, _IREE_CONFIG, _TASKSET, _THREADS,
      _TFLITE_LATENCY, _IREE_LATENCY, _IREE_VS_TFLITE_LATENCY, _TFLITE_MEMORY,
      _IREE_MEMORY, _IREE_VS_TFLITE_MEMORY
  ])

  model_list = get_tflite_model_list(dataframe)
  for model in model_list:
    df = dataframe[dataframe.model.str.startswith(model)]
    # If result does not use FP16, remove FP16 results from dataframe to
    # maintain apples-to-apples comparisons.
    if not model.endswith("fp16"):
      df = df[~df.model.str.endswith("fp16")]

    if _TASKSET in df.columns:
      tasksets = df.taskset.unique()
    else:
      tasksets = ["none"]

    for taskset in tasksets:
      per_taskset_df = df if taskset == "none" else df[df.taskset == taskset]
      threads = per_taskset_df.threads.unique()
      for thread in threads:
        per_thread_df = per_taskset_df[per_taskset_df.threads == thread]
        tflite_df = get_fastest_result(
            model, per_thread_df[per_thread_df.runtime == "tflite"])
        if tflite_df.empty:
          continue

        tflite_latency = tflite_df[_LATENCY].iloc[0]
        tflite_memory = tflite_df[_MEMORY].iloc[0]
        if tflite_latency == 0 or tflite_memory == 0:
          continue

        full_model_name = tflite_df.model.iloc[0]
        # For TFLite config, we only want to know if XNNPack was used. The other
        # configuration settings are covered in other columns.
        tflite_config = "no xnnpack" if full_model_name.endswith(
            "noxnn") else "xnnpack"

        iree_df = per_thread_df[per_thread_df.runtime == "iree"]
        for _, row in iree_df.iterrows():
          iree_config = row[_DRIVER]
          model_name = row[_MODEL]
          if model_name.endswith("im2col_mmt4d"):
            iree_config += ", im2col, mmt4d"
          elif model_name.endswith("mmt4d"):
            iree_config += ", mmt4d"
          elif model_name.endswith("padfuse"):
            iree_config += ", fused pad"

          iree_latency = row[_LATENCY]
          latency_comparison = html_utils.format_latency_comparison(
              iree_latency, tflite_latency)
          iree_memory = row[_MEMORY]
          memory_comparison = html_utils.format_memory_comparison(
              iree_memory, tflite_memory)

          if iree_latency == 0 or iree_memory == 0:
            continue

          summary.loc[len(summary)] = [
              model, _MODEL_TO_DATA_TYPE[model], tflite_config, iree_config,
              taskset, thread, f"{tflite_latency:.1f}", f"{iree_latency:.1f}",
              latency_comparison, f"{tflite_memory:,.0f}",
              f"{iree_memory:,.0f}", memory_comparison
          ]

  summary = summary.round(2)
  st = get_common_html_style(summary, title)
  st = st.set_properties(subset=[_TASKSET, _THREADS],
                         **{
                             "width": "100",
                             "text-align": "center",
                         })
  st = st.set_properties(subset=[_TFLITE_CONFIG],
                         **{
                             "width": "150px",
                             "text-align": "left",
                         })
  st = st.set_properties(subset=[_IREE_CONFIG],
                         **{
                             "width": "300px",
                             "text-align": "left",
                         })
  if platform != "mobile":
    st.hide_columns(subset=[_TASKSET])

  return st.to_html().replace("\\n", "<br>") + "<br/>"


def main(args):
  """Summarizes IREE vs TFLite benchmark results."""
  if args.platform == _PLATFORM_SERVER:
    cpu_drivers = ["cpu", "local-task"]
    gpu_drivers = ["gpu", "cuda"]
  else:
    cpu_drivers = ["cpu", "local-task"]
    gpu_drivers = ["gpu", "vulkan", "adreno"]

  version_html = (f"<i>IREE version: {args.iree_version}</i><br/>"
                  f"<i>TFlite version: {args.tflite_version}</i><br/>"
                  f"<i>last updated: {date.today().isoformat()}</i><br/><br/>")
  html = html_utils.generate_header_and_legend(version_html)

  df = pd.read_csv(args.input_csv)

  # Generate CPU Summary.
  results = df[df[_DRIVER].isin(cpu_drivers)]
  html += generate_summary(results, args.platform.capitalize() + " CPU Summary")

  # Generate GPU Summary.
  results = df[df[_DRIVER].isin(gpu_drivers)]
  html += generate_summary(results, args.platform.capitalize() + " GPU Summary")

  # Generate CPU Detailed View.
  results = df[df[_DRIVER].isin(cpu_drivers)]
  html += generate_detail(results,
                          args.platform.capitalize() + " CPU Detailed",
                          args.platform)

  # Generate GPU Detailed View.
  results = df[df[_DRIVER].isin(gpu_drivers)]
  html += generate_detail(results,
                          args.platform.capitalize() + " GPU Detailed",
                          args.platform)

  args.output_path.write_text(html)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--iree_version",
                      type=str,
                      default=None,
                      required=True,
                      help="The IREE version.")
  parser.add_argument("--tflite_version",
                      type=str,
                      default=None,
                      required=True,
                      help="The TFLite version.")
  parser.add_argument(
      "--platform",
      action="store",
      type=str.lower,
      help=
      "The platform the models were benchmarked on. Either server or mobile.",
      required=True,
      choices=[_PLATFORM_SERVER, _PLATFORM_MOBILE])
  parser.add_argument(
      "--input_csv",
      type=str,
      default=None,
      help=
      "The path to the csv file containing benchmark results for both IREE and TFLite."
  )
  parser.add_argument(
      "--output_path",
      type=pathlib.Path,
      default="/tmp/summary.html",
      help="The path to the output html file that summarizes results.")
  return parser.parse_args()


if __name__ == "__main__":
  main(parse_args())
