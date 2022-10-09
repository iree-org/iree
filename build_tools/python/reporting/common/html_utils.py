# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pandas as pd

_LEGEND_0 = ">10.0x better"
_LEGEND_1 = ">2.0x, <=10.0x better"
_LEGEND_2 = ">=1.0x, <=2.0x better"
_LEGEND_3 = ">1.0x, <1.15x worse"
_LEGEND_4 = ">=1.15x, <2.0x worse"
_LEGEND_5 = ">=2.0x, <10.x worse"
_LEGEND_6 = ">=10.0x worse"


def get_table_css():
  styles = [
      dict(selector="tr:hover", props=[("background", "#f4f4f4")]),
      dict(selector="tbody tr", props=[("background-color", "#ffffff")]),
      dict(selector="tbody td", props=[("border", "1px solid #dddfe1")]),
      dict(selector="th",
           props=[("background-color", "#54585d"), ("color", "#ffffff"),
                  ("font-weight", "bold"), ("border", "1px solid #54585d"),
                  ("padding", "10px")]),
      dict(selector="td", props=[("padding", "10px")]),
      dict(selector="",
           props=[("border-collapse", "collapse"),
                  ("font-family", "Tahoma, Geneva, sans-serif")]),
      dict(selector="caption",
           props=[("text-align", "center"), ("padding", "10px"),
                  ("font-weight", "bold"), ("font-size", "1.2em"),
                  ("color", "#636363")]),
  ]
  return styles


def style_legend(v):
  if _LEGEND_0 in v:
    props = "background-color: #0277BD;"
  elif _LEGEND_1 in v:
    props = "background-color: #2E7D32;"
  elif _LEGEND_2 in v:
    props = "background-color: #66BB6A;"
  elif _LEGEND_3 in v:
    props = "background-color: #FBC02D;"
  elif _LEGEND_4 in v:
    props = "background-color: #E57373;"
  elif _LEGEND_5 in v:
    props = "background-color: #C62828;"
  else:
    props = "background-color: #880E4F"
  return props


def generate_header_and_legend(version_html):
  html = "<style type='text/css'>:root { font-family: Tahoma, Geneva, sans-serif; color: #636363; } h3 {text-align: center; }</style>"
  html = html + version_html

  legend = pd.DataFrame(columns=[""])
  legend.loc[len(legend)] = [_LEGEND_0]
  legend.loc[len(legend)] = [_LEGEND_1]
  legend.loc[len(legend)] = [_LEGEND_2]
  legend.loc[len(legend)] = [_LEGEND_3]
  legend.loc[len(legend)] = [_LEGEND_4]
  legend.loc[len(legend)] = [_LEGEND_5]
  legend.loc[len(legend)] = [_LEGEND_6]

  styled_legend = legend.style.set_table_styles(get_table_css())
  styled_legend.set_caption("Legend")
  styled_legend = styled_legend.set_properties(**{"color": "#ffffff"})
  styled_legend = styled_legend.set_properties(**{"width": "200px"})
  styled_legend = styled_legend.applymap(style_legend)
  styled_legend = styled_legend.hide(axis="index")
  styled_legend = styled_legend.hide(axis="columns")
  html = html + styled_legend.to_html() + "<br/>"
  return html


def style_speedup(v):
  if v > 10.0:
    props = "background-color: #0277BD;"
  elif v > 2.0:
    props = "background-color: #2E7D32;"
  elif v >= 1.0:
    props = "background-color: #66BB6A;"
  else:
    props = "background-color: #FBC02D;"
  return props


def style_slowdown(v):
  if v >= 10.0:
    props = "background-color: #880E4F"
  elif v >= 2.0:
    props = "background-color: #C62828;"
  elif v > 1.15:
    props = "background-color: #E57373;"
  else:
    props = "background-color: #FBC02D;"
  return props


def style_performance(v):
  if "faster" in v:
    return style_speedup(float(v.split("x")[0]))
  else:
    return style_slowdown(float(v.split("x")[0]))


def style_latency(v):
  if v == "nan":
    return "color: #636363"
  if "faster" in v:
    return style_speedup(float(v.split("x")[0]))
  else:
    return style_slowdown(float(v.split("x")[0]))


def style_memory(v):
  if v == "nan":
    return "color: #636363"
  if "smaller" in v:
    return style_speedup(float(v.split("x")[0]))
  else:
    return style_slowdown(float(v.split("x")[0]))


def format_latency_comparison(iree_latency, baseline_latency):
  if iree_latency == 0 or baseline_latency == 0:
    return "nan"

  speedup = baseline_latency / iree_latency
  slowdown = iree_latency / baseline_latency
  faster_label = "{:.2f}x faster"
  slower_label = "{:.2f}x slower"
  latency = faster_label.format(
      speedup) if speedup >= 1.0 else slower_label.format(slowdown)
  return latency


def format_memory_comparison(iree_memory, baseline_memory):
  if iree_memory == 0 or baseline_memory == 0:
    return "nan"

  smaller = baseline_memory / iree_memory
  larger = iree_memory / baseline_memory
  smaller_label = "{:.2f}x smaller"
  larger_label = "{:0.2f}x larger"
  memory = smaller_label.format(
      smaller) if smaller >= 1.0 else larger_label.format(larger)
  return memory
