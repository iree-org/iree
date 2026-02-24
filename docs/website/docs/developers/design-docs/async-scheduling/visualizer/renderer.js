// SVG-based renderer for the vector clock explainer.
//
// Produces two coordinated views:
//   1. DAG View — topological layout of the operation dependency graph.
//      Nodes are operations, edges are semaphore wait/signal pairs.
//   2. Execution Timeline — hardware lanes as rows, operations as bars over
//   time.
//      Shows WHERE and WHEN operations run on physical hardware.
//
// Both views update together when the tick changes: node/bar colors reflect
// operation state, and a time cursor sweeps the timeline.

import {computeDependencies, frontierToString} from './simulator.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

// Read the current theme colors from CSS custom properties. Called at renderer
// creation time so SVG elements pick up the active palette (light or dark).
// SVG attributes cannot use var() directly, so we snapshot the computed values
// and apply them as literal attribute strings.
export function readThemeColors() {
  const style = getComputedStyle(document.body);
  const get = (name) => style.getPropertyValue(name).trim();
  return {
    state: {
      pending: {
        fill: get('--state-pending-fill'),
        stroke: get('--state-pending-stroke'),
        text: get('--state-pending-text')
      },
      ready: {
        fill: get('--state-ready-fill'),
        stroke: get('--state-ready-stroke'),
        text: get('--state-ready-text')
      },
      in_flight: {
        fill: get('--state-in-flight-fill'),
        stroke: get('--state-in-flight-stroke'),
        text: get('--state-in-flight-text')
      },
      retired: {
        fill: get('--state-retired-fill'),
        stroke: get('--state-retired-stroke'),
        text: get('--state-retired-text')
      },
    },
    arrow_sat: get('--arrow-sat'),
    arrow_unsat: get('--arrow-unsat'),
    arrow_hl: get('--arrow-hl'),
    lane_even: get('--lane-even'),
    lane_odd: get('--lane-odd'),
    grid_line: get('--grid-line'),
    grid_label: get('--grid-label'),
    lane_label: get('--lane-label'),
    hw_sublabel: get('--hw-sublabel'),
    edge_label: get('--edge-label'),
    cursor: get('--cursor-color'),
  };
}

// ---------------------------------------------------------------------------
// SVG helpers
// ---------------------------------------------------------------------------

function el(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, String(value));
  }
  return node;
}

function textEl(content, attrs = {}) {
  const node = el('text', attrs);
  node.textContent = content;
  return node;
}

// ---------------------------------------------------------------------------
// DAG layout — topological layering
// ---------------------------------------------------------------------------

// Assign each operation to a layer based on its longest dependency path.
// Sources (no waits) go in layer 0. Each dependent is max(parent layers) + 1.
function computeLayers(scenario, deps) {
  const layers = {};
  for (const op of scenario.operations) {
    layers[op.id] = 0;
  }

  // Build parent map: op_id -> [parent_op_ids].
  const parents = {};
  for (const op of scenario.operations) {
    parents[op.id] = [];
  }
  for (const dep of deps) {
    parents[dep.to].push(dep.from);
  }

  // Iterative longest-path computation.
  let changed = true;
  while (changed) {
    changed = false;
    for (const op of scenario.operations) {
      for (const parent_id of parents[op.id]) {
        const candidate = layers[parent_id] + 1;
        if (candidate > layers[op.id]) {
          layers[op.id] = candidate;
          changed = true;
        }
      }
    }
  }

  return layers;
}

// ---------------------------------------------------------------------------
// DAG View
// ---------------------------------------------------------------------------

const DAG = {
  NODE_W: 110,
  NODE_H: 40,
  LAYER_GAP: 150,
  NODE_GAP: 56,
  PAD_X: 30,
  PAD_Y: 24,
};

function buildDagSvg(scenario, deps, layers, state, theme) {
  // Group operations by layer.
  const layer_groups = {};
  let max_layer = 0;
  for (const op of scenario.operations) {
    const layer = layers[op.id];
    if (!layer_groups[layer]) layer_groups[layer] = [];
    layer_groups[layer].push(op);
    if (layer > max_layer) max_layer = layer;
  }

  const num_layers = max_layer + 1;
  const max_per_layer =
      Math.max(...Object.values(layer_groups).map(g => g.length));

  const content_w = num_layers * DAG.LAYER_GAP;
  const content_h = max_per_layer * DAG.NODE_GAP;
  const svg_w = DAG.PAD_X * 2 + content_w;
  const svg_h = DAG.PAD_Y * 2 + Math.max(content_h, DAG.NODE_H + 20);

  // Compute node positions — centered vertically within the SVG.
  const positions = {};
  for (let layer = 0; layer <= max_layer; layer++) {
    const ops = layer_groups[layer] || [];
    const total_h = ops.length * DAG.NODE_GAP;
    const start_y = (svg_h - total_h) / 2 + (DAG.NODE_GAP - DAG.NODE_H) / 2;
    for (let i = 0; i < ops.length; i++) {
      positions[ops[i].id] = {
        x: DAG.PAD_X + layer * DAG.LAYER_GAP,
        y: start_y + i * DAG.NODE_GAP,
      };
    }
  }

  const svg = el('svg', {
    width: svg_w,
    height: svg_h,
    viewBox: `0 0 ${svg_w} ${svg_h}`,
  });

  // Defs: arrowhead markers.
  const defs = el('defs');
  for (const [id, color] of [
           ['dag-arrow-sat', theme.arrow_sat],
           ['dag-arrow-unsat', theme.arrow_unsat],
           ['dag-arrow-hl', theme.arrow_hl],
  ]) {
    const marker = el('marker', {
      id,
      markerWidth: 8,
      markerHeight: 6,
      refX: 8,
      refY: 3,
      orient: 'auto',
    });
    marker.appendChild(el('polygon', {
      points: '0 0, 8 3, 0 6',
      fill: color,
    }));
    defs.appendChild(marker);
  }
  svg.appendChild(defs);

  // Draw edges (behind nodes).
  state.dag_edges = [];
  for (const dep of deps) {
    const from_pos = positions[dep.from];
    const to_pos = positions[dep.to];

    const x1 = from_pos.x + DAG.NODE_W;
    const y1 = from_pos.y + DAG.NODE_H / 2;
    const x2 = to_pos.x;
    const y2 = to_pos.y + DAG.NODE_H / 2;
    const mx = (x1 + x2) / 2;

    const path_d = `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;

    const path = el('path', {
      d: path_d,
      fill: 'none',
      stroke: theme.arrow_unsat,
      'stroke-width': 1.5,
      'stroke-dasharray': '6,4',
      'marker-end': 'url(#dag-arrow-unsat)',
    });
    svg.appendChild(path);

    // Wider invisible hit target for hover.
    const hit = el('path', {
      d: path_d,
      fill: 'none',
      stroke: 'transparent',
      'stroke-width': 14,
      cursor: 'pointer',
    });
    svg.appendChild(hit);

    // Edge label at midpoint.
    const sem = scenario.semaphores.find(s => s.id === dep.semaphore);
    const label_text = `${sem ? sem.label : dep.semaphore}:${dep.value}`;
    const label_x = mx;
    const label_y = (y1 + y2) / 2 - 6;
    svg.appendChild(textEl(label_text, {
      x: label_x,
      y: label_y,
      'text-anchor': 'middle',
      'font-size': 9,
      fill: theme.edge_label,
      'font-family': 'var(--mono)',
    }));

    setup_edge_tooltip(hit, dep, scenario);
    state.dag_edges.push({path, dep});
  }

  // Draw nodes.
  state.dag_nodes = {};
  for (const op of scenario.operations) {
    const pos = positions[op.id];
    const group = el('g', {cursor: 'pointer'});

    const rect = el('rect', {
      x: pos.x,
      y: pos.y,
      width: DAG.NODE_W,
      height: DAG.NODE_H,
      rx: 6,
      fill: theme.state.pending.fill,
      stroke: theme.state.pending.stroke,
      'stroke-width': 1.5,
    });
    group.appendChild(rect);

    const label = textEl(op.label, {
      x: pos.x + DAG.NODE_W / 2,
      y: pos.y + 16,
      'text-anchor': 'middle',
      'font-size': 12,
      'font-weight': 600,
      fill: theme.state.pending.text,
      'font-family': 'var(--sans)',
    });
    group.appendChild(label);

    // Hardware sublabel.
    const hw = scenario.hardware.find(h => h.id === op.hardware);
    group.appendChild(textEl(hw ? hw.label : op.hardware, {
      x: pos.x + DAG.NODE_W / 2,
      y: pos.y + 30,
      'text-anchor': 'middle',
      'font-size': 9,
      fill: theme.hw_sublabel,
      'font-family': 'var(--sans)',
    }));

    svg.appendChild(group);
    state.dag_nodes[op.id] = {rect, label, group};
  }

  // Highlight group — sits above nodes so highlighted edges are fully visible
  // even when the graph is dense. pointer-events: none lets clicks pass
  // through.
  const dag_hl_group = el('g', {'pointer-events': 'none'});
  svg.appendChild(dag_hl_group);
  state.dag_hl_group = dag_hl_group;

  // Wire up tooltips for nodes.
  for (const op of scenario.operations) {
    setup_op_tooltip(state.dag_nodes[op.id].group, op, state, scenario);
  }

  return svg;
}

// ---------------------------------------------------------------------------
// Execution Timeline View
// ---------------------------------------------------------------------------

const TL = {
  LABEL_W: 120,
  TICK_W: 50,
  LANE_H: 44,
  LANE_GAP: 8,
  BAR_H: 32,
  HEADER_H: 24,
  PAD: 16,
};

function buildTimelineSvg(scenario, final_snapshot, max_tick, state, theme) {
  const hw_lanes = scenario.hardware;
  const num_lanes = hw_lanes.length;

  // Adaptive tick width. Floor at 26px — below that, arrows become unreadable.
  // The container scrolls horizontally for wide timelines.
  const tick_w = max_tick <= 14 ? 50 :
      max_tick <= 24            ? 42 :
      max_tick <= 40            ? 34 :
                                  26;

  const svg_w = TL.LABEL_W + (max_tick + 1) * tick_w + TL.PAD;
  const svg_h = TL.HEADER_H + num_lanes * (TL.LANE_H + TL.LANE_GAP) -
      TL.LANE_GAP + TL.PAD;

  const svg = el('svg', {
    width: svg_w,
    height: svg_h,
    viewBox: `0 0 ${svg_w} ${svg_h}`,
  });

  const defs = el('defs');
  svg.appendChild(defs);

  const lanes_bottom =
      TL.HEADER_H + num_lanes * (TL.LANE_H + TL.LANE_GAP) - TL.LANE_GAP;

  // Lane backgrounds.
  for (let i = 0; i < num_lanes; i++) {
    const y = TL.HEADER_H + i * (TL.LANE_H + TL.LANE_GAP);
    svg.appendChild(el('rect', {
      x: TL.LABEL_W,
      y,
      width: (max_tick + 1) * tick_w,
      height: TL.LANE_H,
      fill: i % 2 === 0 ? theme.lane_even : theme.lane_odd,
      rx: 4,
    }));
  }

  // Time grid lines and labels.
  for (let t = 0; t <= max_tick; t++) {
    const x = TL.LABEL_W + t * tick_w;
    svg.appendChild(el('line', {
      x1: x,
      y1: TL.HEADER_H - 4,
      x2: x,
      y2: lanes_bottom,
      stroke: theme.grid_line,
      'stroke-width': 1,
    }));
    svg.appendChild(textEl(String(t), {
      x: x + tick_w / 2,
      y: TL.HEADER_H - 8,
      'text-anchor': 'middle',
      'font-size': 10,
      fill: theme.grid_label,
      'font-family': 'var(--sans)',
    }));
  }

  // Lane labels.
  for (let i = 0; i < num_lanes; i++) {
    const y = TL.HEADER_H + i * (TL.LANE_H + TL.LANE_GAP) + TL.LANE_H / 2 + 4;
    svg.appendChild(textEl(hw_lanes[i].label, {
      x: TL.LABEL_W - 10,
      y,
      'text-anchor': 'end',
      'font-size': 12,
      'font-weight': 500,
      fill: theme.lane_label,
      'font-family': 'var(--sans)',
    }));
  }

  // Build hardware index for Y positioning.
  const hw_index = {};
  hw_lanes.forEach((hw, i) => {
    hw_index[hw.id] = i;
  });

  // Inset for operation bars — scales with tick width for narrow scenarios.
  const OP_INSET = Math.min(16, Math.floor(tick_w * 0.3));

  // Dependency arrows between operations on the timeline.
  const tl_deps = computeDependencies(scenario);
  state.tl_edges = [];
  for (const dep of tl_deps) {
    const from_state = final_snapshot.operations[dep.from];
    const to_state = final_snapshot.operations[dep.to];
    if (from_state.end_tick === null || to_state.start_tick === null) continue;

    const from_lane = hw_index[from_state.assigned_lane];
    const to_lane = hw_index[to_state.assigned_lane];

    // Connect from right edge of source bar to left edge of target bar.
    const x1 = TL.LABEL_W + from_state.end_tick * tick_w - OP_INSET;
    const y1 =
        TL.HEADER_H + from_lane * (TL.LANE_H + TL.LANE_GAP) + TL.LANE_H / 2;
    const x2 = TL.LABEL_W + to_state.start_tick * tick_w + OP_INSET;
    const y2 =
        TL.HEADER_H + to_lane * (TL.LANE_H + TL.LANE_GAP) + TL.LANE_H / 2;
    const mx = (x1 + x2) / 2;

    const path_d = `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
    const path = el('path', {
      d: path_d,
      fill: 'none',
      stroke: theme.arrow_unsat,
      'stroke-width': 1.5,
      'stroke-dasharray': '6,4',
      'marker-end': 'url(#tl-arrow-unsat)',
    });
    svg.appendChild(path);
    state.tl_edges.push({path, dep});
  }

  // Arrow markers for timeline.
  for (const [id, color] of [
           ['tl-arrow-sat', theme.arrow_sat],
           ['tl-arrow-unsat', theme.arrow_unsat],
           ['tl-arrow-hl', theme.arrow_hl],
  ]) {
    const marker = el('marker', {
      id,
      markerWidth: 8,
      markerHeight: 6,
      refX: 8,
      refY: 3,
      orient: 'auto',
    });
    marker.appendChild(el('polygon', {
      points: '0 0, 8 3, 0 6',
      fill: color,
    }));
    defs.appendChild(marker);
  }

  // Operation bars.
  state.tl_bars = {};
  for (const op of scenario.operations) {
    const op_state = final_snapshot.operations[op.id];
    if (op_state.start_tick === null) continue;

    const lane = hw_index[op_state.assigned_lane];
    const x = TL.LABEL_W + op_state.start_tick * tick_w + OP_INSET;
    const y = TL.HEADER_H + lane * (TL.LANE_H + TL.LANE_GAP) +
        (TL.LANE_H - TL.BAR_H) / 2;
    const w = op.duration * tick_w - OP_INSET * 2;

    const group = el('g', {cursor: 'pointer'});

    // Clip text to bar bounds.
    const clip_id = `clip-tl-${op.id}`;
    const clip = el('clipPath', {id: clip_id});
    clip.appendChild(
        el('rect', {x: x + 4, y, width: Math.max(0, w - 8), height: TL.BAR_H}));
    defs.appendChild(clip);

    const rect = el('rect', {
      x,
      y,
      width: w,
      height: TL.BAR_H,
      rx: 5,
      fill: theme.state.pending.fill,
      stroke: theme.state.pending.stroke,
      'stroke-width': 1.5,
    });
    group.appendChild(rect);

    const font_size = w > 100 ? 11 : w > 60 ? 10 : w > 35 ? 9 : 8;
    const label = textEl(op.label, {
      x: x + w / 2,
      y: y + TL.BAR_H / 2 + 4,
      'text-anchor': 'middle',
      'font-size': font_size,
      'font-weight': 500,
      fill: theme.state.pending.text,
      'font-family': 'var(--sans)',
      'clip-path': `url(#${clip_id})`,
    });
    group.appendChild(label);

    svg.appendChild(group);
    state.tl_bars[op.id] = {rect, label, group};

    setup_op_tooltip(group, op, state, scenario);
  }

  // Highlight group — sits above bars so highlighted edges are fully visible.
  const tl_hl_group = el('g', {'pointer-events': 'none'});
  svg.appendChild(tl_hl_group);
  state.tl_hl_group = tl_hl_group;

  // Time cursor.
  state.tl_cursor = el('line', {
    x1: TL.LABEL_W,
    y1: TL.HEADER_H - 4,
    x2: TL.LABEL_W,
    y2: lanes_bottom,
    stroke: theme.cursor,
    'stroke-width': 2,
    'stroke-opacity': 0.8,
    'pointer-events': 'none',
  });
  svg.appendChild(state.tl_cursor);

  state.tl_tick_w = tick_w;

  return svg;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Create a renderer for a scenario.
//
// When base_scenario is provided (scaling active), the DAG is built from the
// base scenario in static mode — all nodes show "retired" colors and don't
// animate with the time cursor. The timeline uses the (scaled) scenario.
export function createRenderer(
    dag_container, timeline_container, scenario, snapshots, base_scenario) {
  dag_container.innerHTML = '';
  timeline_container.innerHTML = '';

  const theme = readThemeColors();

  const dag_scenario = base_scenario || scenario;
  const dag_static = !!base_scenario;

  // Shared mutable state — populated by the build functions.
  const state = {
    snapshots,
    theme,
    current_tick: 0,
    dag_nodes: {},
    dag_edges: [],
    dag_hl_group: null,
    tl_bars: {},
    tl_edges: [],
    tl_hl_group: null,
    tl_cursor: null,
    tl_tick_w: TL.TICK_W,
  };

  // Build DAG from the base scenario (or the scenario itself at 1x1).
  const dag_deps = computeDependencies(dag_scenario);
  const dag_layers = computeLayers(dag_scenario, dag_deps);
  const dag_svg = buildDagSvg(dag_scenario, dag_deps, dag_layers, state, theme);
  dag_container.appendChild(dag_svg);

  // When the DAG is static (scaled mode), set all nodes to retired appearance
  // and all edges to satisfied. This shows the dependency structure without
  // pretending to track per-tick state.
  if (dag_static) {
    const retired_colors = theme.state.retired;
    for (const op of dag_scenario.operations) {
      const node = state.dag_nodes[op.id];
      node.rect.setAttribute('fill', retired_colors.fill);
      node.rect.setAttribute('stroke', retired_colors.stroke);
      node.rect.removeAttribute('stroke-dasharray');
      node.rect.setAttribute('opacity', '1');
      node.label.setAttribute('fill', retired_colors.text);
    }
    for (const {path} of state.dag_edges) {
      path.setAttribute('stroke', theme.arrow_sat);
      path.setAttribute('stroke-dasharray', 'none');
      path.setAttribute('marker-end', 'url(#dag-arrow-sat)');
    }
  }

  const max_tick = snapshots.length - 1;
  const final_snapshot = snapshots[max_tick];
  const tl_svg =
      buildTimelineSvg(scenario, final_snapshot, max_tick, state, theme);
  timeline_container.appendChild(tl_svg);

  // Wire up hover-to-highlight for edges connected to each node/bar.
  const dag_edge_map = buildEdgeMap(state.dag_edges);
  for (const op of dag_scenario.operations) {
    const node = state.dag_nodes[op.id];
    if (node) {
      setupEdgeHighlight(
          node.group, op.id, dag_edge_map, state.dag_hl_group, 'dag-arrow-hl',
          theme.arrow_hl);
    }
  }

  const tl_edge_map = buildEdgeMap(state.tl_edges);
  for (const op of scenario.operations) {
    const bar = state.tl_bars[op.id];
    if (bar) {
      setupEdgeHighlight(
          bar.group, op.id, tl_edge_map, state.tl_hl_group, 'tl-arrow-hl',
          theme.arrow_hl);
    }
  }

  return {
    setTick(tick) {
      state.current_tick = tick;
      const snapshot = snapshots[Math.min(tick, snapshots.length - 1)];

      // Update DAG node colors — skip when DAG is static (scaled mode).
      if (!dag_static) {
        for (const op of dag_scenario.operations) {
          const op_state = snapshot.operations[op.id].state;
          const colors = theme.state[op_state];
          const node = state.dag_nodes[op.id];
          node.rect.setAttribute('fill', colors.fill);
          node.rect.setAttribute('stroke', colors.stroke);
          node.label.setAttribute('fill', colors.text);

          if (op_state === 'pending') {
            node.rect.setAttribute('stroke-dasharray', '4,3');
            node.rect.setAttribute('opacity', '0.5');
          } else if (op_state === 'ready') {
            node.rect.setAttribute('stroke-dasharray', '4,3');
            node.rect.setAttribute('opacity', '0.75');
          } else {
            node.rect.removeAttribute('stroke-dasharray');
            node.rect.setAttribute('opacity', '1');
          }
        }

        // Update DAG edge styles.
        for (const {path, dep} of state.dag_edges) {
          const from_state = snapshot.operations[dep.from].state;
          const satisfied = from_state === 'retired';
          path.setAttribute(
              'stroke', satisfied ? theme.arrow_sat : theme.arrow_unsat);
          path.setAttribute('stroke-dasharray', satisfied ? 'none' : '6,4');
          path.setAttribute(
              'marker-end',
              satisfied ? 'url(#dag-arrow-sat)' : 'url(#dag-arrow-unsat)');
        }
      }

      // Update timeline bar colors.
      for (const op of scenario.operations) {
        const bar = state.tl_bars[op.id];
        if (!bar) continue;
        const op_state = snapshot.operations[op.id].state;
        const colors = theme.state[op_state];
        bar.rect.setAttribute('fill', colors.fill);
        bar.rect.setAttribute('stroke', colors.stroke);
        bar.label.setAttribute('fill', colors.text);

        if (op_state === 'pending') {
          bar.rect.setAttribute('stroke-dasharray', '4,3');
          bar.rect.setAttribute('opacity', '0.45');
        } else if (op_state === 'ready') {
          bar.rect.setAttribute('stroke-dasharray', '4,3');
          bar.rect.setAttribute('opacity', '0.7');
        } else {
          bar.rect.removeAttribute('stroke-dasharray');
          bar.rect.setAttribute('opacity', '1');
        }
      }

      // Update timeline edge styles.
      for (const {path, dep} of state.tl_edges) {
        const from_state = snapshot.operations[dep.from].state;
        const satisfied = from_state === 'retired';
        path.setAttribute(
            'stroke', satisfied ? theme.arrow_sat : theme.arrow_unsat);
        path.setAttribute('stroke-dasharray', satisfied ? 'none' : '6,4');
        path.setAttribute(
            'marker-end',
            satisfied ? 'url(#tl-arrow-sat)' : 'url(#tl-arrow-unsat)');
      }

      // Move timeline cursor.
      const cursor_x = TL.LABEL_W + tick * state.tl_tick_w;
      state.tl_cursor.setAttribute('x1', cursor_x);
      state.tl_cursor.setAttribute('x2', cursor_x);
    },
  };
}

// ---------------------------------------------------------------------------
// Hover edge highlighting
// ---------------------------------------------------------------------------

// Build a map from operation ID to all edges touching that operation.
function buildEdgeMap(edges) {
  const map = {};
  for (const entry of edges) {
    const from_id = entry.dep.from;
    const to_id = entry.dep.to;
    if (!map[from_id]) map[from_id] = [];
    if (!map[to_id]) map[to_id] = [];
    map[from_id].push(entry);
    map[to_id].push(entry);
  }
  return map;
}

// Attach hover handlers to a node/bar group that highlight connected edges.
// On mouseenter, clones connected edge paths into the highlight group with
// bright styling. On mouseleave, clears all clones.
function setupEdgeHighlight(
    group, op_id, edge_map, hl_group, marker_id, hl_color) {
  const entries = edge_map[op_id];
  if (!entries || entries.length === 0) return;

  group.addEventListener('mouseenter', () => {
    for (const entry of entries) {
      const clone = entry.path.cloneNode(false);
      clone.setAttribute('stroke', hl_color);
      clone.setAttribute('stroke-width', '1.5');
      clone.setAttribute('stroke-dasharray', 'none');
      clone.setAttribute('stroke-opacity', '0.9');
      clone.setAttribute('marker-end', `url(#${marker_id})`);
      hl_group.appendChild(clone);
    }
  });

  group.addEventListener('mouseleave', () => {
    while (hl_group.firstChild) {
      hl_group.removeChild(hl_group.firstChild);
    }
  });
}

// ---------------------------------------------------------------------------
// Tooltip helpers
// ---------------------------------------------------------------------------

function get_tooltip() {
  return document.getElementById('tooltip');
}

function show_tooltip(html, event) {
  const tt = get_tooltip();
  tt.innerHTML = html;
  tt.classList.remove('hidden');
  position_tooltip(tt, event);
}

function hide_tooltip() {
  get_tooltip().classList.add('hidden');
}

function position_tooltip(tt, event) {
  const margin = 16;
  let left = event.clientX + margin;
  let top = event.clientY;

  const rect = tt.getBoundingClientRect();
  if (left + rect.width > window.innerWidth - margin) {
    left = event.clientX - rect.width - margin;
  }
  if (top + rect.height > window.innerHeight - margin) {
    top = window.innerHeight - rect.height - margin;
  }

  tt.style.left = left + 'px';
  tt.style.top = top + 'px';
}

function setup_op_tooltip(group, op, state, scenario) {
  group.addEventListener('mouseenter', (event) => {
    const tick = state.current_tick;
    const snapshot =
        state.snapshots[Math.min(tick, state.snapshots.length - 1)];
    const op_snap = snapshot.operations[op.id];

    // Show the assigned lane if issued, otherwise the hardware type.
    const assigned = op_snap.assigned_lane;
    const hw = assigned ? scenario.hardware.find(h => h.id === assigned) :
                          scenario.hardware.find(h => h.id === op.hardware);
    let html = `<strong>${op.label}</strong>`;
    html +=
        `<div class="tt-row">Hardware: ${hw ? hw.label : op.hardware}</div>`;
    html +=
        `<div class="tt-row">State: ${op_snap.state.replace('_', ' ')}</div>`;

    if (op_snap.start_tick !== null) {
      html += `<div class="tt-row">Time: t=${op_snap.start_tick}`;
      if (op_snap.end_tick !== null) html += ` \u2192 t=${op_snap.end_tick}`;
      html += '</div>';
    }

    if (Object.keys(op_snap.frontier).length > 0) {
      html += `<div class="tt-row tt-frontier">Frontier: ${
          frontierToString(op_snap.frontier)}</div>`;
    }

    const waits = Object.entries(op.wait);
    if (waits.length > 0) {
      html += `<div class="tt-row">Waits: ${
          waits.map(([s, v]) => `${s}\u2265${v}`).join(', ')}</div>`;
    }

    const signals = Object.entries(op.signal);
    if (signals.length > 0) {
      html += `<div class="tt-row">Signals: ${
          signals.map(([s, v]) => `${s}\u2192${v}`).join(', ')}</div>`;
    }

    show_tooltip(html, event);
  });

  group.addEventListener('mousemove', (event) => {
    position_tooltip(get_tooltip(), event);
  });

  group.addEventListener('mouseleave', hide_tooltip);
}

function setup_edge_tooltip(hit_target, dep, scenario) {
  const sem = scenario.semaphores.find(s => s.id === dep.semaphore);
  const label = sem ? sem.label : dep.semaphore;

  hit_target.addEventListener('mouseenter', (event) => {
    show_tooltip(`<strong>${label}</strong> @ value ${dep.value}`, event);
  });

  hit_target.addEventListener('mousemove', (event) => {
    position_tooltip(get_tooltip(), event);
  });

  hit_target.addEventListener('mouseleave', hide_tooltip);
}
