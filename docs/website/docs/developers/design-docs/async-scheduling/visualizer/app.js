// Application entry point: wires the simulator, renderer, and UI controls.
//
// URL parameters:
//   ?inline    — Compact mode for iframe embedding. Hides the header, footer,
//                and bottom panels (semaphore state + events) to focus on the
//                DAG, timeline, and transport controls.
//   ?scenario=<id> — Load a specific scenario by ID on startup.

import {createRenderer} from './renderer.js';
import {expandHardware, getScalableHardware, scaleScenario, supportsDepth,} from './scaling.js';
import {scenarios} from './scenarios.js';
import {frontierToString, simulate} from './simulator.js';

const url_params = new URLSearchParams(window.location.search);
const is_inline = url_params.has('inline');
if (is_inline) {
  document.body.classList.add('inline');
}

// Theme: ?dark parameter forces dark mode. In standalone mode (not inline),
// also follow the OS preference.
if (url_params.has('dark') ||
    (!is_inline && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
  document.body.classList.add('dark');
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let base_scenario = null;
let current_scenario = null;
let current_snapshots = null;
let current_renderer = null;
let current_tick = 0;
let at_defaults = true;
let playing = false;
let play_timer = null;

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const dom = {
  select: document.getElementById('scenario-select'),
  description: document.getElementById('scenario-description'),
  dag: document.getElementById('dag-container'),
  timeline: document.getElementById('timeline-container'),
  slider: document.getElementById('time-slider'),
  time_label: document.getElementById('time-label'),
  time_total: document.getElementById('time-total'),
  sem_state: document.getElementById('sem-state'),
  op_frontiers: document.getElementById('op-frontiers'),
  event_list: document.getElementById('event-list'),
  annotation_section: document.getElementById('annotation-section'),
  annotation_text: document.getElementById('annotation-text'),
  btn_reset: document.getElementById('btn-reset'),
  btn_back: document.getElementById('btn-step-back'),
  btn_play: document.getElementById('btn-play'),
  btn_fwd: document.getElementById('btn-step-fwd'),
  btn_end: document.getElementById('btn-end'),
  width_slider: document.getElementById('width-slider'),
  depth_slider: document.getElementById('depth-slider'),
  hw_slider: document.getElementById('hw-slider'),
  width_value: document.getElementById('width-value'),
  depth_value: document.getElementById('depth-value'),
  hw_value: document.getElementById('hw-value'),
  hw_label: document.getElementById('hw-label'),
  scaling_summary: document.getElementById('scaling-summary'),
};

// ---------------------------------------------------------------------------
// Scenario loading
// ---------------------------------------------------------------------------

// Populate the scenario selector dropdown.
for (const scenario of scenarios) {
  const opt = document.createElement('option');
  opt.value = scenario.id;
  opt.textContent = scenario.name;
  dom.select.appendChild(opt);
}

function load_scenario(scenario) {
  stop_playback();

  base_scenario = scenario;
  dom.description.textContent = scenario.description;

  // Reset scaling sliders.
  dom.width_slider.value = 1;
  dom.depth_slider.value = 1;

  // Enable/disable depth slider based on chain_through support.
  const depth_supported = supportsDepth(scenario);
  dom.depth_slider.disabled = !depth_supported;
  dom.depth_value.classList.toggle('disabled', !depth_supported);

  // Configure hardware count slider.
  const scalable_hw = getScalableHardware(scenario);
  const hw_supported = !!scalable_hw;
  dom.hw_slider.disabled = !hw_supported;
  dom.hw_value.classList.toggle('disabled', !hw_supported);
  if (hw_supported) {
    dom.hw_slider.value = scalable_hw.count;
    dom.hw_label.textContent = scalable_hw.label + 's';
  } else {
    dom.hw_slider.value = 1;
    dom.hw_label.textContent = 'GPUs';
  }

  apply_scaling();
}

// ---------------------------------------------------------------------------
// Scaling
// ---------------------------------------------------------------------------

function apply_scaling() {
  stop_playback();

  const width = parseInt(dom.width_slider.value);
  const depth = parseInt(dom.depth_slider.value);
  const hw_count = parseInt(dom.hw_slider.value);

  dom.width_value.textContent = width;
  dom.depth_value.textContent = depth;
  dom.hw_value.textContent = hw_count;

  const scaled = scaleScenario(base_scenario, width, depth);

  // Expand hardware after width/depth scaling. Hardware expansion creates
  // individual lane entries from count-based hardware definitions, allowing
  // the simulator to dynamically assign operations to lanes.
  const expanded = expandHardware(scaled, hw_count);

  // Determine if any scaling dimension deviates from the base scenario's
  // defaults. Annotations only apply at default settings since tick numbers
  // change when the schedule changes.
  const scalable_hw = getScalableHardware(base_scenario);
  const default_hw_count = scalable_hw ? scalable_hw.count : 1;
  at_defaults = (width === 1 && depth === 1 && hw_count === default_hw_count);
  const is_scaled = !at_defaults;

  // Update summary label.
  const parts = [];
  if (width > 1 || depth > 1) {
    const instance_count = width * depth;
    parts.push(`${instance_count} instance${instance_count > 1 ? 's' : ''}`);
  }
  if (scalable_hw && hw_count !== default_hw_count) {
    parts.push(`${hw_count} ${scalable_hw.label.toLowerCase()}${
        hw_count > 1 ? 's' : ''}`);
  }
  if (parts.length > 0) {
    const op_count = expanded.operations.length;
    dom.scaling_summary.textContent = `= ${parts.join(', ')}, ${op_count} ops`;
  } else {
    dom.scaling_summary.textContent = '';
  }

  current_scenario = expanded;

  // Run the simulation.
  current_snapshots = simulate(expanded);

  // Configure the time slider.
  const max_tick = current_snapshots.length - 1;
  dom.slider.max = max_tick;
  dom.slider.value = 0;
  dom.time_total.textContent = `/ ${max_tick}`;

  // Create the renderer. When scaled, the DAG shows the base scenario in
  // static mode (all nodes retired, no tick animation) so the user can still
  // see the dependency structure. The timeline uses the expanded scenario.
  current_renderer = createRenderer(
      dom.dag,
      dom.timeline,
      expanded,
      current_snapshots,
      is_scaled ? base_scenario : null,
  );

  // Show initial state.
  set_tick(0);
}

// ---------------------------------------------------------------------------
// Tick management
// ---------------------------------------------------------------------------

function set_tick(tick) {
  current_tick = tick;
  dom.slider.value = tick;
  dom.time_label.textContent = `t = ${tick}`;

  if (current_renderer) {
    current_renderer.setTick(tick);
  }

  const snapshot = current_snapshots[tick];
  update_state_display(snapshot);
  update_event_log(snapshot);
  update_annotation(tick);
}

// ---------------------------------------------------------------------------
// State display — semaphores and operation frontiers
// ---------------------------------------------------------------------------

function update_state_display(snapshot) {
  // Semaphore state table.
  const sem_entries = Object.entries(snapshot.semaphores);
  if (sem_entries.length > 0) {
    let html = '<table><thead><tr>' +
        '<th>Semaphore</th><th>Value</th><th>Frontier</th>' +
        '</tr></thead><tbody>';

    for (const [sem_id, state] of sem_entries) {
      const sem_def = current_scenario.semaphores.find(s => s.id === sem_id);
      const label = sem_def ? sem_def.label : sem_id;
      html += '<tr>' +
          `<td class="sem-name">${label}</td>` +
          `<td>${state.value}</td>` +
          `<td class="frontier">${frontierToString(state.frontier)}</td>` +
          '</tr>';
    }
    html += '</tbody></table>';
    dom.sem_state.innerHTML = html;
  } else {
    dom.sem_state.innerHTML = '';
  }

  // Operation frontiers table.
  const op_entries = Object.entries(snapshot.operations);
  if (op_entries.length > 0) {
    let html = '<table><thead><tr>' +
        '<th>Operation</th><th>State</th><th>Frontier</th>' +
        '</tr></thead><tbody>';

    for (const [op_id, state] of op_entries) {
      const op_def = current_scenario.operations.find(o => o.id === op_id);
      const label = op_def ? op_def.label : op_id;
      const display_state = state.state.replace('_', ' ');
      const frontier = Object.keys(state.frontier).length > 0 ?
          frontierToString(state.frontier) :
          '\u2014';
      html += '<tr>' +
          `<td class="op-name">${label}</td>` +
          `<td>${display_state}</td>` +
          `<td class="frontier">${frontier}</td>` +
          '</tr>';
    }
    html += '</tbody></table>';
    dom.op_frontiers.innerHTML = html;
  } else {
    dom.op_frontiers.innerHTML = '';
  }
}

// ---------------------------------------------------------------------------
// Event log
// ---------------------------------------------------------------------------

const EVENT_ICONS = {
  retired: '\u25a0',   // filled square
  signaled: '\u25b2',  // filled triangle
  ready: '\u25cf',     // filled circle
  issued: '\u25b6',    // play symbol
};

function update_event_log(snapshot) {
  if (snapshot.events.length === 0) {
    dom.event_list.innerHTML =
        '<div class="no-events">No events this tick</div>';
    return;
  }

  let html = '';
  for (const event of snapshot.events) {
    const icon = EVENT_ICONS[event.type] || '\u2022';
    html +=
        `<div class="event ${event.type}">${icon} ${event.description}</div>`;
  }
  dom.event_list.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Annotations
// ---------------------------------------------------------------------------

function update_annotation(tick) {
  // Annotations are authored against the base scenario's default schedule.
  // When any scaling dimension deviates from defaults, tick numbers no longer
  // correspond to the authored annotations, so we show a placeholder instead.
  const annotations = at_defaults ? base_scenario.annotations : null;

  if (!annotations || annotations.length === 0) {
    dom.annotation_text.textContent = at_defaults ?
        'Scrub the timeline or step through ticks to see annotations.' :
        'Annotations are available at default settings.';
    dom.annotation_section.classList.remove('active');
    return;
  }

  const annotation = annotations.find(a => a.tick === tick);
  if (annotation) {
    dom.annotation_text.textContent = annotation.text;
    dom.annotation_section.classList.add('active');
  } else {
    dom.annotation_text.textContent =
        'Scrub the timeline or step through ticks to see annotations.';
    dom.annotation_section.classList.remove('active');
  }
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

function stop_playback() {
  playing = false;
  dom.btn_play.textContent = '\u25b6';
  if (play_timer !== null) {
    clearInterval(play_timer);
    play_timer = null;
  }
}

function toggle_playback() {
  if (playing) {
    stop_playback();
    return;
  }

  // If at the end, restart from the beginning.
  if (current_tick >= current_snapshots.length - 1) {
    set_tick(0);
  }

  playing = true;
  dom.btn_play.textContent = '\u23f8';  // pause symbol
  play_timer = setInterval(() => {
    if (current_tick < current_snapshots.length - 1) {
      set_tick(current_tick + 1);
    } else {
      stop_playback();
    }
  }, 600);
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

dom.select.addEventListener('change', () => {
  const scenario = scenarios.find(s => s.id === dom.select.value);
  if (scenario) load_scenario(scenario);
});

dom.slider.addEventListener('input', () => {
  set_tick(parseInt(dom.slider.value));
});

dom.width_slider.addEventListener('input', apply_scaling);

dom.depth_slider.addEventListener('input', () => {
  if (!dom.depth_slider.disabled) apply_scaling();
});

dom.hw_slider.addEventListener('input', () => {
  if (!dom.hw_slider.disabled) apply_scaling();
});

dom.btn_reset.addEventListener('click', () => set_tick(0));

dom.btn_back.addEventListener('click', () => {
  if (current_tick > 0) set_tick(current_tick - 1);
});

dom.btn_play.addEventListener('click', toggle_playback);

dom.btn_fwd.addEventListener('click', () => {
  if (current_tick < current_snapshots.length - 1) set_tick(current_tick + 1);
});

dom.btn_end.addEventListener('click', () => {
  set_tick(current_snapshots.length - 1);
});

// Keyboard shortcuts.
document.addEventListener('keydown', (event) => {
  // Don't capture when focus is on an interactive element.
  if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') {
    return;
  }

  switch (event.key) {
    case 'ArrowLeft':
      event.preventDefault();
      if (current_tick > 0) set_tick(current_tick - 1);
      break;
    case 'ArrowRight':
      event.preventDefault();
      if (current_tick < current_snapshots.length - 1)
        set_tick(current_tick + 1);
      break;
    case ' ':
      event.preventDefault();
      toggle_playback();
      break;
    case 'Home':
      event.preventDefault();
      set_tick(0);
      break;
    case 'End':
      event.preventDefault();
      set_tick(current_snapshots.length - 1);
      break;
  }
});

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

function applyTheme(dark) {
  document.body.classList.toggle('dark', dark);
  // SVG attributes are set in JS, not CSS, so a theme change requires
  // re-rendering. Preserve the current playback position.
  if (current_renderer) {
    const saved_tick = current_tick;
    apply_scaling();
    set_tick(Math.min(saved_tick, current_snapshots.length - 1));
  }
}

// Parent page controls theme in inline mode via postMessage.
window.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'theme') {
    applyTheme(event.data.dark);
  }
});

// Follow OS preference changes in standalone mode.
if (!is_inline) {
  window.matchMedia('(prefers-color-scheme: dark)')
      .addEventListener('change', (event) => applyTheme(event.matches));
}

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------

// Load the scenario specified in the URL, or the first scenario by default.
const initial_id = url_params.get('scenario');
const initial_scenario =
    (initial_id && scenarios.find(s => s.id === initial_id)) || scenarios[0];
dom.select.value = initial_scenario.id;
load_scenario(initial_scenario);
