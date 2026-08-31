/*
 * Front end for the simulation.
 *
 * The server owns the simulation entirely; this file only sends a configuration
 * and paints whatever comes back. Every control list -- objectives, obstacle
 * layouts, senses, actions -- is built from /api/options, so adding a
 * capability in Python makes it appear here with no change to this file.
 */

const el = (id) => document.getElementById(id);

const canvas = el("world");
const ctx = canvas.getContext("2d");
const chart = el("chart");
const chartCtx = chart.getContext("2d");

const timeline = el("timeline");
const timelineCtx = timeline.getContext("2d");

let socket = null;
let options = null;
let layers = null; // static per-run: size, barriers, zone masks
let running = false;

/*
 * How each capability's share has moved, generation by generation.
 *
 * Accumulated here rather than on the server: the server already sends the
 * current expression every generation, and resending the whole history each
 * time would grow quadratically over a long run.
 */
let senseHistory = new Map(); // label -> array of shares, one per generation
let hoveredLabel = null;
let pinnedLabel = null;

// Colours for the zone shadings the objective reports, keyed by the matplotlib
// colormap name the Python side uses.
const ZONE_COLOURS = {
  Greens: "rgba(72, 190, 120, 0.30)",
  Blues: "rgba(90, 150, 240, 0.30)",
  Reds: "rgba(240, 90, 90, 0.32)",
};

/* ------------------------------------------------------------------ setup */

async function init() {
  options = await fetch("/api/options").then((response) => response.json());

  fillSelect(el("objective"), options.objectives, options.defaults.objective);
  fillSelect(el("barriers"), options.barriers, options.defaults.barriers);
  fillSelect(el("reproduction"), options.reproduction, options.defaults.reproduction);

  const numbers = [
    "population", "steps", "generations", "n_genes", "n_inner_neurons",
    "mutation_rate", "initial_energy", "sense_cost", "survival_zone_fraction",
    "zone_shrink", "offspring_per_survivor", "carrying_capacity", "mating_radius",
  ];
  for (const name of numbers) {
    const key = { zone_shrink: "zone_shrink" }[name] || name;
    if (options.defaults[key] !== undefined) el(name).value = options.defaults[key];
  }
  el("energy_enabled").checked = options.defaults.energy_enabled;

  buildCapabilityList("sensors", options.sensors, true);
  buildCapabilityList("actions", options.actions, false);

  el("mutation_rate").addEventListener("input", showMutationRate);
  el("stride").addEventListener("input", showStride);
  el("survival_zone_fraction").addEventListener("input", showZone);
  el("zone_shrink").addEventListener("input", showShrink);
  el("reproduction").addEventListener("change", showMatingField);
  showMutationRate();
  showStride();
  showZone();
  showShrink();
  showMatingField();

  el("start").addEventListener("click", start);
  el("stop").addEventListener("click", stop);

  wireDropdowns();
  connect();
}

function fillSelect(select, values, chosen) {
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === chosen;
    select.append(option);
  }
}

/* ----------------------------------------------------- capability menus */

function buildCapabilityList(kind, items, grouped) {
  const host = el(`${kind}-list`);
  host.innerHTML = "";

  // Senses are grouped by what they tell a creature; actions are a flat list.
  const groups = new Map();
  for (const item of items) {
    const name = grouped ? item.group : "";
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(item);
  }

  for (const [name, members] of groups) {
    if (name) {
      const heading = document.createElement("p");
      heading.className = "group-name";
      heading.textContent = name;
      host.append(heading);
    }
    for (const item of members) {
      const label = document.createElement("label");
      label.className = "check";

      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = item.value;
      box.checked = true;
      box.dataset.kind = kind;
      box.addEventListener("change", () => updateBadge(kind));

      label.append(box, document.createTextNode(item.label));
      host.append(label);
    }
  }
  updateBadge(kind);
}

function boxesFor(kind) {
  return [...document.querySelectorAll(`input[data-kind="${kind}"]`)];
}

function selectedValues(kind) {
  return boxesFor(kind)
    .filter((box) => box.checked)
    .map((box) => Number(box.value));
}

function updateBadge(kind) {
  const boxes = boxesFor(kind);
  const chosen = boxes.filter((box) => box.checked).length;
  el(`${kind}-count`).textContent = `${chosen}/${boxes.length}`;
}

function wireDropdowns() {
  for (const dropdown of document.querySelectorAll(".dropdown")) {
    const toggle = dropdown.querySelector(".dropdown-toggle");
    const menu = dropdown.querySelector(".dropdown-menu");

    toggle.addEventListener("click", (event) => {
      event.stopPropagation();
      const opening = menu.hidden;
      closeAllDropdowns();
      menu.hidden = !opening;
      toggle.setAttribute("aria-expanded", String(opening));
    });

    // Clicks inside the menu must not close it, or ticking several boxes
    // would mean reopening the menu every time.
    menu.addEventListener("click", (event) => event.stopPropagation());
  }

  for (const button of document.querySelectorAll("[data-select]")) {
    button.addEventListener("click", () => {
      const kind = button.dataset.target;
      const checked = button.dataset.select === "all";
      for (const box of boxesFor(kind)) box.checked = checked;
      updateBadge(kind);
    });
  }

  document.addEventListener("click", closeAllDropdowns);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeAllDropdowns();
  });
}

function closeAllDropdowns() {
  for (const menu of document.querySelectorAll(".dropdown-menu")) menu.hidden = true;
  for (const toggle of document.querySelectorAll(".dropdown-toggle")) {
    toggle.setAttribute("aria-expanded", "false");
  }
}

function showMutationRate() {
  el("mutation_label").textContent = `(${Number(el("mutation_rate").value).toFixed(3)})`;
}

function showStride() {
  const stride = Number(el("stride").value);
  el("stride_label").textContent = stride === 1 ? "(every step)" : `(${stride} steps per frame)`;
}

function showZone() {
  el("zone_label").textContent = `(${Math.round(el("survival_zone_fraction").value * 100)}% of the world)`;
}

function showShrink() {
  const shrink = Number(el("zone_shrink").value);
  el("shrink_label").textContent =
    shrink === 0 ? "(fixed target)" : `(${(shrink * 100).toFixed(1)}% smaller each generation)`;
}

// Mating radius only means anything when survivors have to find each other.
function showMatingField() {
  el("mating_radius_field").style.display =
    el("reproduction").value === "sexual" ? "" : "none";
}

/* ------------------------------------------------------------- websocket */

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);

  socket.addEventListener("message", (event) => handle(JSON.parse(event.data)));
  socket.addEventListener("close", () => {
    setRunning(false);
    setStatus("Lost the connection to the server. Reload to reconnect.", true);
  });
}

function start() {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    setStatus("Not connected. Reload the page.", true);
    return;
  }

  const sensors = selectedValues("sensors");
  const actions = selectedValues("actions");
  if (!sensors.length || !actions.length) {
    setStatus("Enable at least one sense and one action.", true);
    return;
  }

  resetChart();
  senseHistory = new Map();
  hoveredLabel = null;
  pinnedLabel = null;
  drawTimeline();
  el("brain").textContent = "Running…";
  setStatus("Starting…");
  setRunning(true);

  socket.send(
    JSON.stringify({
      action: "start",
      objective: el("objective").value,
      barriers: el("barriers").value,
      reproduction: el("reproduction").value,
      population: Number(el("population").value),
      steps: Number(el("steps").value),
      generations: Number(el("generations").value),
      n_genes: Number(el("n_genes").value),
      n_inner_neurons: Number(el("n_inner_neurons").value),
      mutation_rate: Number(el("mutation_rate").value),
      energy_enabled: el("energy_enabled").checked,
      initial_energy: Number(el("initial_energy").value),
      sense_cost: Number(el("sense_cost").value),
      survival_zone_fraction: Number(el("survival_zone_fraction").value),
      zone_shrink: Number(el("zone_shrink").value),
      offspring_per_survivor: Number(el("offspring_per_survivor").value),
      carrying_capacity: Number(el("carrying_capacity").value),
      mating_radius: Number(el("mating_radius").value),
      stride: Number(el("stride").value),
      seed: el("seed").value,
      sensors,
      actions,
    }),
  );
}

function stop() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ action: "stop" }));
  }
}

function handle(message) {
  switch (message.type) {
    case "start":
      layers = prepareLayers(message);
      setStatus(
        `${message.population} creatures · ${message.sensors.length} senses · ` +
          `${message.actions.length} actions`,
      );
      break;
    case "frame":
      drawFrame(message);
      break;
    case "generation":
      el("gen-value").textContent = message.generation;
      el("survivors-value").textContent =
        `${message.survivors} / ${message.previous_population}`;
      el("pop-value").textContent = message.population;
      el("brain").textContent = message.brain || "(no creatures)";
      recordHistory(message.expression);
      drawChart(
        message.history,
        message.populations,
        message.capacity,
        message.lineage_shares,
      );
      drawExpression(message.expression, message.zone_fraction);
      drawTimeline();
      break;
    case "done":
      setRunning(false);
      setStatus(
        message.extinct
          ? "Extinct — nobody left to breed."
          : "Finished.",
        Boolean(message.extinct),
      );
      break;
    case "stopped":
      setRunning(false);
      setStatus("Stopped.");
      break;
    case "error":
      setRunning(false);
      setStatus(message.message, true);
      break;
  }
}

/* --------------------------------------------------------------- drawing */

function decodeMask(encoded, width, height) {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return { bytes, width, height };
}

function prepareLayers(message) {
  const { width, height } = message;
  canvas.width = width;
  canvas.height = height;

  return {
    width,
    height,
    barriers: decodeMask(message.barriers, width, height),
    zones: message.zones.map((zone) => ({
      colour: ZONE_COLOURS[zone.colour] || "rgba(140, 140, 140, 0.25)",
      mask: decodeMask(zone.mask, width, height),
    })),
    dynamic: message.dynamic_zones,
  };
}

/*
 * Grids arrive in the Python array's [x][y] order with y counting upward, while
 * a canvas counts y downward from the top. Both are handled here, once, so the
 * rest of the drawing code can stay in world coordinates.
 */
function paintMask(image, mask, colour, alphaFromValue) {
  const [r, g, b, a] = colour;
  const { bytes, width, height } = mask;
  for (let x = 0; x < width; x += 1) {
    for (let y = 0; y < height; y += 1) {
      const value = bytes[x * height + y];
      if (!value) continue;
      const alpha = alphaFromValue ? (value / 255) * a : a;
      const pixel = ((height - 1 - y) * width + x) * 4;
      const existing = image.data[pixel + 3] / 255;
      const blend = alpha + existing * (1 - alpha);
      image.data[pixel] = (r * alpha + image.data[pixel] * existing * (1 - alpha)) / blend;
      image.data[pixel + 1] = (g * alpha + image.data[pixel + 1] * existing * (1 - alpha)) / blend;
      image.data[pixel + 2] = (b * alpha + image.data[pixel + 2] * existing * (1 - alpha)) / blend;
      image.data[pixel + 3] = blend * 255;
    }
  }
}

function drawFrame(message) {
  if (!layers) return;
  const { width, height } = layers;

  ctx.fillStyle = "#0b0c11";
  ctx.fillRect(0, 0, width, height);

  const image = ctx.createImageData(width, height);

  if (message.zones) {
    layers.zones.forEach((zone, index) => {
      if (message.zones[index]) {
        zone.mask = decodeMask(message.zones[index], width, height);
      }
    });
  }
  for (const zone of layers.zones) {
    paintMask(image, zone.mask, cssToRgba(zone.colour), false);
  }

  const pheromone = decodeMask(message.pheromone, width, height);
  paintMask(image, pheromone, [150, 110, 220, 0.75], true);

  paintMask(image, layers.barriers, [110, 116, 136, 0.95], false);
  ctx.putImageData(image, 0, 0);

  ctx.fillStyle = "#f2f4ff";
  const positions = message.positions;
  for (let i = 0; i < positions.length; i += 2) {
    ctx.fillRect(positions[i], height - 1 - positions[i + 1], 1, 1);
  }

  el("gen-value").textContent = message.generation;
  el("alive-value").textContent = message.alive;
}

function cssToRgba(css) {
  const parts = css.match(/[\d.]+/g).map(Number);
  return [parts[0], parts[1], parts[2], parts[3]];
}

function resetChart() {
  chartCtx.clearRect(0, 0, chart.width, chart.height);
}

function drawChart(history, populations, capacity, lineageShares) {
  const { width, height } = chart;
  const pad = 26;
  chartCtx.clearRect(0, 0, width, height);

  const plotHeight = height - pad * 1.5;
  const toY = (fraction) => height - pad - fraction * plotHeight;

  chartCtx.strokeStyle = "#2c2f3d";
  chartCtx.lineWidth = 1;
  for (const fraction of [0, 0.5, 1]) {
    const y = toY(fraction);
    chartCtx.beginPath();
    chartCtx.moveTo(pad, y);
    chartCtx.lineTo(width - 6, y);
    chartCtx.stroke();
    chartCtx.fillStyle = "#6f7488";
    chartCtx.font = "10px system-ui, sans-serif";
    chartCtx.fillText(`${Math.round(fraction * 100)}%`, 2, y + 3);
  }

  // The replacement line: survive at less than this and the population shrinks.
  const replacement = 1 / Number(el("offspring_per_survivor").value || 2);
  if (replacement > 0 && replacement <= 1) {
    chartCtx.strokeStyle = "#7a5c2e";
    chartCtx.setLineDash([4, 4]);
    chartCtx.beginPath();
    chartCtx.moveTo(pad, toY(replacement));
    chartCtx.lineTo(width - 6, toY(replacement));
    chartCtx.stroke();
    chartCtx.setLineDash([]);
  }

  if (!history.length) return;
  const span = Math.max(history.length - 1, 1);

  const line = (values, colour, scale) => {
    chartCtx.strokeStyle = colour;
    chartCtx.lineWidth = 2;
    chartCtx.beginPath();
    values.forEach((value, index) => {
      const x = pad + (index / span) * (width - pad - 6);
      const y = toY(Math.min(1, value * scale));
      if (index === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    });
    chartCtx.stroke();
  };

  if (populations && capacity) line(populations, "#4ec9a0", 1 / capacity);
  if (lineageShares) line(lineageShares, "#c98ae0", 1);
  line(history, "#6ea8fe", 1);
}

/*
 * Gene expression: what share of the population wires each capability at all.
 * A sense that falls to zero has been actively selected away; one that climbs
 * to 100% has become load-bearing.
 */
function drawExpression(expression, zoneFraction) {
  if (!expression) return;

  renderBars(el("sensor-bars"), expression.sensors, "sensor");
  renderBars(el("action-bars"), expression.actions, "action");

  const lineages = expression.lineages;
  el("lineage-value").textContent = `${lineages.alive} / ${lineages.founding}`;

  el("expression-summary").textContent =
    `${expression.population} creatures · ` +
    `${expression.mean_senses_used.toFixed(1)} senses wired on average · ` +
    `upkeep ${expression.mean_upkeep.toFixed(2)}/step · ` +
    `${lineages.alive} of ${lineages.founding} founding lineages left ` +
    `(${Math.round(lineages.remaining * 100)}%), biggest holds ` +
    `${Math.round(lineages.dominant_share * 100)}% · ` +
    `zone ${(zoneFraction * 100).toFixed(1)}%`;
}

function renderBars(host, entries, kind) {
  // Rebuild only when the set of labels changes; otherwise update in place so
  // the CSS width transition animates instead of flickering.
  if (host.children.length !== entries.length) {
    host.innerHTML = "";
    for (const entry of entries) {
      const row = document.createElement("div");
      row.className = `bar-row ${kind}`;
      row.dataset.label = entry.label;
      row.innerHTML =
        `<span class="bar-label"></span>` +
        `<span class="bar-track"><span class="bar-fill"></span></span>` +
        `<span class="bar-value"></span>`;
      attachHistoryHandlers(row, entry.label);
      host.append(row);
    }
  }

  entries.forEach((entry, index) => {
    const row = host.children[index];
    row.classList.toggle("dropped", entry.share === 0);
    row.classList.toggle("pinned", pinnedLabel === entry.label);
    row.querySelector(".bar-label").textContent = entry.label;
    row.querySelector(".bar-fill").style.width = `${entry.share * 100}%`;
    row.querySelector(".bar-value").textContent = `${Math.round(entry.share * 100)}%`;
  });
}

/*
 * Sweeping the pointer across the bars is the interaction: pointerenter is
 * enough for a drag, because holding the button down while moving still fires
 * enter on each row it crosses. Clicking pins one so the pointer can leave.
 */
function attachHistoryHandlers(row, label) {
  row.addEventListener("pointerenter", () => {
    hoveredLabel = label;
    drawTimeline();
  });
  row.addEventListener("pointerleave", () => {
    if (hoveredLabel === label) hoveredLabel = null;
    drawTimeline();
  });
  row.addEventListener("click", () => {
    pinnedLabel = pinnedLabel === label ? null : label;
    for (const other of document.querySelectorAll(".bar-row")) {
      other.classList.toggle("pinned", other.dataset.label === pinnedLabel);
    }
    drawTimeline();
  });
}

function recordHistory(expression) {
  for (const entry of [...expression.sensors, ...expression.actions]) {
    if (!senseHistory.has(entry.label)) senseHistory.set(entry.label, []);
    senseHistory.get(entry.label).push(entry.share);
  }
}

function drawTimeline() {
  const { width, height } = timeline;
  const label = hoveredLabel || pinnedLabel;
  const series = label ? senseHistory.get(label) : null;
  const pad = 26;

  timelineCtx.clearRect(0, 0, width, height);

  timelineCtx.strokeStyle = "#2c2f3d";
  timelineCtx.lineWidth = 1;
  timelineCtx.fillStyle = "#6f7488";
  timelineCtx.font = "10px system-ui, sans-serif";
  for (const fraction of [0, 0.5, 1]) {
    const y = height - pad + fraction * (pad - height + pad / 2);
    timelineCtx.beginPath();
    timelineCtx.moveTo(pad, y);
    timelineCtx.lineTo(width - 6, y);
    timelineCtx.stroke();
    timelineCtx.fillText(`${Math.round(fraction * 100)}%`, 2, y + 3);
  }

  const head = el("timeline-label");
  const hint = el("timeline-hint");

  if (!series || series.length === 0) {
    head.textContent = label
      ? `${label} — no history yet`
      : "Sweep across a bar to see its history";
    hint.textContent = pinnedLabel ? `pinned: ${pinnedLabel}` : "";
    return;
  }

  const first = series[0];
  const last = series[series.length - 1];
  const peak = Math.max(...series);
  head.textContent =
    `${label} — now ${Math.round(last * 100)}%, ` +
    `started ${Math.round(first * 100)}%, peaked ${Math.round(peak * 100)}%`;
  hint.textContent = pinnedLabel === label ? "pinned — click again to release" : "click to pin";

  const span = Math.max(series.length - 1, 1);
  const toX = (index) => pad + (index / span) * (width - pad - 6);
  const toY = (value) => height - pad - value * (height - pad * 1.5);

  // A filled area under the line, so a capability that dies out reads as a
  // shape collapsing rather than just a line drifting downward.
  timelineCtx.beginPath();
  timelineCtx.moveTo(toX(0), toY(0));
  series.forEach((value, index) => timelineCtx.lineTo(toX(index), toY(value)));
  timelineCtx.lineTo(toX(series.length - 1), toY(0));
  timelineCtx.closePath();
  timelineCtx.fillStyle = "rgba(110, 168, 254, 0.18)";
  timelineCtx.fill();

  timelineCtx.beginPath();
  series.forEach((value, index) => {
    const x = toX(index);
    const y = toY(value);
    if (index === 0) timelineCtx.moveTo(x, y);
    else timelineCtx.lineTo(x, y);
  });
  timelineCtx.strokeStyle = "#6ea8fe";
  timelineCtx.lineWidth = 2;
  timelineCtx.stroke();

  timelineCtx.fillStyle = "#6ea8fe";
  timelineCtx.beginPath();
  timelineCtx.arc(toX(series.length - 1), toY(last), 3, 0, Math.PI * 2);
  timelineCtx.fill();
}

/* ---------------------------------------------------------------- status */

function setRunning(state) {
  running = state;
  el("start").disabled = state;
  el("stop").disabled = !state;
}

function setStatus(text, isError = false) {
  const status = el("status");
  status.textContent = text;
  status.classList.toggle("error", isError);
}

init();
