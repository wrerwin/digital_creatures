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

/*
 * Recall: frames are archived generation by generation as a run goes, and the
 * whole archive becomes browsable once it ends.
 *
 * A hundred generations at 200 frames each would be some 270MB of pheromone
 * data, so the archive lives on a frame budget. When it overflows, whole
 * generations are dropped — but the one dropped is always the most *redundant*,
 * meaning the one whose neighbours are closest. The first and last are never
 * dropped. So a long run degrades into an evenly spread sample of itself rather
 * than keeping only the tail, which preserves what is actually interesting:
 * being able to compare the beginning against the end.
 */
const FRAME_BUDGET = 5000;
const MAX_FRAMES_PER_GENERATION = 900;

let recording = [];
let archive = new Map(); // generation number -> frames for that generation
let archivedFrames = 0;
let selectedFrames = [];
let playbackTimer = null;
let playhead = 0;

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

  // The skill panels create the brain-size inputs, so they must exist before
  // the defaults below are written into them.
  buildSkillDashboard(options);

  const numbers = [
    "population", "steps", "generations", "n_genes", "n_inner_neurons",
    "mutation_rate", "initial_energy", "sense_cost", "survival_zone_fraction",
    "zone_shrink", "offspring_per_survivor", "offspring_at_capacity",
    "carrying_capacity", "mating_radius",
  ];
  for (const name of numbers) {
    if (options.defaults[name] !== undefined) el(name).value = options.defaults[name];
  }
  el("energy_enabled").checked = options.defaults.energy_enabled;
  el("sense_cost").addEventListener("input", showUpkeep);
  el("energy_enabled").addEventListener("change", showUpkeep);
  showUpkeep();

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

  el("steps").addEventListener("input", showUpkeep);
  el("initial_energy").addEventListener("input", showUpkeep);
  el("start").addEventListener("click", start);
  el("stop").addEventListener("click", stop);

  el("replay-play").addEventListener("click", togglePlayback);
  el("replay-generation").addEventListener("change", loadRecalledGeneration);
  el("replay-scrub").addEventListener("input", (event) => {
    stopPlayback(); // dragging takes over from playback
    showRecalledFrame(Number(event.target.value));
  });

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

/*
 * One collapsible panel per category, built from /api/options.
 *
 * A capability's category and its tooltip both come from the server, which
 * reads them off the enums and the sensing functions' own docstrings — so a
 * new sense appears here, in the right group, correctly explained, with no
 * change to this file.
 */
function buildSkillDashboard(options) {
  const host = el("skills");
  host.innerHTML = "";

  for (const category of options.categories) {
    const members = options.capabilities.filter((item) => item.category === category.id);

    const group = document.createElement("section");
    group.className = "skill-group";
    group.dataset.category = category.id;
    group.innerHTML = `
      <button type="button" class="skill-header" aria-expanded="true">
        <span class="skill-icon">${category.icon}</span>
        <span class="skill-name">${category.name}</span>
        <span class="badge" data-count="${category.id}"></span>
        <span class="chevron" aria-hidden="true"></span>
      </button>
      <div class="skill-body">
        <p class="skill-blurb">${category.blurb}</p>
        <div class="skill-buttons">
          <button type="button" data-select="all">enable all</button>
          <button type="button" data-select="none">disable all</button>
        </div>
        <div class="skill-list"></div>
      </div>`;

    const list = group.querySelector(".skill-list");
    for (const item of members) list.append(skillRow(item, category.id));

    group.querySelector(".skill-header").addEventListener("click", () => {
      group.classList.toggle("collapsed");
      group
        .querySelector(".skill-header")
        .setAttribute("aria-expanded", String(!group.classList.contains("collapsed")));
    });

    for (const button of group.querySelectorAll("[data-select]")) {
      button.addEventListener("click", () => {
        const wanted = button.dataset.select === "all";
        for (const box of group.querySelectorAll("input[type=checkbox]")) box.checked = wanted;
        updateCounts();
      });
    }

    // Brain size belongs with intelligence: it is how much there is to think
    // with, as opposed to what there is to think about.
    if (category.id === "intelligence") {
      const extras = document.createElement("div");
      extras.className = "brain-size";
      extras.innerHTML = `
        <label>Genes<input id="n_genes" type="number" min="1" max="200" step="1" /></label>
        <label>Inner neurons
          <input id="n_inner_neurons" type="number" min="1" max="32" step="1" /></label>`;
      group.querySelector(".skill-body").append(extras);
    }

    host.append(group);
  }

  el("skills-toggle-all").addEventListener("click", toggleAllGroups);
  updateCounts();
}

function skillRow(item, categoryId) {
  const label = document.createElement("label");
  label.className = "check";
  label.dataset.tip = item.description;
  label.title = item.description; // native fallback, and for accessibility

  const box = document.createElement("input");
  box.type = "checkbox";
  box.value = item.value;
  box.checked = true;
  box.dataset.kind = item.kind;
  box.dataset.category = categoryId;
  box.addEventListener("change", updateCounts);

  const name = document.createElement("span");
  name.className = "check-label";
  name.textContent = item.label;

  label.append(box, name);
  return label;
}

function boxesFor(kind) {
  return [...document.querySelectorAll(`input[data-kind="${kind}"]`)];
}

function selectedValues(kind) {
  return boxesFor(kind)
    .filter((box) => box.checked)
    .map((box) => Number(box.value));
}

function updateCounts() {
  for (const badge of document.querySelectorAll("[data-count]")) {
    const id = badge.dataset.count;
    const boxes = [...document.querySelectorAll(`input[data-category="${id}"]`)];
    const chosen = boxes.filter((box) => box.checked).length;
    badge.textContent = `${chosen}/${boxes.length}`;
    // A category with nothing enabled is worth flagging: no sensing at all, or
    // no way to move, means the run cannot start.
    badge.classList.toggle("empty", chosen === 0);
  }
  showUpkeep();
}

function toggleAllGroups() {
  const groups = [...document.querySelectorAll(".skill-group")];
  const collapsing = groups.some((group) => !group.classList.contains("collapsed"));
  for (const group of groups) {
    group.classList.toggle("collapsed", collapsing);
    group.querySelector(".skill-header").setAttribute("aria-expanded", String(!collapsing));
  }
  el("skills-toggle-all").textContent = collapsing ? "expand all" : "collapse all";
}

/*
 * What the current selection would cost to run.
 *
 * Upkeep is charged per distinct sense a brain actually wires, so this is the
 * ceiling rather than a prediction — but it is the number that decides whether
 * a generous loadout is affordable, and it moves as boxes are ticked.
 */
function showUpkeep() {
  const note = el("upkeep-note");
  if (!options) return;

  const senses = selectedValues("sensors").length;
  if (!el("energy_enabled").checked) {
    note.classList.remove("costly");
    note.textContent = "Metabolism is off — skills cost nothing to run.";
    return;
  }

  const perStep =
    options.defaults.metabolism_base + senses * Number(el("sense_cost").value || 0);
  const budget = Number(el("initial_energy").value || 0);
  const steps = Number(el("steps").value || 1);
  const affordable = perStep * steps <= budget;

  note.classList.toggle("costly", !affordable);
  note.innerHTML =
    `A brain wiring all ${senses} senses burns <strong>${perStep.toFixed(2)}</strong>/step, ` +
    `about <strong>${Math.round(perStep * steps)}</strong> over a generation ` +
    `of ${steps} steps — against a budget of ${budget}. ` +
    (affordable ? "Affordable." : "That starves before the end.");
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

  // A new run invalidates whatever was recalled from the previous one.
  stopPlayback();
  recording = [];
  archive = new Map();
  archivedFrames = 0;
  selectedFrames = [];
  el("replay").hidden = true;

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
      offspring_at_capacity: Number(el("offspring_at_capacity").value),
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
      recording.push(message);
      if (recording.length > MAX_FRAMES_PER_GENERATION) recording.shift();
      drawFrame(message);
      break;
    case "generation":
      keepForRecall();
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
        message.thresholds,
      );
      drawExpression(message.expression, message.zone_fraction);
      drawTimeline();
      break;
    case "done":
      setRunning(false);
      revealRecall();
      setStatus(
        message.extinct ? "Extinct — nobody left to breed." : "Finished.",
        Boolean(message.extinct),
      );
      break;
    case "stopped":
      setRunning(false);
      revealRecall();
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

/* ------------------------------------------------------------ recall */

/* Archive the generation that just finished. Silent: the controls only appear
   once the whole run is over, so they do not churn while it is going. */
function keepForRecall() {
  if (!recording.length) return;

  // Frames carry the generation they belong to; the generation message counts
  // completed generations, so it is one ahead. Trust the frames.
  const number = recording[0].generation;
  archive.set(number, recording);
  archivedFrames += recording.length;
  recording = [];
  pruneArchive();
}

function pruneArchive() {
  while (archivedFrames > FRAME_BUDGET && archive.size > 2) {
    const generations = [...archive.keys()].sort((a, b) => a - b);

    // Drop whichever interior generation sits in the most crowded stretch, so
    // what remains stays spread across the whole run.
    let victim = generations[1];
    let closest = Infinity;
    for (let i = 1; i < generations.length - 1; i += 1) {
      const gap = generations[i + 1] - generations[i - 1];
      if (gap < closest) {
        closest = gap;
        victim = generations[i];
      }
    }

    archivedFrames -= archive.get(victim).length;
    archive.delete(victim);
  }
}

/* Reveal the archive once the run has stopped, not during it. */
function revealRecall() {
  keepForRecall(); // a generation cut short mid-run is still worth keeping

  const bar = el("replay");
  bar.hidden = archive.size === 0;
  if (bar.hidden) return;

  const generations = [...archive.keys()].sort((a, b) => a - b);
  const picker = el("replay-generation");
  picker.innerHTML = "";
  for (const number of generations) {
    const option = document.createElement("option");
    option.value = String(number);
    option.textContent = `Generation ${number + 1}`;
    picker.append(option);
  }

  const span = generations[generations.length - 1] - generations[0] + 1;
  el("replay-note").textContent =
    archive.size < span
      ? `${archive.size} of ${span} kept — sampled to fit memory`
      : `${archive.size} generation${archive.size === 1 ? "" : "s"}`;

  // Open on the last generation: the one the run ended on.
  picker.value = String(generations[generations.length - 1]);
  loadRecalledGeneration();
}

function loadRecalledGeneration() {
  stopPlayback();
  selectedFrames = archive.get(Number(el("replay-generation").value)) || [];
  const scrub = el("replay-scrub");
  scrub.max = String(Math.max(0, selectedFrames.length - 1));

  // Park on the final frame: the state selection actually acted on.
  showRecalledFrame(selectedFrames.length - 1);
}

function describePlayhead() {
  const frame = selectedFrames[playhead];
  if (!frame) return;
  el("replay-label").textContent =
    `gen ${frame.generation + 1} · step ${frame.step} · ${frame.alive} alive`;
}

function showRecalledFrame(index) {
  if (!selectedFrames.length) return;
  playhead = Math.max(0, Math.min(selectedFrames.length - 1, index));
  el("replay-scrub").value = String(playhead);
  drawFrame(selectedFrames[playhead]);
  describePlayhead();
}

function togglePlayback() {
  if (playbackTimer !== null) {
    stopPlayback();
    return;
  }
  if (!selectedFrames.length) return;

  // Starting from the end would replay nothing, so rewind first.
  if (playhead >= selectedFrames.length - 1) playhead = 0;

  el("replay-play").textContent = "❚❚";
  playbackTimer = window.setInterval(() => {
    if (playhead >= selectedFrames.length - 1) {
      stopPlayback();
      return;
    }
    showRecalledFrame(playhead + 1);
  }, 40);
}

function stopPlayback() {
  if (playbackTimer !== null) window.clearInterval(playbackTimer);
  playbackTimer = null;
  el("replay-play").textContent = "▶";
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

function drawChart(history, populations, capacity, lineageShares, thresholds) {
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

  // The bar each generation had to clear. It rises as the world fills, so
  // where the survival line crosses it is exactly where the population turns.
  if (thresholds && thresholds.length) {
    chartCtx.setLineDash([4, 4]);
    line(thresholds, "#c08a4a", 1);
    chartCtx.setLineDash([]);
  }

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
