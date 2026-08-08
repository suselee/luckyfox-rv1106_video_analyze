const statusGrid = document.querySelector("#status-grid");
const momentsContainer = document.querySelector("#moments");
const refreshButton = document.querySelector("#refresh-button");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatBytes(value) {
  if (!Number.isFinite(value)) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatDate(value) {
  if (!value) return "No data";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function renderHealth(health) {
  const latestLow = health.segments.latest_low;
  const latestHigh = health.segments.latest_high;
  const workers = health.workers;
  const analyzer = workers.analyzer || {};
  const lowRecorder = workers.recorders?.low || {};
  const highRecorder = workers.recorders?.high || {};
  const mqtt = workers.mqtt || {};
  const rv1106 = workers.rv1106 || {};
  const highOnly = health.configured.recording_mode === "high_only";

  const cards = [
    {
      label: "RTSP",
      value:
        health.configured.high_rtsp && (health.configured.low_rtsp || highOnly)
          ? "Configured"
          : "Waiting",
      meta: `Low: ${health.configured.low_rtsp ? "yes" : "no"} | 4K: ${
        health.configured.high_rtsp ? "yes" : "no"
      } | ${health.configured.recording_mode || "unknown"}`,
    },
    {
      label: "Tools",
      value: health.tools.ffmpeg ? "ffmpeg ready" : "ffmpeg missing",
      meta: `ffprobe: ${health.tools.ffprobe ? "ready" : "missing"}`,
    },
    {
      label: "Recorders",
      value: highOnly
        ? `4K ${highRecorder.status || "unknown"}`
        : `Low ${lowRecorder.status || "unknown"}`,
      meta: highOnly
        ? "Low disabled as configured"
        : `4K ${highRecorder.status || "unknown"}`,
    },
    {
      label: "Analyzer",
      value: analyzer.status || "unknown",
      meta:
        analyzer.message ||
        analyzer.segment ||
        analyzer.last_segment ||
        `${health.configured.analysis_image_mode}, ${health.configured.sample_frame_count} frames`,
    },
    {
      label: "Backlog",
      value: highOnly ? "Event-driven" : `${health.segments.pending_low} segments`,
      meta: highOnly
        ? `Latest 4K: ${formatDate(latestHigh?.started_at)}`
        : `Latest low: ${formatDate(latestLow?.started_at)}`,
    },
    {
      label: "Storage",
      value: `${formatBytes(health.storage.free_bytes)} free`,
      meta: `${formatBytes(health.storage.used_bytes)} used`,
    },
    {
      label: "4K Buffer",
      value: latestHigh ? "Receiving" : "No segment",
      meta: latestHigh ? formatDate(latestHigh.started_at) : health.configured.buffer_dir,
    },
    {
      label: "Output",
      value: "Nextcloud folder",
      meta: health.configured.output_dir,
    },
    {
      label: "MQTT",
      value: mqtt.status || "unknown",
      meta: mqtt.last_hit_at
        ? `Last hit: ${formatDate(mqtt.last_hit_at)}`
        : mqtt.message || `${health.configured.mqtt_host || ""}:${health.configured.mqtt_port || ""}`,
    },
    {
      label: "RV1106",
      value: rv1106.pipeline || rv1106.status || "unknown",
      meta:
        rv1106.status === "online"
          ? rv1106.pipeline === "sleeping"
            ? `Scheduled idle · ${rv1106.active_window_start || "07:00"}–${
                rv1106.active_window_end || "21:00"
              } · probable policy ${rv1106.probable_policy || "unknown"}`
            : `CPU ${Number(rv1106.cpu_percent || 0).toFixed(1)}% · ${Number(
              rv1106.temperature_c || 0,
            ).toFixed(1)}°C · p95 ${Number(rv1106.detector_p95_ms || 0).toFixed(1)}ms · ${
              rv1106.confirmed_sessions ?? rv1106.confirmed_tracks ?? 0
            } confirmed (now ${rv1106.confirmed_tracks || 0}) / ${
              rv1106.probable_tracks || 0
            } probable · Face scans ${rv1106.face_scan_attempts || 0} · matches ${
              rv1106.face_track_matches || 0
            } · max similarity ${
              Number(rv1106.max_face_similarity) >= 0
                ? Number(rv1106.max_face_similarity).toFixed(3)
                : "—"
            } · probable verify ${rv1106.probable_verified || 0}/${
              rv1106.probable_rejected || 0
            } · face ${rv1106.probable_face_verified || 0} · faceless ${
              rv1106.probable_faceless_accepted || 0
            } · adult reject ${rv1106.probable_adult_rejected || 0} · no-person ${
              rv1106.probable_no_person_rejected || 0
            }`
          : "Waiting for board heartbeat",
    },
  ];

  statusGrid.innerHTML = cards
    .map(
      (card) => `
        <article class="status-card">
          <span>${escapeHtml(card.label)}</span>
          <strong>${escapeHtml(card.value)}</strong>
          <small>${escapeHtml(card.meta)}</small>
        </article>
      `,
    )
    .join("");
}

function renderMoments(moments) {
  if (!moments.length) {
    momentsContainer.innerHTML = `
      <div class="empty-state">
        <h3>No saved moments yet</h3>
        <p>After RTSP and an analysis backend are configured, selected clips will appear here automatically.</p>
      </div>
    `;
    return;
  }

  momentsContainer.innerHTML = moments
    .map((moment) => {
      const tags = (moment.tags || [])
        .map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`)
        .join("");
      return `
        <article class="moment-card" data-id="${moment.id}" data-favorited="${moment.favorited}">
          <div class="moment-body">
            <div class="moment-title-row">
              <h3>${escapeHtml(moment.title)}</h3>
              <span class="confidence">${Math.round((moment.confidence || 0) * 100)}%</span>
            </div>
            <p>${escapeHtml(moment.summary)}</p>
            <div class="tags">${tags}</div>
            <div class="moment-meta">
              <span>${escapeHtml(formatDate(moment.source_started_at))}</span>
              <span>${escapeHtml(moment.camera_name)}</span>
              <span>${escapeHtml(moment.category || moment.analysis_backend || "vlm")}</span>
            </div>
            <div class="actions">
              <a class="button secondary" href="${moment.download_url}" download>Download clip</a>
              <button class="button secondary" data-action="favorite">${
                moment.favorited ? "Unfavorite" : "Favorite"
              }</button>
              <button class="button danger" data-action="delete">Delete</button>
            </div>
          </div>
        </article>
      `;
    })
    .join("");
}

async function loadAll() {
  refreshButton.disabled = true;
  try {
    const [health, moments] = await Promise.all([
      fetchJson("/api/health"),
      fetchJson("/api/moments"),
    ]);
    renderHealth(health);
    renderMoments(moments.moments || []);
  } finally {
    refreshButton.disabled = false;
  }
}

momentsContainer.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const card = button.closest(".moment-card");
  const id = card.dataset.id;
  const action = button.dataset.action;

  if (action === "favorite") {
    const favorited = card.dataset.favorited !== "true";
    await fetchJson(`/api/moments/${id}/favorite`, {
      method: "POST",
      body: JSON.stringify({ favorited }),
    });
    await loadAll();
  }

  if (action === "delete") {
    const confirmed = window.confirm("Delete this saved clip and metadata?");
    if (!confirmed) return;
    await fetchJson(`/api/moments/${id}`, { method: "DELETE" });
    await loadAll();
  }
});

refreshButton.addEventListener("click", loadAll);
loadAll().catch((error) => {
  statusGrid.innerHTML = `<article class="status-card error"><strong>Load failed</strong><small>${escapeHtml(
    error.message,
  )}</small></article>`;
});
