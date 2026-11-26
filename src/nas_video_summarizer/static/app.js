const statusGrid = document.querySelector("#status-grid");
const momentsContainer = document.querySelector("#moments");
const refreshButton = document.querySelector("#refresh-button");
const tabButtons = Array.from(document.querySelectorAll(".tab-button"));
const unlabeledOnlyCheckbox = document.querySelector("#unlabeled-only");
const sectionHint = document.querySelector("#section-hint");

const uiState = {
  tab: "moments",
  unlabeledOnly: false,
  health: null,
  moments: [],
  rejected: [],
  rejectedLoaded: false,
};

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

function feedbackBadge(label) {
  if (label === "daughter") {
    return '<span class="feedback-badge daughter">✓ 是女儿</span>';
  }
  if (label === "adult_only") {
    return '<span class="feedback-badge adult">✗ 纯大人</span>';
  }
  if (label === "uncertain") {
    return '<span class="feedback-badge uncertain">? 不确定</span>';
  }
  return "";
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
  if (!health) return;
  const latestLow = health.segments.latest_low;
  const latestHigh = health.segments.latest_high;
  const workers = health.workers;
  const analyzer = workers.analyzer || {};
  const lowRecorder = workers.recorders?.low || {};
  const highRecorder = workers.recorders?.high || {};
  const mqtt = workers.mqtt || {};
  const rv1106 = workers.rv1106 || {};
  const feedback = health.feedback || {};
  const fbLabel = feedback.by_label || {};
  const fbMoment = feedback.by_source?.moment || {};
  const fbRejected = feedback.by_source?.rejected || {};
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
    {
      label: "Feedback",
      value: `${feedback.total || 0} labeled`,
      meta: `女儿 ${fbLabel.daughter || 0} · 纯大人 ${fbLabel.adult_only || 0} (删片 ${
        fbMoment.deleted || 0
      }) · 被拒漏检 ${fbRejected.daughter || 0}`,
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
  const visible = uiState.unlabeledOnly
    ? moments.filter((moment) => !moment.feedback_label)
    : moments;

  if (!visible.length) {
    momentsContainer.innerHTML = `
      <div class="empty-state">
        <h3>${uiState.unlabeledOnly && moments.length ? "All saved moments labeled" : "No saved moments yet"}</h3>
        <p>${
          uiState.unlabeledOnly && moments.length
            ? "Uncheck 'Unlabeled only' to view previously reviewed moments."
            : "After RTSP and an analysis backend are configured, selected clips will appear here automatically."
        }</p>
      </div>
    `;
    return;
  }

  momentsContainer.innerHTML = visible
    .map((moment) => {
      const tags = (moment.tags || [])
        .map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`)
        .join("");
      const badge = feedbackBadge(moment.feedback_label);
      const isDaughter = moment.feedback_label === "daughter";
      const previewUrl = moment.preview_url || `/api/moments/${moment.id}/preview.jpg`;
      return `
        <article
          class="moment-card ${moment.feedback_label ? `labeled-${escapeHtml(moment.feedback_label)}` : ""}"
          data-source="moment"
          data-id="${moment.id}"
          data-favorited="${moment.favorited}"
        >
          <div class="preview-wrap">
            <img
              class="moment-preview"
              src="${escapeHtml(previewUrl)}"
              alt="${escapeHtml(moment.title)}"
              loading="lazy"
            />
            ${badge}
          </div>
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
            <div class="feedback-actions">
              <button
                class="button feedback-btn ${isDaughter ? "active-daughter" : "secondary"}"
                data-action="feedback-daughter"
              >✓ 是女儿</button>
              <button
                class="button feedback-btn danger"
                data-action="feedback-adult-delete"
              >✗ 纯大人·删片</button>
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

function renderRejected(items) {
  const visible = uiState.unlabeledOnly
    ? items.filter((item) => !item.feedback_label)
    : items;

  if (!visible.length) {
    momentsContainer.innerHTML = `
      <div class="empty-state">
        <h3>${uiState.unlabeledOnly && items.length ? "All rejected clips labeled" : "No rejected clips archived"}</h3>
        <p>${
          uiState.unlabeledOnly && items.length
            ? "Uncheck 'Unlabeled only' to review already labeled clips."
            : "Rejected probable clips appear here when RV1106_KEEP_REJECTED=true."
        }</p>
      </div>
    `;
    return;
  }

  momentsContainer.innerHTML = visible
    .map((item) => {
      const badge = feedbackBadge(item.feedback_label);
      const label = item.feedback_label || "";
      return `
        <article
          class="moment-card ${label ? `labeled-${escapeHtml(label)}` : ""}"
          data-source="rejected"
          data-name="${escapeHtml(item.name)}"
        >
          <div class="preview-wrap">
            <img
              class="moment-preview"
              src="${escapeHtml(item.preview_url)}"
              alt="${escapeHtml(item.decision)}"
              loading="lazy"
            />
            ${badge}
          </div>
          <div class="moment-body">
            <div class="moment-title-row">
              <h3>${escapeHtml(item.decision)}</h3>
              <span class="confidence">${Math.round((item.score || 0) * 100)}%</span>
            </div>
            <div class="moment-meta">
              <span>${escapeHtml(formatDate(item.archived_at))}</span>
              <span>${escapeHtml(item.identity || "probable")}</span>
              <span>${escapeHtml(item.name)}</span>
            </div>
            <div class="feedback-actions">
              <button
                class="button feedback-btn ${label === "daughter" ? "active-daughter" : "secondary"}"
                data-action="rejected-daughter"
              >✓ 漏检(是女儿)</button>
              <button
                class="button feedback-btn ${label === "adult_only" ? "active-adult" : "danger"}"
                data-action="rejected-adult"
              >✗ 确实纯大人</button>
              <button
                class="button feedback-btn ${label === "uncertain" ? "active-uncertain" : "secondary"}"
                data-action="rejected-uncertain"
              >? 不确定</button>
            </div>
            <div class="actions">
              <button
                class="button rescue-btn ${item.rescued ? "active-rescued" : ""}"
                data-action="rejected-rescue"
                ${item.rescued ? "disabled" : ""}
              >${item.rescued ? "✓ 已捞回保存" : "+ 捞回保存"}</button>
            </div>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderActiveTab() {
  for (const btn of tabButtons) {
    const active = btn.dataset.tab === uiState.tab;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  }
  if (sectionHint) {
    sectionHint.textContent =
      uiState.tab === "moments"
        ? "Automatically saved 4K clips. Review previews, tag daughter / adult-only, or download for VLC."
        : "Rejected probable clips archived on NAS. Label missed daughter clips vs true adult-only rejections.";
  }
  if (uiState.tab === "moments") {
    renderMoments(uiState.moments);
  } else {
    renderRejected(uiState.rejected);
  }
}

async function loadAll() {
  refreshButton.disabled = true;
  try {
    const requests = [
      fetchJson("/api/health"),
      fetchJson("/api/moments"),
    ];
    if (uiState.tab === "rejected" || uiState.rejectedLoaded) {
      requests.push(fetchJson("/api/rejected?limit=60"));
    }
    const [health, moments, rejected] = await Promise.all(requests);
    uiState.health = health;
    uiState.moments = moments.moments || [];
    if (rejected) {
      uiState.rejected = rejected.rejected || [];
      uiState.rejectedLoaded = true;
    }
    renderHealth(uiState.health);
    renderActiveTab();
  } finally {
    refreshButton.disabled = false;
  }
}

async function switchTab(tab) {
  uiState.tab = tab;
  if (tab === "rejected" && !uiState.rejectedLoaded) {
    momentsContainer.innerHTML = '<div class="empty-state"><h3>Loading rejected clips…</h3></div>';
    const data = await fetchJson("/api/rejected?limit=60");
    uiState.rejected = data.rejected || [];
    uiState.rejectedLoaded = true;
  }
  renderActiveTab();
}

for (const btn of tabButtons) {
  btn.addEventListener("click", () => {
    switchTab(btn.dataset.tab).catch((error) => {
      momentsContainer.innerHTML = `<div class="empty-state"><h3>Load failed</h3><p>${escapeHtml(
        error.message,
      )}</p></div>`;
    });
  });
}

if (unlabeledOnlyCheckbox) {
  unlabeledOnlyCheckbox.addEventListener("change", () => {
    uiState.unlabeledOnly = Boolean(unlabeledOnlyCheckbox.checked);
    renderActiveTab();
  });
}

momentsContainer.addEventListener(
  "error",
  (event) => {
    if (event.target && event.target.classList?.contains("moment-preview")) {
      event.target.classList.add("preview-broken");
    }
  },
  true,
);

momentsContainer.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const card = button.closest(".moment-card");
  if (!card) return;
  const action = button.dataset.action;

  button.disabled = true;
  try {
    if (card.dataset.source === "rejected") {
      const name = card.dataset.name;
      if (action === "rejected-rescue") {
        const res = await fetchJson(`/api/rejected/${encodeURIComponent(name)}/rescue`, {
          method: "POST",
          body: JSON.stringify({}),
        });
        const target = uiState.rejected.find((item) => item.name === name);
        if (target) {
          target.rescued = true;
          target.rescued_moment_id = res.moment_id;
          target.feedback_label = "daughter";
        }
        if (uiState.health && res.summary) {
          uiState.health.feedback = res.summary;
          renderHealth(uiState.health);
        }
        const refreshed = await fetchJson("/api/moments");
        uiState.moments = refreshed.moments || [];
        renderActiveTab();
        return;
      }
      const labelMap = {
        "rejected-daughter": "daughter",
        "rejected-adult": "adult_only",
        "rejected-uncertain": "uncertain",
      };
      const label = labelMap[action];
      if (!label) return;
      const res = await fetchJson(`/api/rejected/${encodeURIComponent(name)}/feedback`, {
        method: "POST",
        body: JSON.stringify({ label }),
      });
      const target = uiState.rejected.find((item) => item.name === name);
      if (target) target.feedback_label = label;
      if (uiState.health && res.summary) {
        uiState.health.feedback = res.summary;
        renderHealth(uiState.health);
      }
      renderActiveTab();
      return;
    }

    const id = Number(card.dataset.id);
    if (action === "feedback-daughter") {
      const res = await fetchJson(`/api/moments/${id}/feedback`, {
        method: "POST",
        body: JSON.stringify({ label: "daughter", delete_clip: false }),
      });
      const target = uiState.moments.find((item) => item.id === id);
      if (target) target.feedback_label = "daughter";
      if (uiState.health && res.summary) {
        uiState.health.feedback = res.summary;
        renderHealth(uiState.health);
      }
      renderActiveTab();
      return;
    }

    if (action === "feedback-adult-delete") {
      const confirmed = window.confirm(
        "确认标记为「纯大人」并删除该片段？（原片会转存至被拒样本库用于后续优化）",
      );
      if (!confirmed) return;
      const res = await fetchJson(`/api/moments/${id}/feedback`, {
        method: "POST",
        body: JSON.stringify({ label: "adult_only", delete_clip: true }),
      });
      uiState.moments = uiState.moments.filter((item) => item.id !== id);
      uiState.rejectedLoaded = false;
      if (uiState.health && res.summary) {
        uiState.health.feedback = res.summary;
        renderHealth(uiState.health);
      }
      renderActiveTab();
      return;
    }

    if (action === "favorite") {
      const favorited = card.dataset.favorited !== "true";
      await fetchJson(`/api/moments/${id}/favorite`, {
        method: "POST",
        body: JSON.stringify({ favorited }),
      });
      await loadAll();
      return;
    }

    if (action === "delete") {
      const confirmed = window.confirm("Delete this saved clip and metadata?");
      if (!confirmed) return;
      await fetchJson(`/api/moments/${id}`, { method: "DELETE" });
      await loadAll();
    }
  } finally {
    button.disabled = false;
  }
});

refreshButton.addEventListener("click", loadAll);
loadAll().catch((error) => {
  statusGrid.innerHTML = `<article class="status-card error"><strong>Load failed</strong><small>${escapeHtml(
    error.message,
  )}</small></article>`;
});
