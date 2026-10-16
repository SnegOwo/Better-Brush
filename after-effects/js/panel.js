/*
 * Better Brush — After Effects panel logic.
 *
 * Thin client by design:
 *   - reads the current selection through ExtendScript (js/host.jsx),
 *   - runs the AI work in the local headless engine over HTTP (127.0.0.1),
 *   - applies the finished alpha movie back to the layer via ExtendScript.
 *
 * One button, one character. Anything ambiguous is handed to the full app.
 */
(function () {
  "use strict";

  var DEFAULT_CONFIG = {
    port: 8765,
    pythonwPath: "",
    pythonPath: "",
    projectRoot: ""
  };

  var POLL_MS = 800;
  var BOOT_TIMEOUT_MS = 120000;
  var SELECTION_POLL_MS = 1500;
  var HEALTH_POLL_MS = 5000;
  var PRO_REQUIRED = "__pro_required__";

  /* Engine progress stages, in the order the pipeline reports them. "load" is
   * the cold-start model load: it has no frame count, so it reports 0/0 and
   * the bar runs indeterminate rather than sitting still at zero. "matte" only
   * appears when the soft-matte pass runs. */
  var STAGES = ["load", "extract", "analyze", "propagate", "matte", "export"];
  var STAGE_TEXT = {
    load: "Loading the model",
    extract: "Reading the source",
    analyze: "Finding the character",
    propagate: "Tracking through the clip",
    matte: "Refining the edges",
    export: "Writing the alpha movie"
  };

  /* Brush overlay colours, shared by the live canvas and the exported stroke
   * masks. Ground truth for "add" vs "remove" is the tool, not the colour. */
  var BRUSH_COLOURS = {
    add: "rgba(74, 222, 128, 0.45)",
    remove: "rgba(248, 113, 113, 0.45)"
  };

  /* Short model names (from /health capabilities) and the ModelSize values the
   * engine reports with a finished job both resolve through this table. */
  var MODEL_LABEL = {
    tiny: "SAM 2.1 Tiny",
    small: "SAM 2.1 Small",
    base_plus: "SAM 2.1 Base+",
    large: "SAM 2.1 Large",
    "sam2.1_hiera_tiny": "SAM 2.1 Tiny",
    "sam2.1_hiera_small": "SAM 2.1 Small",
    "sam2.1_hiera_base_plus": "SAM 2.1 Base+",
    "sam2.1_hiera_large": "SAM 2.1 Large",
    "custom_finetune": "fine-tuned model"
  };

  var ERROR_TITLES = {
    no_character: "No character found",
    multiple_subjects: "Several subjects in view",
    unsupported: "Not supported here",
    busy: "Engine busy",
    bad_request: "Rejected by the engine",
    internal: "Something went wrong"
  };

  /* Inline SVG so the panel needs no external assets. */
  var ICONS = {
    film:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" ' +
      'stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" ' +
      'height="16" rx="2"/><path d="M8 4v16M16 4v16M3 12h18"/></svg>',
    spinner:
      '<svg class="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
      'stroke-width="2" stroke-linecap="round"><path d="M12 3a9 9 0 1 0 9 9" ' +
      'opacity=".9"/></svg>',
    check:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
      'stroke-linecap="round" stroke-linejoin="round"><path d="M4 12.5 9.5 18 20 6.5"/></svg>',
    alert:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" ' +
      'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3.5 22 20H2z"/>' +
      '<path d="M12 10v4.5M12 17.4v.1"/></svg>'
  };

  var ui = {};
  var state = {
    config: DEFAULT_CONFIG,
    capabilities: null,
    health: null,
    engine: "unknown",
    selection: null,
    jobId: null,
    mode: "empty",
    running: false,
    jobTimer: null,
    selectionTimer: null,
    healthTimer: null,
    tickTimer: null,
    startedAt: 0,
    lastStageIndex: -1,
    lastPercent: 0,
    quality: "balanced",
    outputMode: "replace",
    matting: true,
    previewPath: null,
    previewSize: { w: 0, h: 0 },
    /* The original selection, kept apart from state.selection: once a run has
     * replaced the layer's source, state.selection points at the cut-out, and a
     * correction must still read the original footage. */
    source: null,
    /* Where the next apply goes: the original layer for a fresh run, the
     * result layer for a correction. */
    apply: null,
    resumeDir: null,
    brush: { tool: "add", size: 48, strokes: [], stroke: null, painting: false },
    /* Which frame the brush is showing. The panel can render any frame of a
     * finished result, so a correction can target the frame that is wrong. */
    frameCount: 0,
    keyIdx: 0,
    brushFrame: 0,
    frameUrl: null,
    frameTimer: null,
    resultModel: "",
    doneTitle: "",
    doneText: "",
    doneMeta: "",
    errTitle: "",
    errText: "",
    errHandOff: false,
    errDetail: "",
    readyNote: ""
  };

  /* ------------------------------------------------------------------ */
  /* DOM helpers                                                         */
  /* ------------------------------------------------------------------ */
  function id(name) {
    return document.getElementById(name);
  }

  function show(el, visible) {
    el.classList.toggle("hidden", !visible);
  }

  function modelLabel(name) {
    if (name === "custom" || name === "custom_finetune") {
      return (state.capabilities && state.capabilities.custom_label) ||
        "fine-tuned model";
    }
    return MODEL_LABEL[name] || name || "—";
  }

  function formatFps(fps) {
    if (!fps) {
      return "";
    }
    var rounded = Math.round(fps * 100) / 100;
    return String(rounded) + " fps";
  }

  function formatElapsed(ms) {
    var total = Math.max(0, Math.floor(ms / 1000));
    var minutes = Math.floor(total / 60);
    var seconds = total % 60;
    return minutes + ":" + (seconds < 10 ? "0" : "") + seconds;
  }

  /* CEP serves this page over file://, so a local preview can be shown
   * directly — that is the whole point of writing preview.png next to the
   * alpha movie. */
  function fileUrl(path) {
    if (!path) {
      return "";
    }
    var text = String(path).replace(/\\/g, "/");
    if (text.charAt(0) !== "/") {
      text = "/" + text;
    }
    return "file://" + encodeURI(text);
  }

  function setStatus(kind, text) {
    ui.statusDot.className = "dot" + (kind ? " " + kind : "");
    ui.statusText.textContent = text;
  }

  /* ------------------------------------------------------------------ */
  /* ExtendScript bridge                                                 */
  /* ------------------------------------------------------------------ */
  function bbEval(script) {
    return new Promise(function (resolve, reject) {
      if (!window.__adobe_cep__ || !window.__adobe_cep__.evalScript) {
        reject(new Error("Not running inside After Effects."));
        return;
      }
      window.__adobe_cep__.evalScript(script, function (result) {
        resolve(result);
      });
    });
  }

  function parseJson(text) {
    if (typeof text !== "string") {
      return null;
    }
    var trimmed = text.replace(/^\s+|\s+$/g, "");
    if (trimmed.charAt(0) !== "{") {
      return null;
    }
    try {
      return JSON.parse(trimmed);
    } catch (err) {
      return null;
    }
  }

  function callHost(script) {
    return bbEval(script).then(function (raw) {
      return parseJson(raw);
    });
  }

  function csString(value) {
    return JSON.stringify(value == null ? "" : String(value));
  }

  /* ------------------------------------------------------------------ */
  /* Config + engine                                                     */
  /* ------------------------------------------------------------------ */
  function loadConfig() {
    return fetch("config.json", { cache: "no-store" })
      .then(function (resp) {
        return resp.ok ? resp.json() : null;
      })
      .catch(function () {
        return null;
      })
      .then(function (cfg) {
        if (cfg && typeof cfg === "object") {
          state.config = Object.assign({}, DEFAULT_CONFIG, cfg);
        }
        return state.config;
      });
  }

  function baseUrl() {
    return "http://127.0.0.1:" + (state.config.port || 8765);
  }

  function getJson(url, timeoutMs) {
    var options = { cache: "no-store" };
    var controller = null;
    var timer = null;
    if (typeof AbortController !== "undefined") {
      controller = new AbortController();
      options.signal = controller.signal;
      timer = setTimeout(function () {
        controller.abort();
      }, timeoutMs);
    }
    return fetch(url, options)
      .then(function (resp) {
        return resp.ok ? resp.json() : null;
      })
      .catch(function () {
        return null;
      })
      .then(function (data) {
        if (timer) {
          clearTimeout(timer);
        }
        return data;
      });
  }

  /* `full` also asks the engine for its capability snapshot (installed models,
   * device). That probe touches the hardware, so it is only fetched at start-up
   * and once per finished job — never on the 5 s health poll. */
  function refreshHealth(full) {
    var url = baseUrl() + "/health" + (full ? "?full=1" : "");
    return getJson(url, full ? 20000 : 1500).then(function (health) {
      applyHealth(health);
      return health;
    });
  }

  function applyHealth(health) {
    if (!health) {
      state.health = null;
      state.engine = "offline";
      setStatus("offline", "engine offline");
      render();
      return;
    }
    state.health = health;
    state.engine = "online";
    if (health.capabilities) {
      state.capabilities = health.capabilities;
    }
    setStatus(health.busy ? "busy" : "online", health.busy ? "working" : "ready");
    if (!state.running) {
      adoptRunningJob(health);
    }
    render();
  }

  function waitForEngine(deadline) {
    return getJson(baseUrl() + "/health", 1500).then(function (health) {
      if (health) {
        applyHealth(health);
        return true;
      }
      if (Date.now() > deadline) {
        throw new Error(
          "The engine did not start in time. Open Better Brush once to finish setup."
        );
      }
      return new Promise(function (resolve) {
        setTimeout(resolve, 1000);
      }).then(function () {
        return waitForEngine(deadline);
      });
    });
  }

  function ensureEngine() {
    return getJson(baseUrl() + "/health", 1500).then(function (health) {
      if (health) {
        applyHealth(health);
        return true;
      }
      setStatus("busy", "starting");
      render();
      return callHost(
        "bbLaunchEngine(" +
          csString(state.config.pythonwPath) + "," +
          csString(state.config.projectRoot) + "," +
          (state.config.port || 8765) +
        ")"
      ).then(function (res) {
        if (res && res.ok === false) {
          throw new Error(res.reason || "Could not start the engine.");
        }
        return waitForEngine(Date.now() + BOOT_TIMEOUT_MS);
      });
    });
  }

  /* ------------------------------------------------------------------ */
  /* Selection                                                           */
  /* ------------------------------------------------------------------ */
  /* Skipped while a job runs — the layer's source is being replaced — except
   * when the panel adopted a job it did not start and has no layer to show. */
  function refreshSelection() {
    if (state.running && state.selection) {
      return Promise.resolve();
    }
    return callHost("bbSelectedLayer()")
      .then(function (sel) {
        state.selection = sel;
      })
      .catch(function () {
        state.selection = null;
      })
      .then(function () {
        render();
      });
  }

  function selectionMeta(sel) {
    var parts = [];
    if (sel.comp) {
      parts.push(sel.comp);
    }
    if (sel.width && sel.height) {
      parts.push(sel.width + "×" + sel.height);
    }
    parts.push(sel.isStill ? "still image" : sel.frames + " frames");
    if (!sel.isStill && sel.fps) {
      parts.push(formatFps(sel.fps));
    }
    return parts.join(" · ");
  }

  /* ------------------------------------------------------------------ */
  /* Rendering                                                           */
  /* ------------------------------------------------------------------ */
  function render() {
    var sel = state.selection;
    var pro = state.health ? !!state.health.pro : true;
    var plan = {
      tone: "",
      icon: "film",
      title: "Select a footage layer",
      text: "Pick a layer in the timeline, then click Remove Background.",
      action: { label: "Refresh", value: "refresh" },
      chip: false,
      stages: false,
      preview: false,
      brush: false,
      elapsed: false
    };

    if (state.mode === "running") {
      plan.tone = "tone-run";
      plan.icon = "spinner";
      plan.title = "Removing the background";
      plan.text = "Tracking the character through the clip. Long shots take a while.";
      plan.action = null;
      plan.chip = !!(sel && sel.ok);
      plan.stages = true;
      plan.elapsed = true;
    } else if (state.mode === "done") {
      plan.tone = "tone-ok";
      plan.icon = "check";
      plan.title = state.doneTitle;
      plan.text = state.doneText;
      plan.action = { label: "Run again", value: "again" };
      plan.chip = !!(sel && sel.ok);
      plan.preview = !!state.previewPath;
      plan.brush = !!state.resumeDir;
    } else if (state.mode === "error") {
      plan.tone = "tone-err";
      plan.icon = "alert";
      plan.title = state.errTitle || ERROR_TITLES.internal;
      plan.text = state.errText;
      plan.action = state.errHandOff
        ? { label: "Open full app", value: "openapp" }
        : null;
      plan.chip = !!(sel && sel.ok);
    } else if (sel && sel.ok) {
      plan.title = "Ready";
      plan.text = sel.isStill
        ? "Still image — the background behind the character will be removed."
        : "One character, one click. Clips with several subjects open in the full app.";
      plan.action = null;
      plan.chip = true;
    } else if (sel && sel.reason) {
      plan.title = "Select a footage layer";
      plan.text = sel.reason;
      plan.action = { label: "Refresh", value: "refresh" };
    }

    if (state.mode !== "running" && state.readyNote && state.mode !== "done") {
      plan.text = state.readyNote;
    }

    ui.hero.className = "hero" + (plan.tone ? " " + plan.tone : "");
    ui.heroIcon.innerHTML = ICONS[plan.icon] || ICONS.film;
    ui.heroTitle.textContent = plan.title;
    ui.heroText.textContent = plan.text || "";

    show(ui.layerChip, plan.chip);
    if (plan.chip && sel) {
      ui.layerName.textContent = sel.layer || "—";
      ui.layerMeta.textContent = selectionMeta(sel);
    }

    show(ui.stages, plan.stages);
    show(ui.progressWrap, plan.stages);
    show(ui.progressMeta, plan.stages);
    show(ui.elapsed, plan.elapsed);
    show(ui.previewWrap, plan.preview);
    if (plan.preview && state.previewPath) {
      ui.previewImg.src = fileUrl(state.previewPath);
      ui.previewCaption.textContent = state.doneMeta
        ? "Key frame — " + state.doneMeta
        : "";
    }
    show(ui.brushTools, plan.brush);
    show(ui.brushCanvas, plan.brush);
    if (plan.brush) {
      syncBrushCanvas();
    }

    show(ui.heroAction, !!plan.action);
    if (plan.action) {
      ui.heroAction.textContent = plan.action.label;
      ui.heroAction.setAttribute("data-action", plan.action.value);
    }

    // Pro is a client-side gate, same as the desktop app.
    show(ui.notice, !pro);
    if (!pro) {
      ui.notice.className = "notice";
      ui.notice.textContent =
        "Better Brush Pro is required. Activate it in the desktop app, then reopen this panel.";
    }

    ui.noticeDetail.textContent = state.errDetail || "";
    show(ui.noticeDetail, !!state.errDetail);
    ui.noticeDetail.className = "notice-detail";

    ui.runBtn.disabled = state.running || !pro || !(sel && sel.ok);
    show(ui.cancelBtn, state.running);
    updateModelNote();
    updateMattingNote();
  }

  function updateModelNote() {
    var caps = state.capabilities;
    var map = caps && caps.quality_models;
    var name = map ? map[state.quality] : null;
    ui.modelNote.textContent = name ? "Uses " + modelLabel(name) : "—";
  }

  function mattingAvailable() {
    /* Older engines do not report the capability, so only an explicit false
     * disables the switch. */
    return !(state.capabilities && state.capabilities.matting === false);
  }

  function updateMattingNote() {
    var available = mattingAvailable();
    ui.mattingToggle.disabled = !available || state.running;
    if (!available) {
      ui.mattingToggle.checked = false;
      ui.mattingNote.textContent = "Matting model not installed.";
    } else if (state.matting) {
      ui.mattingNote.textContent = "Softens hair and edges — slower.";
    } else {
      ui.mattingNote.textContent = "Edges follow the tracked outline.";
    }
    ui.mattingToggle.parentNode.className =
      "toggle" + (available ? "" : " disabled");
  }

  /* ------------------------------------------------------------------ */
  /* Brush correction                                                    */
  /* ------------------------------------------------------------------ */
  function syncBrushCanvas() {
    var size = state.previewSize;
    if (!size || !size.w || !size.h) {
      return;
    }
    ui.brushStage.style.setProperty("--ar", size.w + " / " + size.h);
    if (ui.brushCanvas.width !== size.w || ui.brushCanvas.height !== size.h) {
      ui.brushCanvas.width = size.w;
      ui.brushCanvas.height = size.h;
    }
    redrawBrush();
  }

  function brushContext() {
    return ui.brushCanvas.getContext("2d");
  }

  function redrawBrush() {
    var ctx = brushContext();
    ctx.clearRect(0, 0, ui.brushCanvas.width, ui.brushCanvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    var strokes = state.brush.strokes;
    for (var i = 0; i < strokes.length; i++) {
      drawStroke(ctx, strokes[i]);
    }
    if (state.brush.stroke) {
      drawStroke(ctx, state.brush.stroke);
    }
  }

  function drawStroke(ctx, stroke, colour) {
    if (!stroke || !stroke.points.length) {
      return;
    }
    var paint = colour || BRUSH_COLOURS[stroke.tool] || BRUSH_COLOURS.add;
    ctx.strokeStyle = paint;
    ctx.fillStyle = paint;
    ctx.lineWidth = stroke.width;
    var points = stroke.points;
    if (points.length === 1) {
      ctx.beginPath();
      ctx.arc(points[0].x, points[0].y, stroke.width / 2, 0, Math.PI * 2);
      ctx.fill();
      return;
    }
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (var i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.stroke();
  }

  function canvasPoint(event) {
    var rect = ui.brushCanvas.getBoundingClientRect();
    var scaleX = ui.brushCanvas.width / (rect.width || 1);
    var scaleY = ui.brushCanvas.height / (rect.height || 1);
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
      scale: (scaleX + scaleY) / 2
    };
  }

  function brushDown(event) {
    if (state.running || !state.resumeDir) {
      return;
    }
    event.preventDefault();
    var point = canvasPoint(event);
    state.brush.painting = true;
    state.brush.stroke = {
      tool: state.brush.tool,
      width: Math.max(2, state.brush.size * point.scale),
      points: [{ x: point.x, y: point.y }]
    };
    try {
      ui.brushCanvas.setPointerCapture(event.pointerId);
    } catch (err) { /* pointer capture is best-effort */ }
    redrawBrush();
  }

  function brushMove(event) {
    if (!state.brush.painting || !state.brush.stroke) {
      return;
    }
    var point = canvasPoint(event);
    state.brush.stroke.points.push({ x: point.x, y: point.y });
    redrawBrush();
  }

  function brushUp() {
    if (!state.brush.painting) {
      return;
    }
    state.brush.painting = false;
    if (state.brush.stroke) {
      state.brush.strokes.push(state.brush.stroke);
      state.brush.stroke = null;
    }
    redrawBrush();
  }

  function clearBrush() {
    state.brush.strokes = [];
    state.brush.stroke = null;
    redrawBrush();
  }

  /* ------------------------------------------------------------------ */
  /* Frame scrubber                                                      */
  /* ------------------------------------------------------------------ */
  function frameLabel(index) {
    return (index + 1) + "/" + (state.frameCount || 0);
  }

  function resetFrameView() {
    var count = state.frameCount || 0;
    var max = count > 0 ? count - 1 : 0;
    ui.brushFrame.min = "0";
    ui.brushFrame.max = String(max);
    var index = Math.max(0, Math.min(state.keyIdx || 0, max));
    ui.brushFrame.value = String(index);
    state.brushFrame = index;
    ui.brushFrameLabel.textContent = count ? frameLabel(index) : "—";
  }

  /* Pull one frame's cut-out from the engine. This is what makes a correction
   * target the frame that is actually wrong instead of the key frame. */
  function loadFrame(index) {
    state.brushFrame = index;
    ui.brushFrameLabel.textContent = frameLabel(index);
    if (!state.resumeDir) {
      return;
    }
    var url =
      baseUrl() +
      "/frame?dir=" + encodeURIComponent(state.resumeDir) +
      "&index=" + index;
    fetch(url)
      .then(function (resp) {
        return resp.ok ? resp.blob() : null;
      })
      .then(function (blob) {
        if (!blob) {
          return;
        }
        if (state.frameUrl) {
          URL.revokeObjectURL(state.frameUrl);
        }
        state.frameUrl = URL.createObjectURL(blob);
        ui.previewImg.src = state.frameUrl;
      })
      .catch(function () { /* a failed fetch keeps the previous frame */ });
  }

  function scrubFrame() {
    var index = parseInt(ui.brushFrame.value, 10) || 0;
    ui.brushFrameLabel.textContent = frameLabel(index);
    /* Strokes belong to the frame they were painted on: moving the playhead
     * moves on, so the overlay never lands on the wrong frame. */
    if (index !== state.brushFrame && state.brush.strokes.length) {
      clearBrush();
    }
    if (state.frameTimer) {
      clearTimeout(state.frameTimer);
    }
    /* Debounce: dragging the handle would otherwise decode a frame per pixel. */
    state.frameTimer = setTimeout(function () {
      state.frameTimer = null;
      loadFrame(index);
    }, 140);
  }

  /* Rasterise one tool's strokes into a base64 PNG in source resolution, or
   * null when nothing was painted with that tool. */
  function rasterizeStrokes(tool) {
    var size = state.previewSize;
    var strokes = state.brush.strokes.filter(function (stroke) {
      return stroke.tool === tool;
    });
    if (!strokes.length || !size.w || !size.h) {
      return null;
    }
    var canvas = document.createElement("canvas");
    canvas.width = size.w;
    canvas.height = size.h;
    var ctx = canvas.getContext("2d");
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    strokes.forEach(function (stroke) {
      drawStroke(ctx, stroke, "#ffffff");
    });
    return canvas.toDataURL("image/png");
  }

  function applyCorrection() {
    if (state.running || !state.resumeDir || !state.source) {
      return;
    }
    var add = rasterizeStrokes("add");
    var remove = rasterizeStrokes("remove");
    if (!add && !remove) {
      return;
    }
    startCorrection(add, remove);
  }

  function renderProgress(stage, current, total, message) {
    var idx = STAGES.indexOf(stage);
    if (idx < 0) {
      idx = state.lastStageIndex;
    }
    if (idx > state.lastStageIndex) {
      state.lastStageIndex = idx;
    }
    idx = state.lastStageIndex;

    var items = ui.stages.children;
    for (var i = 0; i < items.length; i++) {
      items[i].className = i < idx ? "done" : (i === idx ? "active" : "");
    }

    var within = total > 0 ? Math.min(1, current / total) : 0;
    var percent = ((idx + within) / STAGES.length) * 100;
    if (percent < state.lastPercent) {
      percent = state.lastPercent;
    }
    state.lastPercent = percent;
    /* A stage with no frame count (loading the model) has nothing to divide,
     * so it sweeps instead of sitting at zero percent. The inline width has to
     * step aside or it would beat the stylesheet's animation. */
    var indeterminate = total <= 0;
    ui.progressWrap.classList.toggle("indeterminate", indeterminate);
    ui.progressBar.style.width = indeterminate ? "" : Math.round(percent) + "%";

    var label = STAGE_TEXT[STAGES[idx]] || "Working";
    ui.progressMeta.textContent =
      label + (message ? " — " + message : "") + "  ·  " + Math.round(percent) + "%";
  }

  /* ------------------------------------------------------------------ */
  /* Run                                                                 */
  /* ------------------------------------------------------------------ */
  function startTick() {
    stopTick();
    state.tickTimer = setInterval(function () {
      ui.elapsed.textContent = formatElapsed(Date.now() - state.startedAt);
    }, 500);
    ui.elapsed.textContent = formatElapsed(0);
  }

  function stopTick() {
    if (state.tickTimer) {
      clearInterval(state.tickTimer);
      state.tickTimer = null;
    }
  }

  function beginRun() {
    state.mode = "running";
    state.running = true;
    state.jobId = null;
    state.previewPath = null;
    state.errDetail = "";
    state.readyNote = "";
    state.lastStageIndex = -1;
    state.lastPercent = 0;
    state.startedAt = Date.now();
    /* A correction consumes the strokes it was built from, so the surface
     * starts clean for the next pass. */
    state.brush.strokes = [];
    state.brush.stroke = null;
    startTick();
    /* Open on the first stage the engine reports ("load"), or the monotonic
     * stage index below would mark it done and sit on "reading source". */
    renderProgress("load", 0, 0, "starting");
    render();
  }

  function jobBody(src, extra) {
    var body = {
      input_path: src.file,
      frame_in: 0,
      frame_out: -1,
      fps: src.fps,
      width: src.width,
      height: src.height,
      preset: "anime",
      quality: state.quality,
      matting: !!state.matting && mattingAvailable()
    };
    if (extra) {
      Object.keys(extra).forEach(function (key) {
        body[key] = extra[key];
      });
    }
    return body;
  }

  function postJob(body) {
    return fetch(baseUrl() + "/job", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (resp) {
      return resp.json().then(function (data) {
        if (resp.status === 202 && data && data.id) {
          return data;
        }
        var error = data && data.error ? data.error : null;
        var message = error ? error.message : "HTTP " + resp.status;
        if (error && error.kind === "busy") {
          message = "The engine is already working on another job. Try again shortly.";
        }
        var wrapped = new Error(message);
        wrapped.kind = error ? error.kind : "internal";
        throw wrapped;
      });
    });
  }

  function startJob() {
    if (state.running) {
      return;
    }
    var sel = state.selection;
    if (!sel || !sel.ok) {
      return;
    }
    /* Freeze the original footage and the apply target now: after the run
     * replaces the layer's source, state.selection will describe the cut-out. */
    state.source = {
      file: sel.file,
      fps: sel.fps,
      width: sel.width,
      height: sel.height,
      compId: sel.compId,
      layerId: sel.layerId
    };
    state.apply = {
      compId: sel.compId,
      layerId: sel.layerId,
      mode: state.outputMode
    };
    state.resumeDir = null;
    beginRun();

    ensureEngine()
      .then(function () {
        if (state.health && state.health.pro === false) {
          throw new Error(PRO_REQUIRED);
        }
        return postJob(jobBody(state.source));
      })
      .then(function (job) {
        state.jobId = job.id;
        pollJob();
      })
      .catch(function (err) {
        var message = err && err.message ? err.message : String(err);
        if (message === PRO_REQUIRED) {
          failRun(
            "Better Brush Pro is required for the After Effects panel.",
            "internal",
            false
          );
        } else {
          failRun(message, err && err.kind ? err.kind : "internal", false);
        }
      });
  }

  /* Re-run the clip from the stored key frame with the painted strokes layered
   * over the seed mask, then replace the result layer with the new cut-out. */
  function startCorrection(addPng, removePng) {
    if (state.running || !state.resumeDir || !state.source) {
      return;
    }
    var body = jobBody(state.source, {
      resume_dir: state.resumeDir,
      correction_frame: state.brushFrame,
      correction_add: addPng || null,
      correction_remove: removePng || null
    });
    /* A correction always replaces the current result layer, whichever mode
     * produced it. */
    state.apply = {
      compId: state.apply ? state.apply.compId : null,
      layerId: state.apply ? state.apply.layerId : null,
      mode: "replace"
    };
    beginRun();

    ensureEngine()
      .then(function () {
        return postJob(body);
      })
      .then(function (job) {
        state.jobId = job.id;
        pollJob();
      })
      .catch(function (err) {
        failRun(
          err && err.message ? err.message : String(err),
          err && err.kind ? err.kind : "internal",
          false
        );
      });
  }

  function pollJob() {
    if (!state.jobId) {
      return;
    }
    getJson(baseUrl() + "/job/" + state.jobId, 10000)
      .then(function (job) {
        if (job) {
          onJob(job);
        } else {
          schedulePoll();
        }
      })
      .catch(function () {
        schedulePoll();
      });
  }

  function schedulePoll() {
    state.jobTimer = setTimeout(pollJob, POLL_MS);
  }

  function onJob(job) {
    if (job.state === "done") {
      finishJob(job.result);
      return;
    }
    if (job.state === "error") {
      var error = job.error || {};
      var kind = error.kind || "internal";
      failRun(error.message || "Something went wrong.", kind, isHandOff(kind));
      return;
    }
    if (job.state === "cancelled") {
      endRun();
      state.mode = "ready";
      state.readyNote = "Cancelled.";
      render();
      refreshSelection();
      return;
    }

    var progress = job.progress || {};
    renderProgress(
      progress.stage || "extract",
      progress.current || 0,
      progress.total || 0,
      progress.message || ""
    );
    schedulePoll();
  }

  function isHandOff(kind) {
    return kind === "no_character" || kind === "multiple_subjects";
  }

  function endRun() {
    state.running = false;
    state.jobId = null;
    stopTick();
    if (state.jobTimer) {
      clearTimeout(state.jobTimer);
      state.jobTimer = null;
    }
  }

  function failRun(message, kind, handOff) {
    endRun();
    state.mode = "error";
    state.errTitle = ERROR_TITLES[kind] || ERROR_TITLES.internal;
    state.errText = message;
    state.errHandOff = !!handOff;
    state.errDetail =
      kind === "internal" && state.health && state.health.log_path
        ? "Details: " + state.health.log_path
        : "";
    render();
    if (handOff) {
      openFullApp();
    }
  }

  function finishJob(result) {
    if (!result) {
      failRun("The engine finished without returning a result.", "internal", false);
      return;
    }
    var apply = state.apply;
    if (!apply && state.selection) {
      /* Adopted a job the panel did not start: fall back to the live selection
       * rather than an unknown target. */
      apply = {
        compId: state.selection.compId,
        layerId: state.selection.layerId,
        mode: state.outputMode
      };
    }
    apply = apply || {};
    var script =
      "bbApplyResult(" +
      csString(result.output_path) + "," +
      csString(apply.mode || state.outputMode) + "," +
      (result.fps || 0) + "," +
      (apply.compId != null ? apply.compId : -1) + "," +
      (apply.layerId != null ? apply.layerId : -1) +
      ")";

    callHost(script)
      .then(function (res) {
        if (!res || !res.ok) {
          failRun(
            "Processed, but applying it to After Effects failed: " +
              (res && res.reason ? res.reason : "unknown reason"),
            "internal",
            false
          );
          return;
        }
        endRun();
        state.mode = "done";
        state.resultModel = result.model || "";
        state.previewPath = result.preview_path || null;
        state.previewSize = { w: result.width || 0, h: result.height || 0 };
        state.resumeDir = result.work_dir || null;
        state.frameCount = result.frame_count || 0;
        state.keyIdx = result.key_idx || 0;
        if (state.frameUrl) {
          URL.revokeObjectURL(state.frameUrl);
          state.frameUrl = null;
        }
        clearBrush();
        resetFrameView();
        /* From here on, applying goes to the layer that now holds the cut-out,
         * so a correction replaces this result rather than the original. */
        state.apply = {
          compId: res.compId != null ? res.compId : apply.compId,
          layerId: res.layerId != null ? res.layerId : apply.layerId,
          mode: "replace"
        };
        state.doneTitle =
          "Done — " + result.frame_count +
          (result.frame_count === 1 ? " frame" : " frames");
        state.doneText =
          state.outputMode === "replace"
            ? "The layer's source was replaced with the cut-out."
            : "Added as a new footage layer above the original.";
        if (result.matting) {
          state.doneText += " Soft matting applied.";
        }
        state.doneMeta =
          result.frame_count + (result.frame_count === 1 ? " frame" : " frames") +
          " · " + result.width + "×" + result.height +
          " · " + modelLabel(result.model) +
          (result.matting ? " · matted" : "");
        render();
        refreshHealth(true);
        refreshSelection();
      })
      .catch(function (err) {
        failRun(
          "Processed, but applying it to After Effects failed: " +
            (err && err.message ? err.message : String(err)),
          "internal",
          false
        );
      });
  }

  function cancelJob() {
    if (!state.jobId) {
      return;
    }
    fetch(baseUrl() + "/job/" + state.jobId + "/cancel", { method: "POST" })
      .catch(function () {})
      .then(function () {
        ui.progressMeta.textContent = "Cancelling…";
      });
  }

  function openFullApp() {
    /* Pass the original footage, not the current selection: after a replace the
     * selection describes the cut-out, and the app should open the source clip. */
    var file = state.source
      ? state.source.file
      : state.selection
        ? state.selection.file
        : "";
    return callHost(
      "bbOpenFullApp(" +
        csString(state.config.pythonPath) + "," +
        csString(state.config.projectRoot) + "," +
        csString(file) +
      ")"
    ).catch(function () {});
  }

  /* Attach to a job the engine is already running (the panel was reopened, or
   * another panel instance started one) so progress is never invisible. */
  function adoptRunningJob(health) {
    if (!health || !health.busy || !health.job || !health.job.id || state.running) {
      return false;
    }
    beginRun();
    state.jobId = health.job.id;
    var progress = health.job.progress || {};
    if (progress.stage) {
      renderProgress(
        progress.stage,
        progress.current || 0,
        progress.total || 0,
        progress.message || ""
      );
    }
    pollJob();
    return true;
  }

  /* ------------------------------------------------------------------ */
  /* Settings                                                            */
  /* ------------------------------------------------------------------ */
  function setSegment(node, value) {
    var buttons = node.getElementsByTagName("button");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].className = buttons[i].getAttribute("data-value") === value ? "on" : "";
    }
  }

  function saveSettings() {
    try {
      localStorage.setItem(
        "bb.settings",
        JSON.stringify({
          quality: state.quality,
          mode: state.outputMode,
          matting: state.matting
        })
      );
    } catch (err) { /* storage may be unavailable; not fatal */ }
  }

  function restoreSettings() {
    try {
      var raw = localStorage.getItem("bb.settings");
      if (!raw) {
        return;
      }
      var saved = JSON.parse(raw);
      if (saved && saved.quality) {
        state.quality = saved.quality;
      }
      if (saved && saved.mode) {
        state.outputMode = saved.mode;
      }
      if (saved && typeof saved.matting === "boolean") {
        state.matting = saved.matting;
      }
    } catch (err) { /* ignore malformed settings */ }
  }

  /* ------------------------------------------------------------------ */
  /* Boot                                                                */
  /* ------------------------------------------------------------------ */
  function bindUi() {
    ui = {
      statusDot: id("statusDot"),
      statusText: id("statusText"),
      hero: id("hero"),
      heroIcon: id("heroIcon"),
      heroTitle: id("heroTitle"),
      heroText: id("heroText"),
      heroAction: id("heroAction"),
      layerChip: id("layerChip"),
      layerName: id("layerName"),
      layerMeta: id("layerMeta"),
      elapsed: id("elapsed"),
      stages: id("stages"),
      progressWrap: id("progressWrap"),
      progressBar: id("progressBar"),
      progressMeta: id("progressMeta"),
      previewWrap: id("previewWrap"),
      previewImg: id("previewImg"),
      previewCaption: id("previewCaption"),
      brushStage: id("brushStage"),
      brushCanvas: id("brushCanvas"),
      brushTools: id("brushTools"),
      brushToolSeg: id("brushToolSeg"),
      brushFrame: id("brushFrame"),
      brushFrameLabel: id("brushFrameLabel"),
      brushSize: id("brushSize"),
      brushClear: id("brushClear"),
      brushApply: id("brushApply"),
      notice: id("notice"),
      noticeDetail: id("noticeDetail"),
      qualitySeg: id("qualitySeg"),
      modeSeg: id("modeSeg"),
      modelNote: id("modelNote"),
      mattingToggle: id("mattingToggle"),
      mattingNote: id("mattingNote"),
      runBtn: id("runBtn"),
      cancelBtn: id("cancelBtn"),
      openAppBtn: id("openAppBtn")
    };
  }

  function bindSegments() {
    function closestButton(node) {
      while (node && node.nodeType === 1 && node !== document.body) {
        if (node.getAttribute && node.getAttribute("data-value")) {
          return node;
        }
        node = node.parentNode;
      }
      return null;
    }

    var segs = [
      { node: ui.qualitySeg, key: "quality" },
      { node: ui.modeSeg, key: "outputMode" }
    ];
    segs.forEach(function (seg) {
      seg.node.addEventListener("click", function (event) {
        var button = closestButton(event.target);
        if (!button) {
          return;
        }
        var value = button.getAttribute("data-value");
        state[seg.key] = value;
        setSegment(seg.node, value);
        saveSettings();
        render();
      });
    });
    setSegment(ui.qualitySeg, state.quality);
    setSegment(ui.modeSeg, state.outputMode);
    setSegment(ui.brushToolSeg, state.brush.tool);
    ui.brushToolSeg.addEventListener("click", function (event) {
      var button = closestButton(event.target);
      if (!button) {
        return;
      }
      state.brush.tool = button.getAttribute("data-value");
      setSegment(ui.brushToolSeg, state.brush.tool);
    });
    ui.brushSize.value = String(state.brush.size);
    ui.mattingToggle.checked = state.matting;
  }

  function init() {
    bindUi();
    restoreSettings();
    bindSegments();

    ui.runBtn.addEventListener("click", startJob);
    ui.cancelBtn.addEventListener("click", cancelJob);
    ui.openAppBtn.addEventListener("click", openFullApp);

    ui.mattingToggle.addEventListener("change", function () {
      state.matting = ui.mattingToggle.checked;
      saveSettings();
      render();
    });

    ui.brushCanvas.addEventListener("pointerdown", brushDown);
    ui.brushCanvas.addEventListener("pointermove", brushMove);
    ui.brushCanvas.addEventListener("pointerup", brushUp);
    ui.brushCanvas.addEventListener("pointercancel", brushUp);
    ui.brushCanvas.addEventListener("pointerleave", function () {
      if (state.brush.painting) {
        brushUp();
      }
    });
    ui.brushFrame.addEventListener("input", scrubFrame);
    ui.brushSize.addEventListener("input", function () {
      state.brush.size = parseInt(ui.brushSize.value, 10) || 48;
    });
    ui.brushClear.addEventListener("click", clearBrush);
    ui.brushApply.addEventListener("click", applyCorrection);

    ui.heroAction.addEventListener("click", function () {
      var action = ui.heroAction.getAttribute("data-action");
      if (action === "openapp") {
        openFullApp();
      } else if (action === "refresh") {
        refreshSelection();
      } else if (action === "again") {
        startJob();
      }
    });

    render();
    loadConfig()
      .then(function () {
        return refreshHealth(true);
      })
      .then(function () {
        return refreshSelection();
      })
      .then(function () {
        state.selectionTimer = setInterval(refreshSelection, SELECTION_POLL_MS);
        state.healthTimer = setInterval(function () {
          refreshHealth(false);
        }, HEALTH_POLL_MS);
      });
  }

  window.addEventListener("unload", function () {
    if (state.selectionTimer) { clearInterval(state.selectionTimer); }
    if (state.healthTimer) { clearInterval(state.healthTimer); }
    if (state.jobTimer) { clearTimeout(state.jobTimer); }
    stopTick();
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
