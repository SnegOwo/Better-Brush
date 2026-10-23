/*
 * Better Brush — Premiere Pro panel logic.
 *
 * A port of the After Effects panel: the UI, the engine client and the brush
 * are identical, and everything Premiere-specific lives in js/host.jsx. Kept
 * as its own file rather than shared, because a CEP extension has to be
 * self-contained inside its own folder.
 *
 * Thin client by design:
 *   - reads the current timeline selection through ExtendScript (js/host.jsx),
 *   - runs the AI work in the local headless engine over HTTP (127.0.0.1),
 *   - puts the finished alpha movie back on the timeline via ExtendScript.
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

  /* The cursor ring takes the tool's colour, so the mode is obvious from the
   * brush itself without reading the Add/Remove segment. */
  var BRUSH_INK = {
    add: "#4ade80",
    remove: "#f87171"
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
    selectionJson: "__unset__",
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
    /* What the preview image is currently showing: "preview" for the key-frame
     * still, "frame" for a scrubbed cut-out, "original" for the compare view.
     * render() only ever touches the image in "preview". */
    previewSrc: null,
    previewToken: 0,
    viewing: "preview",
    /* The original selection, kept apart from state.selection: once a run has
     * replaced the layer's source, state.selection points at the cut-out, and a
     * correction must still read the original footage. */
    source: null,
    /* Where the next apply goes: the original layer for a fresh run, the
     * result layer for a correction. */
    apply: null,
    resumeDir: null,
    brush: {
      tool: "add",
      size: 48,
      strokes: [],
      stroke: null,
      painting: false,
      panning: null,
      space: false,
      compare: false
    },
    /* Which part of the frame the brush is looking at: a zoom factor and the
     * top-left of the visible window in normalised source coordinates. Zoom is
     * a CSS transform on #brushLayer, so nothing downstream has to know. */
    view: { zoom: 1, x0: 0, y0: 0 },
    /* Which frame the brush is showing. The panel can render any frame of a
     * finished result, so a correction can target the frame that is wrong. */
    frameCount: 0,
    keyIdx: 0,
    brushFrame: 0,
    frameUrl: null,
    frameIndex: -1,
    sourceUrl: null,
    sourceIndex: -1,
    frameSeq: 0,
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

  /* The token is bumped once per finished run, so a corrected result reloads
   * even though its preview file keeps the same path. */
  function previewUrl(path) {
    return fileUrl(path) + "?v=" + state.previewToken;
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
        reject(new Error("Not running inside Premiere Pro."));
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

  /* A reply that is not JSON means the host script never ran; Premiere answers
   * "EvalScript error." when it throws. Showing that beats showing the
   * empty-selection prompt, which reads as "the panel cannot see my clip". */
  function hostSilent() {
    return {
      ok: false,
      reason: "The host script did not answer the panel. Quit and reopen Premiere Pro."
    };
  }

  function hostMissing() {
    return { ok: false, reason: "The panel is not connected to Premiere Pro." };
  }

  /* ------------------------------------------------------------------ */
  /* Launching programs                                                  */
  /* ------------------------------------------------------------------ */
  /* After Effects gives ExtendScript a `system` object with callSystem on it;
   * Premiere Pro has no `system` object at all. So the panel starts programs
   * itself where CEP hands it a Node.js runtime (the manifest asks for one),
   * and falls back to the host script where it does not. */
  var lastSpawnError = null;

  function nodeRequire(name) {
    try {
      if (typeof require === "function") {
        return require(name);
      }
      if (window.cep_node && window.cep_node.require) {
        return window.cep_node.require(name);
      }
    } catch (err) { /* no Node in this host: the host script takes over */ }
    return null;
  }

  /* Detached and windowless, so closing the host does not close the engine and
   * no console flashes up. Returns whether Node was able to try at all. */
  function spawnDetached(exe, args, cwd) {
    var childProcess = nodeRequire("child_process");
    if (!childProcess || !childProcess.spawn || !exe || !cwd) {
      /* A missing Python path or project root is config.json's business, and the
       * host script reports that in a sentence rather than as a timeout. */
      return false;
    }
    try {
      var child = childProcess.spawn(exe, args, {
        cwd: cwd,
        detached: true,
        stdio: "ignore",
        windowsHide: true
      });
      lastSpawnError = null;
      child.on("error", function (err) {
        lastSpawnError = err;
      });
      if (child && typeof child.unref === "function") {
        child.unref();
      }
      return true;
    } catch (err) {
      lastSpawnError = err;
      return false;
    }
  }

  function launchEngine() {
    var port = state.config.port || 8765;
    var exe = state.config.pythonwPath || state.config.pythonPath;
    if (spawnDetached(
      exe,
      ["-m", "src.engine", "--port", String(port), "--quiet"],
      state.config.projectRoot
    )) {
      return Promise.resolve(null);
    }
    return callHost(
      "bbLaunchEngine(" +
        csString(state.config.pythonwPath) + "," +
        csString(state.config.projectRoot) + "," +
        port +
      ")"
    );
  }

  function launchFullApp(file) {
    var exe = state.config.pythonwPath || state.config.pythonPath;
    var args = ["-m", "src.main"];
    if (file) {
      args.push(file);
    }
    if (spawnDetached(exe, args, state.config.projectRoot)) {
      return Promise.resolve(null);
    }
    return callHost(
      "bbOpenFullApp(" +
        csString(state.config.pythonwPath || state.config.pythonPath) + "," +
        csString(state.config.projectRoot) + "," +
        csString(file) +
      ")"
    );
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
          "The engine did not start in time." +
          (lastSpawnError ? " (" + lastSpawnError.message + ")" : "") +
          " Open Better Brush once to finish setup."
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
      return launchEngine().then(function (res) {
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
    /* The poll runs every 1.5 s, and re-rendering the whole panel on an
     * unchanged selection is pure churn: it re-lays out the brush tools and
     * re-asserts the preview image while the user is painting on it. */
    function apply(sel) {
      var encoded = JSON.stringify(sel || null);
      if (encoded === state.selectionJson) {
        return;
      }
      state.selectionJson = encoded;
      state.selection = sel;
      render();
    }
    return callHost("bbSelectedLayer()")
      .then(function (sel) {
        apply(sel || hostSilent());
      })
      .catch(function () {
        apply(hostMissing());
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
    var plan = {
      tone: "",
      icon: "film",
      title: "Select a clip",
      text: "Pick a clip in the timeline, then click Remove Background.",
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
      plan.text = "Tracking the character. Long shots take a while.";
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
      plan.title = "Select a clip";
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
    if (plan.preview && state.previewPath && state.viewing === "preview") {
      /* The key-frame still, unless the user has scrubbed away from it: the
       * panel re-renders on every poll, and without the guard the preview would
       * snap back to the key frame a second after each scrub. */
      var want = previewUrl(state.previewPath);
      if (want !== state.previewSrc) {
        ui.previewImg.src = want;
        state.previewSrc = want;
      }
      ui.previewCaption.textContent = state.doneMeta
        ? "Key frame — " + state.doneMeta
        : "";
    } else if (plan.preview && state.viewing === "frame") {
      ui.previewCaption.textContent = state.doneMeta
        ? "Frame " + (state.brushFrame + 1) + " of the cut-out — " + state.doneMeta
        : "";
    }
    show(ui.brushTools, plan.brush);
    show(ui.brushCanvas, plan.brush);
    if (!plan.brush) {
      show(ui.brushRing, false);
    }
    if (plan.brush) {
      syncBrushCanvas();
    }

    show(ui.heroAction, !!plan.action);
    if (plan.action) {
      ui.heroAction.textContent = plan.action.label;
      ui.heroAction.setAttribute("data-action", plan.action.value);
    }

    ui.noticeDetail.textContent = state.errDetail || "";
    show(ui.noticeDetail, !!state.errDetail);
    ui.noticeDetail.className = "notice-detail";

    ui.runBtn.disabled = state.running || !(sel && sel.ok);
    show(ui.cancelBtn, state.running);
    updateModelNote();
    updateMattingNote();
  }

  function shortModelLabel(name) {
    return String(modelLabel(name)).replace(/^SAM 2\.1 /, "");
  }

  function updateModelNote() {
    var caps = state.capabilities;
    var map = caps && caps.quality_models;
    var name = map ? map[state.quality] : null;
    /* Terse on purpose: the note sits inside the Quality row, and the button's
     * tooltip plus the done card carry the full model name. */
    ui.modelNote.textContent = name ? shortModelLabel(name) : "—";
  }

  function mattingAvailable() {
    /* Older engines do not report the capability, so only an explicit false
     * disables the switch. */
    return !(state.capabilities && state.capabilities.matting === false);
  }

  function updateMattingNote() {
    var available = mattingAvailable();
    ui.mattingToggle.disabled = !available || state.running;
    /* Only speaks when it has something the user cannot see: the on/off state is
     * the checkbox, and the rest lives in the tooltip. */
    ui.mattingNote.textContent = available ? "" : "Not installed";
    if (!available) {
      ui.mattingToggle.checked = false;
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

  /* ------------------------------------------------------------------ */
  /* Zoom and pan                                                        */
  /* ------------------------------------------------------------------ */
  /* Zoom multiplies the layer that holds both the image and the stroke
   * canvas, so the two can never drift apart at any magnification, and the
   * pointer maths below keeps working unchanged: a zoomed canvas simply
   * reports a proportionally bigger bounding box. */
  var ZOOM_STEPS = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24];
  var ZOOM_MAX = 24;

  function viewSpan() {
    return 1 / state.view.zoom;
  }

  function applyView() {
    var view = state.view;
    var span = viewSpan();
    /* Keep the window inside the frame: the only thing worse than no zoom is a
     * zoom you can pan off the artwork into the checkerboard. */
    view.x0 = Math.max(0, Math.min(view.x0, 1 - span));
    view.y0 = Math.max(0, Math.min(view.y0, 1 - span));
    if (view.zoom <= 1.0001) {
      view.zoom = 1;
      ui.brushLayer.style.transform = "";
      ui.zoomLabel.textContent = "Fit";
      ui.brushStage.classList.remove("pan");
      return;
    }
    ui.brushLayer.style.transform =
      "translate(" + (-view.x0 * view.zoom * 100).toFixed(3) + "%, " +
      (-view.y0 * view.zoom * 100).toFixed(3) + "%) scale(" + view.zoom + ")";
    ui.zoomLabel.textContent = (Math.round(view.zoom * 10) / 10) + "×";
  }

  function resetView() {
    state.view = { zoom: 1, x0: 0, y0: 0 };
    applyView();
  }

  /* Zoom about a point in normalised image coordinates, so whatever is under
   * the cursor stays under the cursor. */
  function zoomTo(zoom, anchor) {
    var view = state.view;
    var oldSpan = viewSpan();
    var next = Math.max(1, Math.min(zoom, ZOOM_MAX));
    var point = anchor;
    if (!point) {
      point = { x: view.x0 + oldSpan / 2, y: view.y0 + oldSpan / 2 };
    }
    var span = 1 / next;
    var ratio = span / oldSpan;
    view.x0 = point.x - (point.x - view.x0) * ratio;
    view.y0 = point.y - (point.y - view.y0) * ratio;
    view.zoom = next;
    applyView();
  }

  function stepZoom(direction) {
    var current = state.view.zoom;
    var next = direction > 0 ? ZOOM_MAX : 1;
    var i;
    if (direction > 0) {
      for (i = 0; i < ZOOM_STEPS.length; i++) {
        if (ZOOM_STEPS[i] > current + 0.001) {
          next = ZOOM_STEPS[i];
          break;
        }
      }
    } else {
      for (i = ZOOM_STEPS.length - 1; i >= 0; i--) {
        if (ZOOM_STEPS[i] < current - 0.001) {
          next = ZOOM_STEPS[i];
          break;
        }
      }
    }
    zoomTo(next);
  }

  /* Where the cursor is in normalised source coordinates. */
  function normalisedPoint(event) {
    var rect = ui.brushCanvas.getBoundingClientRect();
    var span = viewSpan();
    return {
      x: state.view.x0 + ((event.clientX - rect.left) / (rect.width || 1)) * span,
      y: state.view.y0 + ((event.clientY - rect.top) / (rect.height || 1)) * span
    };
  }

  /* Wheel deltas arrive in pixels, lines or pages depending on the device; make
   * them all read as pixels so a trackpad and a mouse feel the same. */
  function wheelDelta(event, axis, rect) {
    var mode = event.deltaMode;
    var factor = mode === 1 ? 16 : mode === 2 ? (axis === "y" ? rect.height : rect.width) : 1;
    return (axis === "y" ? event.deltaY || 0 : event.deltaX || 0) * factor;
  }

  /* The wheel scrolls the image, the way every editor does; Ctrl (or ⌘) with
   * the wheel zooms about the pointer instead, and the +/− buttons step through
   * the fixed zoom levels. */
  function wheelScroll(event) {
    if (!state.resumeDir || state.running || state.brush.compare) {
      return;
    }
    event.preventDefault();
    if (event.ctrlKey || event.metaKey) {
      if (event.deltaY) {
        zoomTo(state.view.zoom * Math.pow(1.0016, -event.deltaY), normalisedPoint(event));
      }
      return;
    }
    if (state.view.zoom <= 1.0001) {
      /* Fitted: there is nothing beyond the frame to scroll to. */
      return;
    }
    var rect = ui.brushStage.getBoundingClientRect();
    var span = viewSpan();
    /* Shift swaps the axis, for a wheel that only reports up/down. */
    state.view.x0 +=
      ((event.shiftKey ? wheelDelta(event, "y", rect) : wheelDelta(event, "x", rect)) /
        (rect.width || 1)) * span;
    state.view.y0 +=
      ((event.shiftKey ? 0 : wheelDelta(event, "y", rect)) / (rect.height || 1)) * span;
    applyView();
  }

  function canPan() {
    return !!state.resumeDir && state.view.zoom > 1.0001;
  }

  function panDown(event) {
    if (!canPan()) {
      return false;
    }
    event.preventDefault();
    state.brush.panning = {
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      rect: ui.brushStage.getBoundingClientRect(),
      x0: state.view.x0,
      y0: state.view.y0
    };
    ui.brushStage.classList.add("panning");
    try {
      ui.brushCanvas.setPointerCapture(event.pointerId);
    } catch (err) { /* pointer capture is best-effort */ }
    return true;
  }

  function panMove(event) {
    var pan = state.brush.panning;
    if (!pan) {
      return;
    }
    var span = viewSpan();
    state.view.x0 =
      pan.x0 - ((event.clientX - pan.x) / (pan.rect.width || 1)) * span;
    state.view.y0 =
      pan.y0 - ((event.clientY - pan.y) / (pan.rect.height || 1)) * span;
    applyView();
  }

  function panUp() {
    if (!state.brush.panning) {
      return;
    }
    state.brush.panning = null;
    ui.brushStage.classList.remove("panning");
  }

  /* The brush footprint, drawn as a DOM circle in screen pixels — the size the
   * slider actually means — so it costs no canvas work on every mouse move. */
  function moveRing(event) {
    if (state.brush.compare || state.brush.panning) {
      show(ui.brushRing, false);
      return;
    }
    var rect = ui.brushStage.getBoundingClientRect();
    var size = Math.max(6, state.brush.size);
    ui.brushRing.style.width = size + "px";
    ui.brushRing.style.height = size + "px";
    ui.brushRing.style.borderColor = BRUSH_INK[state.brush.tool] || "#fff";
    ui.brushRing.style.left = (event.clientX - rect.left) + "px";
    ui.brushRing.style.top = (event.clientY - rect.top) + "px";
    show(ui.brushRing, true);
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
    updateBrushActions();
  }

  /* Apply and Undo have nothing to do until something is painted, so say so. */
  function updateBrushActions() {
    var empty = !state.brush.strokes.length;
    ui.brushApply.disabled = empty;
    ui.brushUndo.disabled = empty;
    ui.brushClear.disabled = empty;
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
    if (state.running || !state.resumeDir || state.brush.compare) {
      return;
    }
    /* Middle-drag and space-drag pan; the left button paints. */
    if (event.button === 1 || state.brush.space) {
      panDown(event);
      return;
    }
    if (event.button !== 0) {
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
    moveRing(event);
    if (state.brush.panning) {
      panMove(event);
      return;
    }
    if (!state.brush.painting || !state.brush.stroke) {
      return;
    }
    var point = canvasPoint(event);
    state.brush.stroke.points.push({ x: point.x, y: point.y });
    redrawBrush();
  }

  function brushUp() {
    panUp();
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

  function undoStroke() {
    if (!state.brush.strokes.length) {
      return;
    }
    state.brush.strokes.pop();
    redrawBrush();
  }

  function clearBrush() {
    state.brush.strokes = [];
    state.brush.stroke = null;
    redrawBrush();
  }

  function setTool(tool) {
    state.brush.tool = tool === "remove" ? "remove" : "add";
    setSegment(ui.brushToolSeg, state.brush.tool);
    ui.brushRing.style.borderColor = BRUSH_INK[state.brush.tool];
  }

  function setBrushSize(size) {
    state.brush.size = Math.max(10, Math.min(160, Math.round(size) || 48));
    ui.brushSize.value = String(state.brush.size);
    show(ui.brushRing, false);
  }

  /* Keyboard speed-ups, live only while a result is on screen. Shortcuts are
   * published in the tooltips rather than a help panel, because a docked panel
   * has no room for one. */
  function typingInField(target) {
    if (!target) {
      return false;
    }
    var tag = (target.tagName || "").toLowerCase();
    if (tag === "textarea" || tag === "select") {
      return true;
    }
    if (tag !== "input") {
      return false;
    }
    var type = (target.getAttribute && target.getAttribute("type")) || "text";
    return type !== "range" && type !== "checkbox" && type !== "radio";
  }

  function onShortcut(event) {
    if (state.running || typingInField(event.target)) {
      return;
    }
    if (event.ctrlKey || event.metaKey) {
      if (!state.resumeDir) {
        return;
      }
      if (event.key === "z" || event.key === "Z") {
        event.preventDefault();
        undoStroke();
      } else if (event.key === "Enter") {
        event.preventDefault();
        applyCorrection();
      }
      return;
    }
    if (!state.resumeDir || event.altKey) {
      return;
    }
    var key = event.key;
    if (key === " " || key === "Spacebar") {
      /* Space turns the left button into a hand, the way paint tools do. */
      state.brush.space = true;
      if (canPan()) {
        ui.brushStage.classList.add("pan");
      }
      event.preventDefault();
    } else if (key === "a" || key === "A") {
      setTool("add");
    } else if (key === "r" || key === "R") {
      setTool("remove");
    } else if (key === "[") {
      setBrushSize(state.brush.size - 6);
    } else if (key === "]") {
      setBrushSize(state.brush.size + 6);
    } else if (key === "," || key === "<") {
      stepFrame(-1);
    } else if (key === "." || key === ">") {
      stepFrame(1);
    } else if (key === "+" || key === "=") {
      stepZoom(1);
    } else if (key === "-" || key === "_") {
      stepZoom(-1);
    } else if (key === "0") {
      resetView();
    } else if (key === "c" || key === "C") {
      toggleOriginal();
    }
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

  function frameRequest(index, original) {
    return baseUrl() +
      "/frame?dir=" + encodeURIComponent(state.resumeDir) +
      "&index=" + index + (original ? "&original=1" : "");
  }

  function setFrameImage(blob, original, index) {
    var url = URL.createObjectURL(blob);
    state.viewing = original ? "original" : "frame";
    if (original) {
      if (state.sourceUrl) {
        URL.revokeObjectURL(state.sourceUrl);
      }
      state.sourceUrl = url;
      state.sourceIndex = index;
    } else {
      if (state.frameUrl) {
        URL.revokeObjectURL(state.frameUrl);
      }
      state.frameUrl = url;
      state.frameIndex = index;
    }
    ui.previewImg.src = url;
  }

  /* Pull one frame from the engine: the cut-out, or the untouched source when
   * the compare toggle is on. A sequence number discards the answer to a
   * request the user has already moved past, so dragging the playhead can never
   * paint the wrong frame onto the stage. */
  function fetchFrame(index, original) {
    if (!state.resumeDir) {
      return;
    }
    var seq = ++state.frameSeq;
    fetch(frameRequest(index, original))
      .then(function (resp) {
        return resp.ok ? resp.blob() : null;
      })
      .then(function (blob) {
        if (!blob || seq !== state.frameSeq) {
          return;
        }
        setFrameImage(blob, original, index);
      })
      .catch(function () { /* a failed fetch keeps the previous frame */ });
  }

  function showOriginal(on) {
    state.brush.compare = !!on;
    state.viewing = state.brush.compare ? "original" : "frame";
    ui.compareBtn.classList.toggle("on", state.brush.compare);
    show(ui.brushRing, false);
    if (!state.resumeDir) {
      return;
    }
    if (state.brush.compare) {
      /* Already fetched for this frame? Then this is instant. */
      if (state.sourceIndex === state.brushFrame && state.sourceUrl) {
        ui.previewImg.src = state.sourceUrl;
        return;
      }
      fetchFrame(state.brushFrame, true);
      return;
    }
    if (state.frameIndex === state.brushFrame && state.frameUrl) {
      ui.previewImg.src = state.frameUrl;
      return;
    }
    fetchFrame(state.brushFrame, false);
  }

  function toggleOriginal() {
    showOriginal(!state.brush.compare);
  }

  /* The frame's cut-out, unless the user is holding the compare view. */
  function loadFrame(index) {
    state.brushFrame = index;
    state.viewing = state.brush.compare ? "original" : "frame";
    ui.brushFrameLabel.textContent = frameLabel(index);
    if (!state.resumeDir) {
      return;
    }
    if (state.brush.compare) {
      if (state.sourceIndex === index && state.sourceUrl) {
        ui.previewImg.src = state.sourceUrl;
        return;
      }
      fetchFrame(index, true);
      return;
    }
    if (state.frameIndex === index && state.frameUrl) {
      ui.previewImg.src = state.frameUrl;
      return;
    }
    fetchFrame(index, false);
  }

  function scrubFrame(immediate) {
    var index = parseInt(ui.brushFrame.value, 10) || 0;
    ui.brushFrameLabel.textContent = frameLabel(index);
    /* Strokes belong to the frame they were painted on: moving the playhead
     * moves on, so the overlay never lands on the wrong frame. */
    if (index !== state.brushFrame && state.brush.strokes.length) {
      clearBrush();
    }
    if (state.frameTimer) {
      clearTimeout(state.frameTimer);
      state.frameTimer = null;
    }
    if (immediate) {
      loadFrame(index);
      return;
    }
    /* Debounce: dragging the handle would otherwise decode a frame per pixel. */
    state.frameTimer = setTimeout(function () {
      state.frameTimer = null;
      loadFrame(index);
    }, 140);
  }

  function stepFrame(delta) {
    var max = state.frameCount > 0 ? state.frameCount - 1 : 0;
    var next = Math.max(0, Math.min(state.brushFrame + delta, max));
    if (next === state.brushFrame) {
      return;
    }
    ui.brushFrame.value = String(next);
    scrubFrame(true);
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
      matting: !!state.matting && mattingAvailable(),
      /* The cut-out stands in for the clip it came from, so it carries that
       * clip's audio: the engine read the whole source, and a replacement that
       * silently drops the sound is the wrong answer in either host. Premiere
       * is the one that refuses it outright, since changeMediaPath will not
       * swap in a file with no audio where the media it replaces had some. */
      keep_audio: true
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
        return postJob(jobBody(state.source));
      })
      .then(function (job) {
        state.jobId = job.id;
        pollJob();
      })
      .catch(function (err) {
        var message = err && err.message ? err.message : String(err);
        failRun(message, err && err.kind ? err.kind : "internal", false);
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
            "Processed, but applying it to Premiere Pro failed: " +
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
        state.previewToken += 1;
        state.previewSrc = null;
        state.viewing = "preview";
        state.resumeDir = result.work_dir || null;
        state.frameCount = result.frame_count || 0;
        state.keyIdx = result.key_idx || 0;
        /* Everything fetched for the previous result is stale now (a correction
         * rewrites the masks), so drop it and show the new key frame. */
        state.frameIndex = -1;
        state.sourceIndex = -1;
        state.frameSeq += 1;
        state.brush.compare = false;
        ui.compareBtn.classList.remove("on");
        if (state.frameUrl) {
          URL.revokeObjectURL(state.frameUrl);
          state.frameUrl = null;
        }
        if (state.sourceUrl) {
          URL.revokeObjectURL(state.sourceUrl);
          state.sourceUrl = null;
        }
        clearBrush();
        resetFrameView();
        resetView();
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
            ? "The clip's media was replaced with the cut-out."
            : "Added as a clip on the track above.";
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
          "Processed, but applying it to Premiere Pro failed: " +
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
    /* pythonw, not python: the app is a window, so it needs no console — and a
     * console child launched from the host is what took the host down with it
     * on close. */
    return launchFullApp(file).catch(function () {});
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
      brushLayer: id("brushLayer"),
      brushRing: id("brushRing"),
      brushCanvas: id("brushCanvas"),
      brushTools: id("brushTools"),
      brushToolSeg: id("brushToolSeg"),
      brushFrame: id("brushFrame"),
      brushFrameLabel: id("brushFrameLabel"),
      brushPrev: id("brushPrev"),
      brushNext: id("brushNext"),
      brushSize: id("brushSize"),
      brushClear: id("brushClear"),
      brushUndo: id("brushUndo"),
      brushApply: id("brushApply"),
      zoomIn: id("zoomIn"),
      zoomOut: id("zoomOut"),
      zoomFit: id("zoomFit"),
      zoomLabel: id("zoomLabel"),
      compareBtn: id("compareBtn"),
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
      setTool(button.getAttribute("data-value"));
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
      show(ui.brushRing, false);
      if (state.brush.painting) {
        brushUp();
      }
    });
    /* The wheel scrolls, Ctrl+wheel zooms; middle-drag and space-drag pan. All
     * three are captured on the canvas, the only element that should see them. */
    ui.brushCanvas.addEventListener("wheel", wheelScroll, { passive: false });
    ui.brushCanvas.addEventListener("contextmenu", function (event) {
      event.preventDefault();
    });
    ui.brushCanvas.addEventListener("dblclick", function (event) {
      event.preventDefault();
      resetView();
    });
    ui.brushFrame.addEventListener("input", function () {
      scrubFrame(false);
    });
    ui.brushPrev.addEventListener("click", function () {
      stepFrame(-1);
    });
    ui.brushNext.addEventListener("click", function () {
      stepFrame(1);
    });
    ui.brushSize.addEventListener("input", function () {
      setBrushSize(parseInt(ui.brushSize.value, 10));
    });
    ui.zoomIn.addEventListener("click", function () {
      stepZoom(1);
    });
    ui.zoomOut.addEventListener("click", function () {
      stepZoom(-1);
    });
    ui.zoomFit.addEventListener("click", resetView);
    ui.compareBtn.addEventListener("click", toggleOriginal);
    ui.brushUndo.addEventListener("click", undoStroke);
    ui.brushClear.addEventListener("click", clearBrush);
    ui.brushApply.addEventListener("click", applyCorrection);
    document.addEventListener("keydown", onShortcut);
    document.addEventListener("keyup", function (event) {
      if (event.key === " " || event.key === "Spacebar") {
        state.brush.space = false;
        ui.brushStage.classList.remove("pan");
      }
    });

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
