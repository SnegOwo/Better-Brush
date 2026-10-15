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

  var ui = {};
  var state = {
    config: DEFAULT_CONFIG,
    engine: "unknown",
    health: null,
    selection: null,
    jobId: null,
    running: false,
    jobTimer: null,
    selectionTimer: null,
    healthTimer: null
  };

  /* ------------------------------------------------------------------ */
  /* DOM helpers                                                         */
  /* ------------------------------------------------------------------ */
  function id(name) {
    return document.getElementById(name);
  }

  function setMessage(text, kind) {
    ui.message.textContent = text || "";
    ui.message.className = "message" + (kind ? " " + kind : "");
  }

  function setStatus(kind, text) {
    ui.statusDot.className = "dot" + (kind ? " " + kind : "");
    ui.statusText.textContent = text;
  }

  function setProgress(percent) {
    var value = Math.max(0, Math.min(100, Math.round(percent)));
    ui.progressBar.style.width = value + "%";
  }

  function stageLabel(stage) {
    var map = {
      queued: "Queued",
      extract: "Reading frames",
      analyze: "Finding the character",
      propagate: "Tracking",
      export: "Writing alpha",
      done: "Finishing"
    };
    return map[stage] || "Working";
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

  function getHealth() {
    var options = { cache: "no-store" };
    var controller = null;
    var timer = null;
    if (typeof AbortController !== "undefined") {
      controller = new AbortController();
      options.signal = controller.signal;
      timer = setTimeout(function () {
        controller.abort();
      }, 1500);
    }
    return fetch(baseUrl() + "/health", options)
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

  function applyHealth(health) {
    if (!health) {
      state.health = null;
      state.engine = "offline";
      setStatus("offline", "engine offline");
      ui.engineInfo.textContent = "engine: offline";
      updateButtons();
      return;
    }
    state.health = health;
    state.engine = "online";
    setStatus(health.busy ? "busy" : "online", health.busy ? "working…" : "engine ready");
    ui.engineInfo.textContent =
      "engine v" + (health.engine_version || "?") + " · " + (health.pro ? "Pro" : "Lite");
    ui.proBanner.classList.toggle("hidden", !!health.pro);
    updateButtons();
  }

  function refreshHealth() {
    return getHealth().then(function (health) {
      applyHealth(health);
      return health;
    });
  }

  function waitForEngine(deadline) {
    return getHealth().then(function (health) {
      if (health) {
        applyHealth(health);
        return true;
      }
      if (Date.now() > deadline) {
        throw new Error("The engine did not start in time. Open Better Brush once to finish setup.");
      }
      return new Promise(function (resolve) {
        setTimeout(resolve, 1000);
      }).then(function () {
        return waitForEngine(deadline);
      });
    });
  }

  function ensureEngine() {
    return getHealth().then(function (health) {
      if (health) {
        applyHealth(health);
        return true;
      }
      setStatus("busy", "starting engine…");
      setMessage("Starting the Better Brush engine…");
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
  function renderSelection(sel) {
    ui.infoComp.textContent = sel && sel.comp ? sel.comp : "—";
    ui.infoLayer.textContent = sel && sel.layer ? sel.layer : "—";
    ui.infoSize.textContent = sel && sel.width ? sel.width + "×" + sel.height : "—";
    ui.infoFrames.textContent =
      sel && sel.ok ? (sel.isStill ? "1 (still)" : String(sel.frames)) : "—";

    ui.selectionHint.className = "hint";
    if (!sel) {
      ui.selectionHint.textContent = "";
      return;
    }
    if (sel.ok) {
      ui.selectionHint.textContent = sel.isStill
        ? "Still image — background removal applies to this frame."
        : "Ready.";
    } else {
      ui.selectionHint.textContent = sel.reason || "Select a footage layer.";
      ui.selectionHint.className = "hint warn";
    }
  }

  function refreshSelection() {
    if (state.running) {
      return Promise.resolve();
    }
    return callHost("bbSelectedLayer()")
      .then(function (sel) {
        state.selection = sel;
        renderSelection(sel);
        updateButtons();
      })
      .catch(function () {
        state.selection = null;
        renderSelection(null);
        updateButtons();
      });
  }

  /* ------------------------------------------------------------------ */
  /* Run                                                                 */
  /* ------------------------------------------------------------------ */
  function updateButtons() {
    var pro = state.health ? !!state.health.pro : true;
    var ready = !!(state.selection && state.selection.ok);
    var canRun = !state.running && pro && ready;
    ui.runBtn.disabled = !canRun;
    ui.cancelBtn.classList.toggle("hidden", !state.running);
    ui.refreshBtn.disabled = state.running;
  }

  function resetRun() {
    state.running = false;
    state.jobId = null;
    ui.progressWrap.classList.add("hidden");
    updateButtons();
    refreshSelection();
  }

  function postJob(sel) {
    var body = {
      input_path: sel.file,
      frame_in: 0,
      frame_out: -1,
      fps: sel.fps,
      width: sel.width,
      height: sel.height,
      preset: "anime",
      quality: ui.qualitySelect.value
    };
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
        throw new Error(message);
      });
    });
  }

  function startJob() {
    if (state.running) {
      return;
    }
    var sel = state.selection;
    if (!sel || !sel.ok) {
      setMessage(sel && sel.reason ? sel.reason : "Select a footage layer first.", "err");
      return;
    }

    state.running = true;
    ui.progressWrap.classList.remove("hidden");
    setProgress(0);
    setMessage("Starting…");
    updateButtons();

    ensureEngine()
      .then(function () {
        if (state.health && state.health.pro === false) {
          throw new Error("__PRO__");
        }
        return postJob(sel);
      })
      .then(function (job) {
        state.jobId = job.id;
        setMessage("Working…");
        pollJob();
      })
      .catch(function (err) {
        var message = err && err.message ? err.message : String(err);
        resetRun();
        if (message === "__PRO__") {
          setMessage("Better Brush Pro is required for the After Effects panel.", "err");
        } else {
          setMessage(message, "err");
        }
      });
  }

  function pollJob() {
    if (!state.jobId) {
      return;
    }
    fetch(baseUrl() + "/job/" + state.jobId, { cache: "no-store" })
      .then(function (resp) {
        return resp.json();
      })
      .then(function (job) {
        onJob(job);
      })
      .catch(function () {
        schedulePoll();
      });
  }

  function schedulePoll() {
    state.jobTimer = setTimeout(pollJob, POLL_MS);
  }

  function onJob(job) {
    if (!job) {
      schedulePoll();
      return;
    }
    var progress = job.progress || {};
    if (progress.total > 0) {
      setProgress((progress.current / progress.total) * 100);
    } else {
      setProgress(8);
    }
    if (progress.stage) {
      setMessage(stageLabel(progress.stage) + (progress.message ? " — " + progress.message : "…"));
    }

    if (job.state === "done") {
      finishJob(job.result);
    } else if (job.state === "error") {
      failJob(job.error);
    } else if (job.state === "cancelled") {
      setMessage("Cancelled.");
      resetRun();
    } else {
      schedulePoll();
    }
  }

  function finishJob(result) {
    if (!result) {
      setMessage("Finished, but no result was returned.", "err");
      resetRun();
      return;
    }
    var mode = ui.modeSelect.value;
    var sel = state.selection || {};
    var script =
      "bbApplyResult(" +
      csString(result.output_path) + "," +
      csString(mode) + "," +
      (result.fps || 0) + "," +
      (sel.compId != null ? sel.compId : -1) + "," +
      (sel.layerId != null ? sel.layerId : -1) +
      ")";

    callHost(script)
      .then(function (res) {
        if (res && res.ok) {
          var verb = mode === "replace" ? "Layer replaced" : "Added as a new layer";
          setMessage(
            "Done — " + result.frame_count + " frame" +
              (result.frame_count === 1 ? "" : "s") + ". " + verb + ".",
            "ok"
          );
        } else {
          setMessage(
            "Processed, but applying failed: " + (res && res.reason ? res.reason : "unknown"),
            "err"
          );
        }
        resetRun();
      })
      .catch(function (err) {
        setMessage("Processed, but applying failed: " + err.message, "err");
        resetRun();
      });
  }

  function failJob(error) {
    var kind = error && error.kind ? error.kind : "internal";
    var message = error && error.message ? error.message : "Something went wrong.";
    resetRun();

    if (kind === "no_character" || kind === "multiple_subjects") {
      setMessage(message + " Opening the full app…", "err");
      openFullApp();
    } else if (kind === "busy") {
      setMessage("The engine is busy with another job. Try again shortly.", "err");
    } else if (kind === "unsupported") {
      setMessage(message, "err");
    } else {
      setMessage(message, "err");
    }
  }

  function cancelJob() {
    if (!state.jobId) {
      return;
    }
    fetch(baseUrl() + "/job/" + state.jobId + "/cancel", { method: "POST" })
      .catch(function () {})
      .then(function () {
        setMessage("Cancelling…");
      });
  }

  function openFullApp() {
    return callHost(
      "bbOpenFullApp(" +
        csString(state.config.pythonPath) + "," +
        csString(state.config.projectRoot) +
      ")"
    ).catch(function () {});
  }

  /* ------------------------------------------------------------------ */
  /* Settings                                                            */
  /* ------------------------------------------------------------------ */
  function saveSettings() {
    try {
      localStorage.setItem(
        "bb.settings",
        JSON.stringify({
          quality: ui.qualitySelect.value,
          mode: ui.modeSelect.value
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
      var settings = JSON.parse(raw);
      if (settings.quality) {
        ui.qualitySelect.value = settings.quality;
      }
      if (settings.mode) {
        ui.modeSelect.value = settings.mode;
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
      proBanner: id("proBanner"),
      infoComp: id("infoComp"),
      infoLayer: id("infoLayer"),
      infoSize: id("infoSize"),
      infoFrames: id("infoFrames"),
      selectionHint: id("selectionHint"),
      qualitySelect: id("qualitySelect"),
      modeSelect: id("modeSelect"),
      runBtn: id("runBtn"),
      cancelBtn: id("cancelBtn"),
      refreshBtn: id("refreshBtn"),
      progressWrap: id("progressWrap"),
      progressBar: id("progressBar"),
      message: id("message"),
      engineInfo: id("engineInfo"),
      openAppBtn: id("openAppBtn")
    };
  }

  function init() {
    bindUi();
    restoreSettings();

    ui.runBtn.addEventListener("click", startJob);
    ui.cancelBtn.addEventListener("click", cancelJob);
    ui.refreshBtn.addEventListener("click", refreshSelection);
    ui.openAppBtn.addEventListener("click", openFullApp);
    ui.qualitySelect.addEventListener("change", saveSettings);
    ui.modeSelect.addEventListener("change", saveSettings);

    setMessage("Connecting to the Better Brush engine…");
    updateButtons();

    loadConfig()
      .then(function () {
        return refreshHealth();
      })
      .then(function () {
        return refreshSelection();
      })
      .then(function () {
        state.selectionTimer = setInterval(refreshSelection, SELECTION_POLL_MS);
        state.healthTimer = setInterval(refreshHealth, HEALTH_POLL_MS);
      });
  }

  window.addEventListener("unload", function () {
    if (state.selectionTimer) { clearInterval(state.selectionTimer); }
    if (state.healthTimer) { clearInterval(state.healthTimer); }
    if (state.jobTimer) { clearTimeout(state.jobTimer); }
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
