/*
 * BetterBrush.jsx — Better Brush panel for EffectCraft (MVP).
 *
 * A dockable ScriptUI panel with one job: describe the current selection to
 * the Python adapter, wait for it to render an alpha PNG sequence, and hand
 * the resulting EffectCraft commands back to the engine.
 *
 * It never runs inference itself. The heavy work happens in another process
 * on purpose — a script must not block EffectCraft's UI thread, and EffectCraft
 * has no in-process way to hold a torch session.
 *
 * Verified against EffectCraft 0.11.1; see docs/VALIDATION.md in this folder
 * for every API this file relies on and how it was checked.
 *
 * The panel needs Preferences > Scripting & Expressions > "Allow Scripts to
 * Write Files and Access Network" turned on: without it, EffectCraft refuses
 * every File.open() a script attempts (checked at load, and the panel says so).
 */

(function (thisObj) {
  var VERSION = "0.6.0";
  var BRIDGE_NAME = "betterbrush-bridge";
  /* Written by scripts/install.ps1 at install time with the absolute bridge
   * folder (forward slashes). Needed because a *panel* script sees
   * $.fileName as a bare file name — EffectCraft hands us just
   * "BetterBrush.jsx", no directory (verified) — so the file cannot find its
   * own folder by walking up. Left as the placeholder, the panel falls back to
   * the $.fileName walk, which works when the script is run as a plain script
   * (File ▸ Scripts), where $.fileName is a full path. */
  var BRIDGE_OVERRIDE = "@@BB_BRIDGE@@";
  var JOB_PREFIX = "job-";
  var STATUS_PREFIX = "status-";
  var POLL_MS = 700;
  /* How often the panel refreshes itself while it is open. One tick does
   * either half of the work: follow the running job, or notice that the
   * selection / masks / time indicator changed. */
  var TICK_MS = POLL_MS;
  var PREVIEW_W = 300;
  var PREVIEW_H = 169;
  var MATTE_NAME = "BetterBrush Matte";

  /* Stage ids, in pipeline order, with the label the status line shows.
   * "load" reports 0/0, which puts the bar in its indeterminate state — an
   * honest representation of a model load. */
  var STAGE_LABEL = {
    load: "Loading the model",
    extract: "Reading the source",
    analyze: "Finding the subject",
    propagate: "Tracking across frames",
    matte: "Refining the matte",
    post: "Post-processing",
    export: "Writing PNG frames",
    done: "Done"
  };

  /* Quality tiers in the order the dropdown shows them; the engine maps these
   * to checkpoints (VALID_QUALITIES in src/engine/schemas.py), and the note
   * under the dropdown says which one the current choice will load. */
  var QUALITIES = [
    { id: "tiny", label: "Tiny", model: "SAM 2.1 Tiny — fastest, roughest" },
    { id: "fast", label: "Fast", model: "SAM 2.1 Small" },
    { id: "balanced", label: "Balanced", model: "SAM 2.1 Base+ (default)" },
    { id: "best", label: "Best", model: "Best installed checkpoint" }
  ];

  /* Panel chrome. ScriptUI inherits the host's theme, so colour is one of the
   * few levers there are for making a panel read as designed rather than as
   * a form: an accent for the brand, a muted grey for secondary notes, and
   * status tones that make "failed" and "applied" unmistakable. */
  var ACCENT = [0.15, 0.58, 0.45];
  var MUTED = [0.52, 0.52, 0.52];
  var TONE = {
    neutral: [0.2, 0.2, 0.2],
    busy: [0.15, 0.35, 0.7],
    ok: [0.1, 0.5, 0.25],
    error: [0.75, 0.2, 0.15]
  };
  var CONTEXT_FILE = "brush-context.json";
  /* Stroke ink on the preview: green keeps pixels, red removes them. Soft
   * alpha so the matte underneath stays readable while painting. */
  var INK_ADD = [0.16, 0.84, 0.36, 0.38];
  var INK_REMOVE = [0.94, 0.26, 0.22, 0.38];
  /* Bridge files for "Open in Better Brush": the app's exe (written once,
   * then remembered), the request, and the adapter's answer to it. */
  var APP_PATH_FILE = "app-path.json";
  var APP_OPEN_PREFIX = "app-open-";
  var APP_STATUS_PREFIX = "app-open-status-";
  /* The adapter's liveness contract: it stamps adapter.alive every watch
   * cycle, and the installer leaves start-adapter.cmd in the bridge with the
   * exact command to bring one up. */
  var ADAPTER_ALIVE_FILE = "adapter.alive";
  /* .vbs first: wscript starts it with no window, where a .cmd flashes a
   * console. Both are written by the installer. */
  var ADAPTER_START_FILES = ["start-adapter.vbs", "start-adapter.cmd"];
  var ADAPTER_START_AFTER_MS = 8000;

  var state = {
    bridge: null,     // absolute folder path, forward slashes
    writable: false,
    job: null,        // job id currently in flight, or null
    task: -1,         // scheduleTask id of the repeating tick, -1 = none armed
    ui: null,
    btn: null,
    cancelBtn: null,
    bar: null,
    status: null,
    elapsedEl: null,    // seconds-since-submit, its own control (not logged)
    writeError: "",
    blocked: "",        // why the button is dead; echoed verbatim in the status
    lastNote: "",       // last de-duplicated poll observation
    armError: false,    // scheduleTask failing was reported once already
    lastTickError: "",  // de-duplicates tick failures in panel.log
    lastSay: "",        // de-duplicates consecutive status lines in panel.log
    lastTick: 0,        // ms timestamp of the last tick — the heartbeat
    armedAt: 0,         // ms timestamp the current task was armed
    firstTick: false,   // the "auto-update is alive" line was logged once
    armLogged: false,   // "tick armed" was logged once already
    repairLogged: false,// a heartbeat repair was logged once already
    taskFired: false,   // the engine really ran the scheduled task at least once
    timerNoted: false,  // the "this build has no timers" diagnosis was logged once
    lastPump: 0,        // ms timestamp of the last hover-driven tick
    probeAt: 0,         // ms timestamp of the first arm attempt (timer diagnosis)
    jobStart: 0,        // ms timestamp the current job was submitted
    jobDoc: null,   // full job document for the in-flight id
    log: [],
    /* Dock controls, filled in by buildUI. */
    qualityDd: null,
    modelNote: null,
    mattingChk: null,
    postChk: null,
    previewSlot: null,   // group the result image is swapped into
    previewImg: null,    // the current image control, or null
    previewPlaceholder: null,
    previewCap: null,
    brushHint: null,
    brushList: null,
    brushSizeSlider: null,
    brushSizeLabel: null,
    removeForceSlider: null,
    removeForceLabel: null,
    frameNote: null,
    markAddBtn: null,
    markRemoveBtn: null,
    undoStrokeBtn: null,
    clearStrokesBtn: null,
    refreshBtn: null,
    applyBrushBtn: null,
    openAppBtn: null,
    appPathEdit: null,
    appPathBtn: null,
    /* The dock's own settings, read by collect(). */
    opts: { quality: "balanced", matting: true, post: true },
    /* Strokes painted on the frame: {tool, radius, points: [[x, y], ...]} in
     * media pixels. They live in the panel and travel with the job - see
     * "Brush - paint on the frame". */
    strokes: [],
    /* The stroke being painted right now, or null. */
    painting: null,
    /* What the next stroke means: + (add) or - (remove). */
    armedTool: "add",
    /* Brush radius, in media pixels. */
    brushSize: 36,
    /* How hard a red stroke bites, 1-5: 1 is exactly what was painted, 5
     * takes a margin around it (see the adapter's remove_region). */
    removeForce: 3,
    /* Frame the preview is showing, so an idle tick re-points the image only
     * when the time indicator actually moved. */
    shownFrame: null,
    /* Better Brush desktop app: its exe, recorded once in the bridge by
     * app-path.json, and the request the adapter is opening right now. */
    appPath: "",
    appRequest: "",
    appRequestAt: 0,
    adapterCheckAt: 0,   // when the "is the adapter running?" look is due
    adapterStarted: false,// one start attempt per panel session, never more
    /* What the last successful run left behind — the folder a correction
     * resumes from and the matte layer it replaces. Persisted in the bridge. */
    ctx: null
  };

  /* ---------------------------------------------------------------- */
  /* Small utilities                                                   */
  /* ---------------------------------------------------------------- */
  function hasJSON() {
    return typeof JSON === "object" && JSON.stringify;
  }

  function parse(text) {
    /* ExtendScript may emit a UTF-8 BOM when the panel writes a file; strip
     * it so a re-read of our own output still parses. */
    if (text && text.charCodeAt(0) === 0xFEFF) {
      text = text.substring(1);
    }
    if (hasJSON()) {
      return JSON.parse(text);
    }
    /* ExtendScript's JSON has shipped in every engine we have seen, so this
     * is a belt-and-braces path rather than the main one. */
    return eval("(" + text + ")");
  }

  function say(text, tone) {
    if (state.status) {
      state.status.text = text;
      setColor(state.status, TONE[tone || "neutral"]);
    }
    state.log.push(text);
    if (state.log.length > 200) {
      state.log.shift();
    }
    /* Consecutive duplicates are dropped: the tick re-sends the same stage
     * line every 700 ms while the adapter works, and panel.log is rewritten
     * in full on every line — v0.2 logged "Writing PNG frames" seven times in
     * six seconds. */
    if (text !== state.lastSay) {
      state.lastSay = text;
      writeLog(text);
    }
  }

  /* The one number that answers "is this still doing anything?": seconds
   * since this job was submitted, on its own control (say()'s text goes to
   * panel.log, and a counter would rewrite the log every tick). */
  function updateElapsed() {
    if (!state.elapsedEl) {
      return;
    }
    var txt = "";
    if (state.jobStart) {
      var s = Math.max(0, Math.round((nowMs() - state.jobStart) / 1000));
      txt = "elapsed " + (s >= 60
          ? Math.floor(s / 60) + "m " + (s % 60) + "s"
          : s + "s");
    }
    if (state.elapsedEl.text !== txt) {
      state.elapsedEl.text = txt;
    }
  }

  /* Colour and type: ScriptUI takes its theme from the host, so these are the
   * levers a panel has for looking designed. Every call is wrapped — a font
   * the host does not ship must never take the panel down. */
  function setColor(ctrl, rgb) {
    try {
      if (ctrl && ctrl.graphics) {
        ctrl.graphics.foregroundColor = rgb;
      }
    } catch (e) {
      /* plain black is a fine fallback */
    }
  }

  function styleText(ctrl, size, color, bold) {
    if (!ctrl) {
      return;
    }
    try {
      ctrl.graphics.font = ScriptUI.newFont("Segoe UI", bold ? "Bold" : "Regular", size);
    } catch (e) {
      try {
        ctrl.graphics.font = ScriptUI.newFont(ctrl.graphics.font.name, "Regular", size);
      } catch (e2) {
        /* keep the host default */
      }
    }
    if (color) {
      setColor(ctrl, color);
    }
  }

  function setBar(fraction) {
    if (!state.bar) {
      return;
    }
    /* -1 means indeterminate: EffectCraft's progressbar has no such mode, so
     * sit at a low value and let the status line carry the stage name. */
    state.bar.value = fraction < 0 ? 8 : Math.round(fraction * 100);
  }

  function pad(n) {
    return n < 10 ? "0" + n : String(n);
  }

  function jobId() {
    var d = new Date();
    var rnd = Math.floor(Math.random() * 0xffffff).toString(16);
    return (
      "bb-" +
      d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + "-" +
      pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds()) +
      "-" + rnd
    );
  }

  /* ---------------------------------------------------------------- */
  /* Bridge folder                                                     */
  /* ---------------------------------------------------------------- */
  /* The panel derives the folder from its own file: EffectCraft exposes no
   * Folder.temp and no Folder.current (both are null), and the settings folder
   * is %APPDATA%\EffectCraft normally but sits beside the executable in
   * portable mode. The script always lives at
   * <settings>/Scripts/ScriptUI Panels/BetterBrush.jsx, so three levels up is
   * the settings folder either way. */
  function resolveBridge() {
    /* 1. install.ps1 stamped the absolute bridge folder into this file. Only
     *    trust it while it still describes a real install — the panel file
     *    must sit at <bridge>/../Scripts/ScriptUI Panels/BetterBrush.jsx — so
     *    a stale path after a move falls through instead of writing into a
     *    folder EffectCraft no longer owns. */
    if (BRIDGE_OVERRIDE && BRIDGE_OVERRIDE.indexOf("@@") < 0) {
      try {
        var stamped = new File(BRIDGE_OVERRIDE +
            "/../Scripts/ScriptUI Panels/BetterBrush.jsx");
        if (stamped.exists) {
          return BRIDGE_OVERRIDE;
        }
      } catch (e0) {
        /* fall through to the path walk */
      }
    }

    /* 2. Derive it from the script's own file. EffectCraft exposes no
     * Folder.temp / Folder.current / Folder.appData (all null), and a plain
     * script gets a full path in $.fileName — only the panel host does not. */
    var me = null;
    try {
      if (typeof $ === "object" && $.fileName) {
        me = new File($.fileName);
      }
    } catch (e) {
      me = null;
    }
    if (!me || !me.fsName) {
      return null;
    }
    try {
      var panels = me.parent;                 // .../ScriptUI Panels
      var scripts = panels ? panels.parent : null;   // .../Scripts
      var settings = scripts ? scripts.parent : null;  // .../EffectCraft
      if (!settings || !settings.fsName || !panels) {
        return null;
      }
      /* Refuse to derive a bridge folder from an arbitrary location: run in
       * place (from the repository, say) this would create a bridge inside
       * the checkout. Only a file that really sits in the panel host counts. */
      var host = String(panels.name || "");
      if (host.toLowerCase().indexOf("scriptui panels") < 0) {
        return null;
      }
      return settings.fsName + "/" + BRIDGE_NAME;
    } catch (e2) {
      return null;
    }
  }

  function writeFile(folder, name, text) {
    var f = new File(folder + "/" + name);
    f.encoding = "UTF-8";
    if (!f.open("w")) {
      throw new Error("could not open " + name + " for writing");
    }
    f.write(text);
    f.close();
    return f;
  }

  function readFile(folder, name) {
    var f = new File(folder + "/" + name);
    if (!f.exists) {
      return null;
    }
    f.encoding = "UTF-8";
    if (!f.open("r")) {
      return null;
    }
    var text = f.read();
    f.close();
    return text;
  }

  function writeLog(line) {
    if (!state.writable || !state.bridge) {
      return;
    }
    try {
      var stamp = new Date().toISOString ? new Date().toISOString() : "";
      var path = state.bridge + "/panel.log";
      /* Read-then-rewrite, not seek(): this build's File does not honour
       * seek(0, 2) — the old version left overlapping fragments in
       * panel.log, which is exactly what we saw in the field. */
      var old = "";
      var rf = new File(path);
      rf.encoding = "UTF-8";
      if (rf.exists && rf.open("r")) {
        old = rf.read() || "";
        rf.close();
      }
      var f = new File(path);
      f.encoding = "UTF-8";
      if (f.open("w")) {
        f.write(old + stamp + " " + line + "\n");
        f.close();
      }
    } catch (e) {
      /* Logging must never break the panel. */
    }
  }

  /* EffectCraft gates file writes behind a preference; probe it once so the
   * user gets an instruction instead of a silent no-op button. */
  function checkWritable() {
    if (!state.bridge) {
      return false;
    }
    try {
      var folder = new Folder(state.bridge);
      if (!folder.exists) {
        folder.create();
      }
      writeFile(state.bridge, ".write-probe", "ok");
      return true;
    } catch (e) {
      state.writeError = String(e.message || e);
      return false;
    }
  }

  /* The one sentence explaining why the panel is dead. Written once at
   * startup and echoed verbatim afterwards: an earlier version let a later
   * status update replace it with a generic line that pointed at text no
   * longer on screen. */
  function setBlocked(reason) {
    state.blocked = reason;
    say(reason, "error");
  }

  /* ---------------------------------------------------------------- */
  /* Selection -> job document                                         */
  /* ---------------------------------------------------------------- */
  function activeComp() {
    var item = app.project ? app.project.activeItem : null;
    if (!item) {
      return null;
    }
    /* A comp responds to numLayers; footage and solids do not. */
    if (typeof item.numLayers !== "number") {
      return null;
    }
    return item;
  }

  function require(cond, message) {
    if (!cond) {
      throw new Error(message);
    }
  }

  function collect() {
    var comp = activeComp();
    require(comp, "Open a composition first.");

    var selected = comp.selectedLayers || [];
    require(selected.length > 0, "Select the footage layer to cut out.");
    require(selected.length === 1, "Select exactly one layer (several are selected).");

    var layer = selected[0];
    var src = layer.source;
    require(src, "That layer has no source (a null, text or shape layer).");

    var file = null;
    try {
      file = src.file;
    } catch (e) {
      file = null;
    }
    require(file && file.fsName,
      "That layer's source is not a file on disk (a solid or a precomp).");

    var stretch = typeof layer.stretch === "number" ? layer.stretch : 100;
    require(Math.abs(stretch - 100) < 0.001,
      "Time-stretched layers are not supported yet — remove the stretch first.");

    var threeD = false;
    try {
      threeD = layer.threeDLayer === true;
    } catch (e2) {
      threeD = false;
    }
    require(!threeD, "3D layers are not supported yet.");

    var fps = src.frameRate || comp.frameRate || 24;
    var start = typeof layer.startTime === "number" ? layer.startTime : 0;
    var inPoint = typeof layer.inPoint === "number" ? layer.inPoint : start;
    var outPoint = typeof layer.outPoint === "number" ? layer.outPoint : start + 1;

    /* Source frames, not comp frames: (comp time - layer start) * media fps.
     * Getting this wrong shifts the matte by the layer's offset and looks like
     * a tracking failure. */
    var frameIn = Math.floor((inPoint - start) * fps + 0.0001);
    var frameOut = Math.ceil((outPoint - start) * fps - 0.0001) - 1;
    if (frameIn < 0) {
      frameIn = 0;
    }
    if (frameOut < frameIn) {
      frameOut = -1; /* -1 = "to the end"; the adapter resolves it */
    }

    var transform = {};
    try {
      transform = {
        anchor: vec(layer.transform.anchorPoint.value),
        position: vec(layer.transform.position.value),
        scale: vec(layer.transform.scale.value),
        rotation: layer.transform.rotation.value
      };
    } catch (e3) {
      throw new Error("could not read the layer's transform: " + e3.message);
    }

    return {
      protocol: 1,
      id: jobId(),
      created: new Date().toString(),
      host: "effectcraft",
      hostVersion: typeof app.version === "string" ? app.version : "",
      /* The real Better Brush pipeline (SAM 2.1 + the repo's presets). The
       * adapter can still force a cheaper run with `--mode simulate`, which
       * overrides whatever is written here — useful on a machine without a
       * GPU or just for a fast plumbing check. */
      mode: "engine",
      source: file.fsName,
      range: { frame_in: frameIn, frame_out: frameOut },
      media: {
        width: src.width || comp.width,
        height: src.height || comp.height,
        frame_rate: fps
      },
      comp: {
        name: comp.name,
        width: comp.width,
        height: comp.height,
        frame_rate: comp.frameRate,
        duration: comp.duration,
        time: comp.time
      },
      source_layer: {
        id: typeof layer.id === "number" ? layer.id : 0,
        name: layer.name,
        index: layer.index,
        in: inPoint,
        out: outPoint,
        start: start,
        stretch: stretch
      },
      /* The import inserts the matte *at* the source layer's index, so the
       * source is pushed down by one and the matte ends up directly above it —
       * exactly where a track matte belongs. */
      target: { matte_index: layer.index },
      transform: transform,
      /* Quality, MatAnyone 2 matting and the post-processing pass all come
       * from the dock — the same three switches the desktop app exposes.
       * "preset" stays anime, as the After Effects panel hard-codes it. */
      options: {
        preset: "anime",
        quality: state.opts.quality,
        matting: !!state.opts.matting,
        post_process: !!state.opts.post
      }
    };
  }

  function vec(value) {
    if (value && typeof value.length === "number") {
      var out = [];
      for (var i = 0; i < value.length; i++) {
        out.push(value[i]);
      }
      return out;
    }
    return [value];
  }

  /* ---------------------------------------------------------------- */
  /* Brush — strokes and the context a correction needs               */
  /* ---------------------------------------------------------------- */
  /* A correction re-tracks an earlier run instead of starting over, so it
   * needs three things the panel remembers: the engine's work folder (seed +
   * masks to resume from), the matte layer's footage item (to swap frames
   * into) and the layer/comp the run belonged to (so strokes drawn somewhere
   * else are never applied to it). All three live in one JSON file in the
   * bridge, written after every successful apply. */

  function readContext() {
    try {
      var text = readFile(state.bridge, CONTEXT_FILE);
      return text ? parse(text) : null;
    } catch (e) {
      return null;
    }
  }

  function saveContext(ctx) {
    state.ctx = ctx;
    if (!state.writable || !state.bridge) {
      return;
    }
    try {
      writeFile(state.bridge, CONTEXT_FILE,
                hasJSON() ? JSON.stringify(ctx, null, 2) : String(ctx));
    } catch (e) {
      writeLog("context save failed: " + String(e.message || e));
    }
  }

  function findMatteLayer(comp, near) {
    if (!comp) {
      return null;
    }
    try {
      if (near && near >= 1 && near <= comp.numLayers) {
        var here = comp.layer(near);
        if (here && here.name === MATTE_NAME) {
          return here;
        }
      }
    } catch (e) {
      /* fall through to the scan */
    }
    for (var n = 1; n <= comp.numLayers; n++) {
      try {
        var layer = comp.layer(n);
        if (layer && layer.name === MATTE_NAME) {
          return layer;
        }
      } catch (e2) {
        /* skip an unreadable entry */
      }
    }
    return null;
  }

  /* Remember what this run left behind, so the brush section knows where a
   * correction should resume and which item it replaces. */
  function captureContext(comp, doc, result, previewCap) {
    var matte = findMatteLayer(comp, doc && doc.target ? doc.target.matte_index : 0);
    if (!matte || !doc || !doc.source_layer) {
      return null;
    }
    var item = null;
    try {
      item = matte.source;
    } catch (e) {
      item = null;
    }
    var matteItem = 0;
    if (item && typeof item.id === "number" && item.id) {
      matteItem = item.id;
    } else if (item && item.name) {
      matteItem = item.name;
    }
    var ctx = {
      comp_name: comp.name,
      layer_id: doc.source_layer.id,
      layer_name: doc.source_layer.name,
      /* The clip itself — "Open in Better Brush" hands this file to the
       * desktop app so it opens on what the panel is working on. */
      source_path: doc.source || "",
      work_dir: (result && result.work_dir) || "",
      /* The rendered sequence itself: the brush paints on the frame the time
       * indicator is on, read straight from these PNGs. */
      png_dir: (result && result.png_dir) || "",
      frame_in: (doc.range && doc.range.frame_in) || 0,
      frame_count: (result && result.frame_count) || 0,
      media_width: (doc.media && doc.media.width) || 0,
      media_height: (doc.media && doc.media.height) || 0,
      /* The job document itself: a correction re-uses it (range, transform,
       * media) instead of asking the user to keep the source layer selected. */
      job: doc,
      matte_layer_index: matte.index,
      matte_item: matteItem,
      /* The result image survives a panel reload the same way the context
       * does: reopening EffectCraft should still show what was rendered. */
      preview: (result && result.preview) || "",
      preview_cap: previewCap || "",
      saved: new Date().toString()
    };
    saveContext(ctx);
    return ctx;
  }

  /* ---------------------------------------------------------------- */
  /* Brush - paint on the frame                                        */
  /* ---------------------------------------------------------------- */
  /* The After Effects and Premiere panels paint on a canvas; ScriptUI has no
   * canvas, but this engine delivers mousedown / mousemove / mouseup to any
   * widget with a listener, with coordinates *relative to that widget* and
   * the modifier keys - which is everything a brush needs. The frame the last
   * run produced is already in the panel, so the result image is the canvas:
   *
   *   drag            paint with the armed tool (+ adds pixels back in,
   *                   - takes them out)
   *   Shift + drag    paint with the other one
   *   Apply fix       re-track from these strokes and swap the new frames in
   *
   * Strokes stay in the panel as polylines in media pixels - the space the
   * engine's correction mask is rasterised in - and travel with the job.
   * Nothing is drawn into the composition, so the matte layer never gains
   * mask shapes and nothing has to be cleaned up afterwards. */

  /* The layer the brush belongs to: the one the last run cut out, found by
   * id (then by name), falling back to the single selected layer. Mask
   * values are not read from it - this keeps the frame in step and says
   * which layer a correction re-tracks. */
  function strokeLayer() {
    var comp = null;
    try {
      comp = activeComp();
    } catch (e) {
      comp = null;
    }
    var ctx = state.ctx;
    if (comp && ctx && (ctx.layer_id || ctx.layer_name)) {
      for (var n = 1; n <= comp.numLayers; n++) {
        try {
          var candidate = comp.layer(n);
          if (!candidate) {
            continue;
          }
          if (ctx.layer_id && typeof candidate.id === "number" &&
              candidate.id === ctx.layer_id) {
            return candidate;
          }
          if (!ctx.layer_id && ctx.layer_name && candidate.name === ctx.layer_name) {
            return candidate;
          }
        } catch (e2) {
          /* skip an unreadable entry */
        }
      }
    }
    try {
      if (comp && comp.selectedLayers && comp.selectedLayers.length === 1) {
        return comp.selectedLayers[0];
      }
    } catch (e3) {
      /* no selection */
    }
    return null;
  }

  /* A correction needs a finished run to resume from. */
  function brushReady() {
    return !!(state.ctx && state.ctx.work_dir && state.ctx.matte_item);
  }

  /* ---------------------------------------------------------------- */
  /* The canvas: the result image                                      */
  /* ---------------------------------------------------------------- */
  /* `remove()` on an image control does nothing in this engine (verified:
   * the child stays in the group), so the control is created once and
   * re-pointed with `.image`; adding a second one stacks a second picture on
   * top of the first - which is exactly the doubled image the panel used to
   * show. */
  function previewSize() {
    var w = (state.ctx && state.ctx.media_width) || 16;
    var h = (state.ctx && state.ctx.media_height) || 9;
    var height = Math.round(PREVIEW_W * h / w);
    if (height < 60) {
      height = 60;
    }
    if (height > 420) {
      height = 420;
    }
    return [PREVIEW_W, height];
  }

  function ensurePreview(file) {
    if (!state.previewSlot) {
      return null;
    }
    if (!state.previewImg) {
      state.previewImg = state.previewSlot.add("image", undefined, file);
      state.previewImg.preferredSize = previewSize();
      state.previewImg.helpTip =
          "Drag to paint a stroke: + keeps pixels, - removes them, Shift swaps.";
      /* The strokes are painted on top of the picture by the engine's own
       * draw list (see drawOverlay), so what is about to be applied is
       * visible while it is painted. */
      state.previewImg.onDraw = drawOverlay;
      try {
        state.previewImg.addEventListener("mousedown", beginStroke);
        state.previewImg.addEventListener("mousemove", continueStroke);
        state.previewImg.addEventListener("mouseup", finishStroke);
        state.previewImg.addEventListener("mouseout", finishStroke);
      } catch (e) {
        writeLog("brush listeners unavailable: " + String(e.message || e));
      }
    }
    return state.previewImg;
  }

  /* onDraw for the image: green where the mallet adds pixels back, red where
   * it takes them out, in soft opacity so the matte stays readable.
   *
   * `drawOSControl()` first is what keeps the picture: without it an onDraw
   * handler *replaces* the control's own paint (engine: ScriptUI view,
   * `custom && !os`), and the frame would go blank. */
  function drawOverlay() {
    var graphics = this.graphics;
    graphics.drawOSControl();
    var ctx = state.ctx;
    if (!ctx || !ctx.media_width || !ctx.media_height) {
      return;
    }
    var bounds = this.bounds;
    var width = bounds[2] - bounds[0];
    var height = bounds[3] - bounds[1];
    if (!(width > 1) || !(height > 1)) {
      return;
    }
    /* Media pixels -> control pixels; the control is sized to the media's
     * aspect, so one scale per axis describes it exactly. */
    var sx = width / ctx.media_width;
    var sy = height / ctx.media_height;
    var count = state.strokes.length + (state.painting ? 1 : 0);
    for (var i = 0; i < count; i++) {
      var stroke = i < state.strokes.length ? state.strokes[i] : state.painting;
      if (!stroke) {
        continue;
      }
      var polygon = strokePolygon(stroke);
      if (!polygon || polygon.length < 3) {
        continue;
      }
      var brush = graphics.newBrush(graphics.BrushType.SOLID_COLOR,
          stroke.tool === "add" ? INK_ADD : INK_REMOVE);
      graphics.newPath();
      graphics.moveTo(polygon[0][0] * sx, polygon[0][1] * sy);
      for (var k = 1; k < polygon.length; k++) {
        graphics.lineTo(polygon[k][0] * sx, polygon[k][1] * sy);
      }
      graphics.closePath();
      graphics.fillPath(brush);
    }
  }

  function showImage(file, caption) {
    if (!state.previewSlot || !file) {
      return false;
    }
    var exists = false;
    try {
      exists = file.exists === true;
    } catch (e) {
      exists = false;
    }
    if (!exists) {
      return false;
    }
    var img = ensurePreview(file);
    if (!img) {
      return false;
    }
    img.image = file;
    img.visible = true;
    img.preferredSize = previewSize();
    if (state.previewPlaceholder) {
      state.previewPlaceholder.visible = false;
    }
    if (state.previewCap) {
      state.previewCap.text = caption || "";
    }
    if (state.ui) {
      state.ui.layout.layout(true);
    }
    return true;
  }

  function showPreview(path, caption) {
    if (!path) {
      return hidePreview();
    }
    try {
      return showImage(new File(path), caption);
    } catch (e) {
      writeLog("preview failed: " + String(e.message || e));
      return false;
    }
  }

  function hidePreview() {
    if (state.previewImg) {
      state.previewImg.visible = false;
    }
    if (state.previewPlaceholder) {
      state.previewPlaceholder.visible = true;
    }
    if (state.ui) {
      state.ui.layout.layout(true);
    }
    return false;
  }

  /* The rendered frame the time indicator is on. The sequence the adapter
   * wrote is 1-based and starts at the job's first source frame. */
  function frameFile(frame) {
    var ctx = state.ctx;
    if (!ctx || !ctx.png_dir || !ctx.frame_count) {
      return null;
    }
    var index = Number(frame) - Number(ctx.frame_in || 0) + 1;
    if (!(index >= 1) || index > ctx.frame_count) {
      return null;
    }
    var name = String(Math.round(index));
    while (name.length < 5) {
      name = "0" + name;
    }
    var file = new File(ctx.png_dir + "/" + name + ".png");
    return file.exists ? file : null;
  }

  /* ---------------------------------------------------------------- */
  /* Painting                                                          */
  /* ---------------------------------------------------------------- */
  /* Coordinates arrive relative to the control (the engine fills the event
   * in with x - widget origin), and the control is sized to the media's
   * aspect, so the mapping to media pixels is one scale factor per axis. */
  function mediaPoint(event) {
    var img = state.previewImg;
    var ctx = state.ctx;
    if (!img || !ctx) {
      return null;
    }
    var bounds = null;
    try {
      bounds = img.bounds;
    } catch (e) {
      bounds = null;
    }
    if (!bounds) {
      return null;
    }
    var width = bounds[2] - bounds[0];
    var height = bounds[3] - bounds[1];
    if (!(width > 1) || !(height > 1) || !ctx.media_width || !ctx.media_height) {
      return null;
    }
    var x = Number(event.clientX);
    var y = Number(event.clientY);
    if (!isFinite(x) || !isFinite(y)) {
      return null;
    }
    return [
      Math.round(x * ctx.media_width / width),
      Math.round(y * ctx.media_height / height)
    ];
  }

  function beginStroke(event) {
    if (state.job || state.painting || !brushReady()) {
      return false;
    }
    if (Number(event.button) !== 0) {
      return false;
    }
    var point = mediaPoint(event);
    if (!point) {
      return false;
    }
    var tool = state.armedTool;
    if (event.shiftKey) {
      tool = (tool === "add") ? "remove" : "add";
    }
    state.painting = {
      tool: tool,
      radius: Math.max(2, state.brushSize),
      points: [point]
    };
    say((tool === "add" ? "+ painting in" : "- painting out") +
        " - drag, then release.", "busy");
    return true;
  }

  function continueStroke(event) {
    if (!state.painting) {
      return false;
    }
    var point = mediaPoint(event);
    if (!point) {
      return false;
    }
    var points = state.painting.points;
    var last = points[points.length - 1];
    var dx = point[0] - last[0];
    var dy = point[1] - last[1];
    /* One point every quarter brush: small polygons, same coverage - the
     * adapter fills the outline at full resolution. */
    var step = Math.max(2, state.painting.radius / 4);
    if ((dx * dx + dy * dy) < (step * step)) {
      return false;
    }
    points.push(point);
    return true;
  }

  function finishStroke() {
    if (!state.painting) {
      return false;
    }
    var stroke = state.painting;
    state.painting = null;
    if (!stroke.points.length) {
      return false;
    }
    state.strokes.push(stroke);
    refreshBrush();
    writeLog("stroke " + stroke.tool + " kept (" + stroke.points.length +
        " points, " + stroke.radius + " px)");
    say("Stroke kept - keep painting, or Apply fix & re-run.", "busy");
    return true;
  }

  /* A stroke as a polygon: the polyline offset by +-radius, which is what a
   * filled pen looks like. One point becomes a small ring. */
  function strokePolygon(stroke) {
    var points = stroke ? stroke.points : null;
    var radius = Math.max(1, (stroke && stroke.radius) || 1);
    if (!points || !points.length) {
      return null;
    }
    var i;
    if (points.length === 1) {
      var ring = [];
      for (i = 0; i < 12; i++) {
        var angle = i * Math.PI * 2 / 12;
        ring.push([points[0][0] + Math.cos(angle) * radius,
                   points[0][1] + Math.sin(angle) * radius]);
      }
      return ring;
    }
    var left = [];
    var right = [];
    for (i = 0; i < points.length; i++) {
      var previous = points[Math.max(0, i - 1)];
      var next = points[Math.min(points.length - 1, i + 1)];
      var dx = next[0] - previous[0];
      var dy = next[1] - previous[1];
      var length = Math.sqrt(dx * dx + dy * dy) || 1;
      var nx = -dy / length;
      var ny = dx / length;
      left.push([points[i][0] + nx * radius, points[i][1] + ny * radius]);
      right.push([points[i][0] - nx * radius, points[i][1] - ny * radius]);
    }
    var polygon = left;
    for (i = right.length - 1; i >= 0; i--) {
      polygon.push(right[i]);
    }
    return polygon;
  }

  function usableStrokeCount() {
    return state.strokes.length;
  }

  function selectedStroke() {
    var list = state.brushList;
    if (!list || list.selection === null || typeof list.selection === "undefined") {
      return null;
    }
    return state.strokes[list.selection.index] || null;
  }

  function updateFrameNote(layer) {
    if (!state.frameNote) {
      return;
    }
    if (!layer) {
      state.frameNote.text = "Frame: -";
      return;
    }
    try {
      var comp = activeComp();
      var frame = frameAt(layer, comp);
      var ctx = state.ctx;
      if (ctx && ctx.frame_count) {
        var index = frame - Number(ctx.frame_in || 0) + 1;
        state.frameNote.text = "Frame " + index + " of " + ctx.frame_count +
            " - the time indicator picks the re-track key frame";
      } else {
        state.frameNote.text = "Frame " + frame +
            " - the time indicator picks the re-track key frame";
      }
    } catch (e) {
      state.frameNote.text = "Frame: -";
    }
  }

  function updateBrushButtons() {
    var idle = !state.job;
    var armed = idle && brushReady();
    if (state.refreshBtn) {
      state.refreshBtn.enabled = idle;
    }
    /* The armed tool reads as "current" — its button is the one that is not
     * pressable — but a selected stroke can always be flipped either way, so
     * both buttons stay live while something is selected. */
    var picked = armed && !!selectedStroke();
    if (state.markAddBtn) {
      state.markAddBtn.enabled = armed && (picked || state.armedTool !== "add");
    }
    if (state.markRemoveBtn) {
      state.markRemoveBtn.enabled = armed && (picked || state.armedTool !== "remove");
    }
    var hasStrokes = armed && state.strokes.length > 0;
    if (state.undoStrokeBtn) {
      state.undoStrokeBtn.enabled = hasStrokes;
    }
    if (state.clearStrokesBtn) {
      state.clearStrokesBtn.enabled = hasStrokes;
    }
    if (state.brushSizeSlider) {
      state.brushSizeSlider.enabled = armed;
    }
    if (state.removeForceSlider) {
      state.removeForceSlider.enabled = armed;
    }
    if (state.applyBrushBtn) {
      state.applyBrushBtn.enabled = idle && brushReady() &&
          usableStrokeCount() > 0;
    }
    if (state.openAppBtn) {
      state.openAppBtn.enabled = idle;
    }
    if (state.appPathBtn) {
      state.appPathBtn.enabled = idle;
    }
  }

  function refreshBrush() {
    var list = state.brushList;
    if (!list) {
      return;
    }
    var layer = strokeLayer();
    var keep = null;
    if (list.selection && typeof list.selection.index === "number") {
      keep = list.selection.index;
    }
    var i;
    list.removeAll();
    for (i = 0; i < state.strokes.length; i++) {
      var stroke = state.strokes[i];
      list.add("item", (stroke.tool === "add" ? "[+] " : "[-] ") + "stroke " +
          (i + 1) + " \u00b7 " + stroke.radius + " px");
    }
    if (keep !== null && keep < state.strokes.length) {
      list.selection = keep;
    }

    if (state.brushHint) {
      if (!brushReady()) {
        state.brushHint.text = "Run Remove Background first \u2014 the frame " +
            "it returns becomes your canvas.";
      } else if (!state.ctx.png_dir) {
        state.brushHint.text = "The rendered frames are gone \u2014 run Remove " +
            "Background again.";
      } else if (!state.strokes.length) {
        state.brushHint.text = "Drag on the frame. + keeps pixels, \u2212 " +
            "deletes them; hold Shift for the other. Armed: " +
            (state.armedTool === "add" ? "+ keep" : "\u2212 delete") + ".";
      } else {
        var keeps = 0;
        var deletes = 0;
        for (var n = 0; n < state.strokes.length; n++) {
          if (state.strokes[n].tool === "add") {
            keeps++;
          } else {
            deletes++;
          }
        }
        state.brushHint.text = keeps + " keep \u00b7 " + deletes +
            " delete \u2014 Apply fix & re-run, or keep painting.";
      }
    }
    updateFrameNote(layer);
    updateBrushButtons();
  }

  /* One button, two jobs: with a stroke selected it flips that stroke,
   * otherwise it arms the tool the next drag paints with. */
  function markStroke(intent) {
    state.armedTool = intent;
    var stroke = selectedStroke();
    if (stroke) {
      stroke.tool = intent;
      refreshBrush();
      say("Stroke flipped to " + (intent === "add" ? "+ (paint in)" :
                                                       "- (paint out)") +
          ".", "busy");
      return;
    }
    refreshBrush();
    say("The next stroke will be " + (intent === "add" ? "+ (paint in)" :
                                                       "- (paint out)") +
        " - drag on the frame.", "busy");
  }

  function undoStroke() {
    if (!state.strokes.length) {
      say("There is no stroke to undo.", "error");
      return;
    }
    state.strokes.pop();
    refreshBrush();
    say("Stroke removed.", "busy");
  }

  function clearStrokes() {
    if (!state.strokes.length) {
      return;
    }
    state.strokes = [];
    refreshBrush();
    say("Strokes cleared.", "busy");
  }


  /* ---------------------------------------------------------------- */
  /* Open in Better Brush                                              */
  /* ---------------------------------------------------------------- */
  /* A panel cannot launch a process — this build's `system.callSystem`
   * throws and `File.execute` only opens a file — so the request goes to the
   * adapter, the one process in this bridge that can start one. The adapter
   * looks for the desktop app at the recorded path, falls back to a search,
   * remembers whatever it found, and answers in app-open-status-<id>.json
   * (the tick picks that up). */

  function readAppPath() {
    try {
      var text = readFile(state.bridge, APP_PATH_FILE);
      var data = text ? parse(text) : null;
      return (data && data.exe) ? String(data.exe) : "";
    } catch (e) {
      return "";
    }
  }

  function saveAppPath() {
    var field = state.appPathEdit;
    var path = field ? String(field.text || "").replace(/^\s+|\s+$/g, "") : "";
    if (!path) {
      say("Paste the path to BetterBrush.exe first.", "error");
      return;
    }
    if (!/better[\s_-]*brush[^\\\/]*\.exe$/i.test(path)) {
      say("That does not look like BetterBrush.exe — check the path.", "error");
      return;
    }
    try {
      writeFile(state.bridge, APP_PATH_FILE,
                hasJSON() ? JSON.stringify({ exe: path }, null, 2) : path);
    } catch (e) {
      say("Could not write the app path: " + String(e.message || e), "error");
      return;
    }
    state.appPath = path;
    writeLog("app path saved: " + path);
    say("Better Brush app remembered — use Open in Better Brush.", "ok");
  }

  /* The clip the brush is working on: the stroke layer's own source, else the
   * path the last job ran on. */
  function clipPath() {
    var layer = strokeLayer();
    if (layer) {
      try {
        var src = layer.source;
        if (src && src.file && src.file.fsName) {
          return String(src.file.fsName);
        }
      } catch (e) {
        /* no file source on this layer */
      }
    }
    return (state.ctx && state.ctx.source_path) ? String(state.ctx.source_path) : "";
  }

  function openInApp() {
    var file = clipPath();
    if (!file) {
      say("No clip to open — select the footage layer.", "error");
      return;
    }
    var id = "open" + nowMs();
    var doc = { id: id, source: file, created: new Date().toString() };
    try {
      writeFile(state.bridge, APP_OPEN_PREFIX + id + ".json",
                hasJSON() ? JSON.stringify(doc, null, 2) : String(doc));
    } catch (e) {
      say("Could not ask the adapter: " + String(e.message || e), "error");
      return;
    }
    state.appRequest = id;
    state.appRequestAt = nowMs();
    writeLog("open in app requested: " + file);
    say("Opening Better Brush\u2026", "busy");
  }

  function pollAppRequest() {
    if (!state.appRequest) {
      return;
    }
    if (nowMs() - state.appRequestAt > 30000) {
      /* An adapter started before v0.4.0 has no idea these files exist and
       * would leave the button spinning forever; say which clock stopped. */
      state.appRequest = "";
      say("The adapter did not answer — it is probably an older one. " +
          "Restart it: run_adapter.py watch", "error");
      return;
    }
    var text = null;
    try {
      text = readFile(state.bridge, APP_STATUS_PREFIX + state.appRequest + ".json");
    } catch (e) {
      return;
    }
    if (!text) {
      return;
    }
    var answer = parse(text);
    state.appRequest = "";
    if (!answer) {
      return;
    }
    if (answer.state === "ok") {
      if (answer.exe) {
        state.appPath = String(answer.exe);
        if (state.appPathEdit) {
          state.appPathEdit.text = state.appPath;
        }
      }
      say("Better Brush opened on " + (answer.source || "the clip") + ".", "ok");
    } else {
      say(answer.message || "The adapter could not open Better Brush.", "error");
      if (state.appPathEdit) {
        try {
          state.appPathEdit.active = true;
        } catch (e2) {
          /* focusing is a nicety, not a requirement */
        }
      }
    }
  }

  /* ---------------------------------------------------------------- */
  /* Starting the adapter when EffectCraft opens                      */
  /* ---------------------------------------------------------------- */
  /* Nothing is registered at logon any more: the installer registers a
   * scheduled task with a WMI trigger on effectcraft.exe itself, so the
   * adapter comes up *when EffectCraft does*. This is the safety net for
   * when that did not happen — a machine the task was never registered on,
   * a task somebody deleted, a run before the installer finished. The panel
   * reads the adapter's own heartbeat (stamped every watch cycle) and, if it
   * is stale, executes <bridge>/start-adapter.cmd, which carries the exact
   * command the installer chose.
   *
   * File.execute() goes through the OS (file.executeFile), so it needs the
   * write/network permission the panel already refuses to run without, and
   * with "Warn User When Executing Files" on — the engine's default —
   * EffectCraft asks first. One confirmation, and only in the case where the
   * adapter really is missing. */
  function adapterAlive() {
    try {
      var text = readFile(state.bridge, ADAPTER_ALIVE_FILE);
      var stamp = text ? parseInt(text, 10) : 0;
      return stamp > 0 ? stamp : 0;
    } catch (e) {
      return 0;
    }
  }

  function adapterIsAlive() {
    return (nowMs() - adapterAlive()) < ADAPTER_START_AFTER_MS;
  }

  /* The file that starts an adapter: the silent .vbs the installer writes
   * (wscript, no window), falling back to the .cmd a human can double-click. */
  function adapterStartFile() {
    for (var i = 0; i < ADAPTER_START_FILES.length; i++) {
      var file = new File(state.bridge + "/" + ADAPTER_START_FILES[i]);
      if (file.exists) {
        return file;
      }
    }
    return null;
  }

  function startAdapter(reason) {
    if (state.adapterStarted || !state.writable || !state.bridge) {
      return false;
    }
    state.adapterStarted = true;
    var file = adapterStartFile();
    var cmd = file ? file.fsName : "";
    if (!file) {
      writeLog("adapter not running (" + reason + ") and no start file in " +
          "the bridge");
      say("The adapter is not running and there is no start file — " +
          "re-run the installer.", "error");
      return false;
    }
    try {
      var res = file.execute();
      if (res && res.needsConfirmation) {
        /* "Warn User When Executing Files" is on (the engine's default), so
         * the host asks before a script may start a program — one modal,
         * answered once per start. The panel cannot answer it for the user,
         * so it says what the question is and how to stop being asked. */
        writeLog("adapter start: EffectCraft asked for permission (" +
            res.needsConfirmation + ")");
        say("EffectCraft is asking permission to start the adapter — allow " +
            "it. Preferences > Scripting > \"Warn User When Executing Files\" " +
            "turns this question off.", "busy");
        return true;
      }
      if (res && res.error) {
        writeLog("could not start the adapter: " + String(res.error));
        say("The adapter could not be started: " + String(res.error), "error");
        return false;
      }
      writeLog("adapter not running (" + reason + ") -> executed " + cmd);
      say("Starting the adapter\u2026", "busy");
      return true;
    } catch (e) {
      writeLog("could not start the adapter: " + String(e.message || e));
      say("The adapter is not running and it could not be started: " +
          String(e.message || e), "error");
      return false;
    }
  }

  /* The correction's key frame: the time indicator's position *within the
   * job's range*, because JobRequest.correction_frame is 0-based relative to
   * frame_in, not an absolute source frame. */
  function correctionFrameFor(job) {
    var comp = activeComp();
    if (!comp) {
      return 0;
    }
    var local = Math.round((comp.time - job.source_layer.start) * job.media.frame_rate);
    var rel = local - job.range.frame_in;
    if (rel < 0) {
      rel = 0;
    }
    if (job.range.frame_out >= 0) {
      var last = job.range.frame_out - job.range.frame_in;
      if (rel > last) {
        rel = last;
      }
    }
    return rel;
  }

  function submitCorrection() {
    if (state.job) {
      say("A job is already running — cancel it first.");
      return;
    }
    var ctx = state.ctx;
    if (!ctx || !ctx.work_dir) {
      say("Nothing to correct yet — click Remove Background first.", "error");
      return;
    }
    /* The correction re-uses the document that produced this matte, so the
     * source layer does not have to be selected any more — the user may well
     * have clicked elsewhere since painting. A context written by an older
     * panel has no document, and then the selection is read as before. */
    var job;
    try {
      if (ctx.job) {
        job = hasJSON() ? JSON.parse(JSON.stringify(ctx.job)) : ctx.job;
        job.id = jobId();
        job.created = new Date().toString();
      } else {
        job = collect();
      }
    } catch (e) {
      say(String(e.message || e), "error");
      return;
    }
    if (ctx.comp_name && job.comp.name !== ctx.comp_name) {
      say("The last run was in \"" + ctx.comp_name + "\" — re-run Remove " +
          "Background here before correcting.", "error");
      return;
    }
    if (ctx.layer_id && job.source_layer.id !== ctx.layer_id) {
      say("The correction context belongs to another layer (" +
          (ctx.layer_name || ctx.layer_id) + ") — re-run Remove Background " +
          "on this one.", "error");
      return;
    }
    if (!ctx.matte_item) {
      say("The matte layer could not be identified — re-run Remove Background.",
          "error");
      return;
    }
    if (!findMatteLayer(activeComp(), ctx.matte_layer_index)) {
      say("The matte layer is gone — re-run Remove Background.", "error");
      return;
    }

    /* Strokes are polylines in media pixels; the engine wants polygons, so
     * each one is outlined by its brush radius here. */
    var add = [];
    var remove = [];
    for (var i = 0; i < state.strokes.length; i++) {
      var stroke = state.strokes[i];
      var polygon = strokePolygon(stroke);
      if (!polygon || polygon.length < 3) {
        continue;
      }
      if (stroke.tool === "add") {
        add.push(polygon);
      } else {
        remove.push(polygon);
      }
    }
    if (!add.length && !remove.length) {
      say("Paint at least one stroke on the frame first.", "error");
      return;
    }

    job.correction = {
      resume_dir: ctx.work_dir,
      frame: correctionFrameFor(job),
      add: add,
      remove: remove,
      replace_item: ctx.matte_item,
      /* No masks are involved any more: the strokes never touched the
       * composition, so there is nothing for the adapter to park. */
      mask_names: [],
      /* Red bites as hard as the force slider says (the adapter grows the
       * region and stamps it out of every frame the matte still covers). */
      remove_force: state.removeForce
    };
    dispatch(job, "Correction");
  }

  /* ---------------------------------------------------------------- */
  /* Submit / poll                                                     */
  /* ---------------------------------------------------------------- */
  function submit() {
    var job;
    try {
      job = collect();
    } catch (e) {
      say(String(e.message || e), "error");
      return;
    }
    dispatch(job, "Job");
  }

  /* One write path for both a fresh run and a correction: the job document
   * differs, the bridge, the polling and the cancellation do not. */
  function dispatch(job, verb) {
    if (state.job) {
      say("A job is already running — cancel it first.");
      return;
    }
    if (!state.writable) {
      say(state.blocked ||
          "EffectCraft is not allowed to write files. Enable Preferences > " +
          "Scripting & Expressions > \"Allow Scripts to Write Files and " +
          "Access Network\", then reopen this panel.", "error");
      return;
    }

    try {
      writeFile(state.bridge, JOB_PREFIX + job.id + ".json",
                hasJSON() ? JSON.stringify(job, null, 2) : String(job));
    } catch (e2) {
      say("Could not write the job file: " + String(e2.message || e2), "error");
      return;
    }

    state.job = job.id;
    state.jobDoc = job;
    setEnabled(false, true);
    setBar(-1);
    writeLog("submitted " + job.id + " (" + verb + ") -> " + state.bridge);
    say(verb + " " + job.id + " queued — waiting for the adapter…", "busy");
    state.jobStart = nowMs();
    updateElapsed();
    /* wake() rather than armTick(): if the repeating task has gone quiet it
     * is cancelled and replaced here, not left dead while the job runs. */
    wake();
  }

  /* ---------------------------------------------------------------- */
  /* The tick — the panel updating itself                              */
  /* ---------------------------------------------------------------- */
  /* The panel has no timer of its own (Window.onTimer is not a ScriptUI
   * property here — verified), so the work is driven by app.scheduleTask.
   * Measured, not guessed: **EffectCraft 0.11.1 never executes a scheduled
   * task at all** — its desktop session is built without a task runner
   * (upstream #1867, fixed in 0.17.0), so the task is queued forever, no
   * repaint is ever requested for it, and every "update" in the early logs
   * came from the user's own click. A v0.3.0 one-shot never ran either.
   * Two consequences, both here:
   *   - one *repeating* task, never re-armed from inside tick() (that would
   *     multiply tasks with every fire), with its liveness checked by a
   *     heartbeat: when a tick has not run for HEARTBEAT_MS, wake() cancels
   *     the dead task and arms a fresh one — so a build that *does* run
   *     timers keeps the panel fresh with no "Check status" button;
   *   - a hover pump (onPanelMove) so a build with no task runner still
   *     follows the job whenever the pointer is over the panel.
   * TIMER_PROBE_MS is how long an armed task gets before the panel says
   * plainly that this build never runs timers, instead of looking frozen. */
  var HEARTBEAT_MS = 4000;
  var TIMER_PROBE_MS = 5000;
  /* The task's source text. It must never throw — the engine cancels a
   * repeating task that throws, which would kill the timer silently — so
   * every step is guarded. It counts its own executions on a global the panel
   * reads: that is what separates "the engine never ran this" from "it ran
   * but could not reach the hook". */
  var TASK_CODE =
    "(function(){try{if(typeof $!=='undefined'&&$&&$.global){$.global.BetterBrushTaskRuns=(($.global.BetterBrushTaskRuns|0)+1);}}catch(e0){}" +
    "var p=null;" +
    "try{if(typeof $!=='undefined'&&$&&$.global&&$.global.BetterBrushPanel)p=$.global.BetterBrushPanel;}catch(e){}" +
    "try{if(!p&&typeof BetterBrushPanel!=='undefined'&&BetterBrushPanel)p=BetterBrushPanel;}catch(e2){}" +
    "try{if(!p&&typeof window!=='undefined'&&window.BetterBrushPanel)p=window.BetterBrushPanel;}catch(e3){}" +
    "if(!p)return false;" +
    "try{return p.tick?p.tick('task'):(p.poll?p.poll('task'):false);}catch(e4){return false;}})()";

  function nowMs() {
    return new Date().getTime();
  }

  /* True when the repeating task stopped firing (the task id alone cannot
   * say that — a task id stays valid whether or not it ever ran). A task that
   * was armed and has never fired counts as stale too: that is exactly the
   * v0.3.0 failure — armed, silent, panel frozen. */
  function tickStale() {
    var since = state.lastTick || state.armedAt;
    return since > 0 && (nowMs() - since) > HEARTBEAT_MS;
  }

  function armTick() {
    /* When the first attempt happened, even if it never got as far as a task
     * id — noteDeadTimer() times the diagnosis from here. */
    state.probeAt = state.probeAt || nowMs();
    if (state.task > 0) {
      return;
    }
    if (typeof app.scheduleTask !== "function") {
      return; /* wake() pumps the panel instead */
    }
    /* A previous load of this panel may still have its repeating task out
     * there: it probes the same globals, so it would tick this panel twice
     * per period. The id is parked on the shared global and cancelled here. */
    try {
      var prev = (typeof $.global === "object" && $.global)
          ? $.global.BetterBrushTickTask : 0;
      if (prev > 0 && typeof app.cancelTask === "function") {
        app.cancelTask(prev);
      }
    } catch (ep) {
      /* nothing to cancel */
    }
    try {
      var id = app.scheduleTask(TASK_CODE, TICK_MS, true);
      state.task = (typeof id === "number" && id > 0) ? id : -1;
      state.armedAt = nowMs();
      /* Once per panel load — this line is what says the auto-update is on,
       * and repair() has its own line, so re-arming never floods the log. */
      if (!state.armLogged) {
        state.armLogged = true;
        writeLog("tick armed task=" + state.task +
            (state.firstTick ? "" : " (waiting for first tick)"));
      }
      try {
        if (typeof $.global === "object" && $.global && state.task > 0) {
          $.global.BetterBrushTickTask = state.task;
        }
      } catch (e2) {
        /* the next load just cannot cancel this task — harmless */
      }
      if (state.task <= 0 && !state.armError) {
        state.armError = true;
        writeLog("scheduleTask returned " + id + " — wake() will pump the panel");
      }
    } catch (e) {
      state.task = -1;
      if (!state.armError) {
        state.armError = true;
        writeLog("armTick threw " + (e.message || e));
      }
    }
  }

  /* The heartbeat: drop a task that stopped firing and arm a new one. The
   * log line is de-duplicated — on a build where the task never fires this
   * would otherwise repeat forever. */
  function rearmTick(why) {
    if (state.task > 0 && typeof app.cancelTask === "function") {
      try {
        app.cancelTask(state.task);
      } catch (e) {
        /* already gone */
      }
    }
    state.task = -1;
    armTick();
    if (!state.repairLogged) {
      state.repairLogged = true;
      writeLog("tick repaired: " + why);
    }
  }

  /* The published entry point: the scheduled task's source text calls this
   * with "task", wake()/the hover pump with their own reason. Failures are
   * de-duplicated so a persistent one cannot flood panel.log at tick rate. */
  function tick(via) {
    state.lastTick = nowMs();
    if (via === "task" && !state.taskFired) {
      state.taskFired = true;
      writeLog("engine timer running (scheduled task fired by EffectCraft)");
    }
    if (!state.firstTick) {
      state.firstTick = true;
      writeLog("first tick (auto-update running) via " + (via || "wake"));
    }
    try {
      pollAppRequest();
      if (state.job) {
        updateElapsed();
        poll();
      } else {
        idleTick();
      }
    } catch (e) {
      var why = "tick failed: " + String(e.message || e);
      if (state.lastTickError !== why) {
        state.lastTickError = why;
        writeLog(why);
      }
    }
  }

  /* Opportunistic pump: every interaction in the panel advances one tick when
   * the timer is missing or has gone quiet, and is a no-op otherwise. */
  function wake() {
    noteDeadTimer();
    if (state.task <= 0) {
      armTick();
      tick("wake");
    } else if (tickStale()) {
      rearmTick("no tick for " + HEARTBEAT_MS + " ms");
      tick("wake");
    }
  }

  /* The engine's timer has now had every chance: it was armed at panel load
   * and this runs from an interaction, which means EffectCraft executed a
   * frame — and run_script_tasks() runs at the top of every frame, before any
   * handler, so a task due in that frame has already fired (taskFired) by the
   * time we get here. If it has not, this build never runs tasks: say so once
   * rather than looking frozen. */
  function noteDeadTimer() {
    if (state.timerNoted || state.taskFired || !state.probeAt) {
      return;
    }
    if (nowMs() - state.probeAt < TIMER_PROBE_MS) {
      return;
    }
    state.timerNoted = true;
    var runs = 0;
    try {
      if (typeof $ === "object" && $ && $.global && $.global.BetterBrushTaskRuns) {
        runs = $.global.BetterBrushTaskRuns | 0;
      }
    } catch (e) {
      /* no counter readable — the conclusion does not change */
    }
    var ver = "";
    try {
      ver = (typeof app !== "undefined" && app && app.version) ? String(app.version) : "";
    } catch (e2) {
      /* no version string */
    }
    if (runs > 0) {
      writeLog("engine timer ran " + runs + "x but never reached the panel hook (EffectCraft " +
          (ver || "?") + ")");
    } else if (state.task <= 0) {
      writeLog("app.scheduleTask did not arm (EffectCraft " + (ver || "?") +
          ") — hover pump only");
    } else {
      writeLog("no engine timer: EffectCraft " + (ver || "?") +
          " never ran the scheduled task (fixed upstream in 0.17.0) — hover pump only");
    }
    if (!state.job && !state.blocked && state.task > 0 && runs === 0) {
      say("This build never runs panel timers (fixed in 0.17.0) — the panel refreshes while you point at it.",
          "busy");
    }
  }

  /* The hover pump: on a build with no task runner the panel can only work
   * when the host sends it an event, and pointer movement over it is the one
   * that needs no click — a listener on the window root receives every
   * control's mousemove, because ScriptUI bubbles them to their containers.
   * Throttled to POLL_MS, and stood down entirely while the engine's own
   * timer is keeping the panel fresh (lastTick), so a fixed build never pays
   * for it twice. */
  function onPanelMove() {
    noteDeadTimer();
    var now = nowMs();
    if (state.blocked) {
      return;
    }
    if (now - state.lastPump < POLL_MS || now - state.lastTick < POLL_MS) {
      return;
    }
    state.lastPump = now;
    armTick();
    tick("hover");
  }

  /* No job in flight: keep the brush section honest about the comp — the
   * selection, the masks on it, and the frame the time indicator sits on
   * (which is what a correction will target). */
  function idleTick() {
    /* The adapter's first look, once, a grace period after EffectCraft
     * opened: a standalone adapter.exe needs a second or two to boot, so a
     * missing heartbeat at load is not yet a missing adapter. */
    if (state.adapterCheckAt &&
        nowMs() - state.adapterCheckAt > ADAPTER_START_AFTER_MS &&
        !adapterIsAlive()) {
      state.adapterCheckAt = 0;
      startAdapter("no heartbeat after load");
    }
    var comp = null;
    try {
      comp = activeComp();
    } catch (e) {
      comp = null;
    }
    var layer = strokeLayer();
    var frame = frameAt(layer, comp);
    /* The preview follows the time indicator: the brush paints on the frame
     * in front of the user, and that same frame is the correction's key
     * frame, so what is painted is what gets re-tracked. */
    if (frame !== state.shownFrame) {
      state.shownFrame = frame;
      var file = brushReady() ? frameFile(frame) : null;
      if (file) {
        showImage(file, state.previewCap ? state.previewCap.text : "");
      }
      updateFrameNote(layer);
    }
  }

  function frameAt(layer, comp) {
    if (!layer || !comp) {
      return -1;
    }
    try {
      var src = layer.source;
      var fps = (src && src.frameRate) || comp.frameRate || 24;
      var start = typeof layer.startTime === "number" ? layer.startTime : 0;
      return Math.round((comp.time - start) * fps);
    } catch (e) {
      return -1;
    }
  }

  /* Log a one-off observation, de-duplicated: the poll runs every 700 ms and
   * the same "still nothing" reason must not flood panel.log. */
  function note(reason) {
    if (state.lastNote === reason) {
      return;
    }
    state.lastNote = reason;
    writeLog("poll: " + reason);
  }

  /* Returns true when it handled a terminal status — the caller (the tick,
   * or Cancel, which polls first so it cannot drop a finished result) must
   * stop after that. */
  function poll() {
    if (!state.job) {
      return false;
    }
    var text;
    try {
      text = readFile(state.bridge, STATUS_PREFIX + state.job + ".json");
    } catch (e) {
      note("read failed: " + (e.message || e));
      return false;
    }
    if (!text) {
      note("no status file yet for " + state.job + " in " + state.bridge);
      /* Silence alone is not proof — the adapter may be mid-boot. The
       * heartbeat is: no status after 12 s *and* no stamp means nothing is
       * watching this bridge, and a job nobody picks up is exactly when a
       * missing adapter costs something. */
      if (state.jobStart && nowMs() - state.jobStart > 12000 &&
          !adapterIsAlive()) {
        startAdapter("no status for 12 s");
      }
      return false; /* the adapter has not picked the job up yet */
    }
    var status;
    try {
      status = parse(text);
    } catch (e2) {
      note("status parse failed: " + (e2.message || e2));
      return false; /* mid-write read: atomic replace makes this rare, never fatal */
    }
    if (!status || !status.state) {
      note("status document has no state");
      return false;
    }
    state.lastNote = "";

    var frac = -1;
    if (typeof status.total === "number" && status.total > 0) {
      frac = Math.max(0, Math.min(1, status.current / status.total));
    }
    if (status.state === "done") {
      frac = 1;
    }

    if (status.state === "queued" || status.state === "running") {
      var label = STAGE_LABEL[status.stage] || status.stage || "Working";
      var line = label +
          (status.message && status.message !== label ? " — " + status.message : "");
      /* Numbers where the adapter reports them: a bar at 0 during a model
       * load looks identical to a bar on a job that is not moving. */
      if (frac >= 0 && typeof status.current === "number" &&
          typeof status.total === "number") {
        line += "  " + status.current + "/" + status.total;
      }
      setBar(frac);
      say(line, "busy");
      updateElapsed();
      return false;
    }

    /* Terminal. The tick keeps running — from here on it is an idle tick. */
    writeLog("terminal state=" + status.state + " cmds=" +
        ((status.result && status.result.commands) || []).length);
    var doc = state.jobDoc;
    state.job = null;
    state.jobDoc = null;
    state.jobStart = 0;
    updateElapsed();

    if (status.state === "cancelled") {
      setBar(0);
      say("Cancelled.");
      setEnabled(true, false);
      refreshBrush();
      return true;
    }
    if (status.state === "error") {
      setBar(0);
      var err = status.error || {};
      say("Failed" + (err.kind ? " (" + err.kind + ")" : "") + ": " +
          (err.message || status.message || "unknown error"), "error");
      setEnabled(true, false);
      refreshBrush();
      return true;
    }

    /* The command list is pure index arithmetic: if layers moved while the
     * adapter was rendering, those indices now name other layers. Refuse
     * instead of matted something unrelated. */
    var drift = layerDrift(doc);
    if (drift) {
      setBar(0);
      say("Nothing applied: " + drift + " Re-run Remove Background.", "error");
      writeLog("drift refused: " + drift);
      setEnabled(true, false);
      return true;
    }

    var failures = applyCommands((status.result && status.result.commands) || []);
    var result = status.result || {};
    if (failures.length) {
      setBar(0);
      say("Rendered, but applying failed: " + failures.join(" | "), "error");
      writeLog("apply failed: " + failures.join(" | "));
      showPreview(result.preview,
          previewCaption(result, { suffix: "not applied" }));
    } else {
      setBar(1);
      if (result.corrected) {
        /* Same layer, new frames: only the folder to resume the *next*
         * correction from has moved on. */
        if (state.ctx) {
          if (result.work_dir) {
            state.ctx.work_dir = result.work_dir;
          }
          /* The corrected frames are a new sequence: the brush has to read
           * *these*, or the next stroke would be painted on the old matte. */
          if (result.png_dir) {
            state.ctx.png_dir = result.png_dir;
          }
          if (result.frame_count) {
            state.ctx.frame_count = result.frame_count;
          }
          if (doc && doc.range) {
            state.ctx.frame_in = doc.range.frame_in;
          }
          if (result.preview) {
            state.ctx.preview = result.preview;
            state.ctx.preview_cap = previewCaption(result, { prefix: "Corrected" });
          }
          state.ctx.saved = new Date().toString();
          saveContext(state.ctx);
        }
        /* The strokes the fix was built from are part of the new matte now:
         * they stop being strokes, so the next correction starts from a
         * clean list instead of re-applying yesterday's paint. */
        state.strokes = [];
        state.shownFrame = null;
        say("Corrected — the matte frames were swapped in place; " +
            "one Ctrl+Z takes it back.", "ok");
        showPreview(result.preview, previewCaption(result, { prefix: "Corrected" }));
      } else {
        var cap = previewCaption(result, {});
        var captured = captureContext(activeComp(), doc, result, cap);
        state.strokes = [];
        state.shownFrame = null;
        var how = result.simulated ? " (simulated matte)" : "";
        say("Applied" + how + " — " + MATTE_NAME + " is the track matte of layer " +
            (result.source_index || "?") + "." +
            (captured && captured.work_dir
                ? " Paint strokes in the viewer, then apply the fix."
                : ""), "ok");
        showPreview(result.preview, cap);
      }
    }
    setEnabled(true, false);
    refreshBrush();
    return true;
  }

  /* Returns "" when the layer the finished job was built for is still where
   * the command list assumes, else a sentence explaining the mismatch.
   *
   * Runs BEFORE the commands apply, so the source must be found at the index
   * the job recorded (`matte_index`) — `matte_index + 1` only becomes its
   * position once the matte has been imported above it. Checking the shifted
   * index first (as an earlier version did) refused every job whose source
   * was the bottom layer of the stack. */
  function layerDrift(doc) {
    if (!doc || !doc.target || !doc.source_layer) {
      return "";
    }
    var comp = activeComp();
    if (!comp) {
      return "The composition is no longer open.";
    }
    if (doc.comp && doc.comp.name && comp.name !== doc.comp.name) {
      return "A different composition is active.";
    }
    var at = doc.source_layer.index || doc.target.matte_index;
    if (at < 1 || at > comp.numLayers) {
      return "The source layer moved in the stack.";
    }
    var layer = comp.layer(at);
    if (!layer) {
      return "The source layer moved in the stack.";
    }
    if (doc.source_layer.id && typeof layer.id === "number" &&
        layer.id !== doc.source_layer.id) {
      /* Ids differ at the recorded index. Tolerate the post-apply layout
       * (matte inserted above pushed the source down one) so a re-poll after
       * a partial run does not report a false move. */
      if (at + 1 <= comp.numLayers) {
        var below = comp.layer(at + 1);
        if (below && below.id === doc.source_layer.id) {
          return "";
        }
      }
      return "The source layer is no longer at index " + at + ".";
    }
    return "";
  }

  function applyCommands(commands) {
    var failures = [];
    if (!commands || !commands.length) {
      failures.push("the adapter returned no commands");
      return failures;
    }
    /* One undo group, so a single Ctrl+Z takes the whole change back. */
    var opened = false;
    var correcting = !!(state.jobDoc && state.jobDoc.correction);
    try {
      if (typeof app.beginUndoGroup === "function") {
        app.beginUndoGroup(correcting ? "Better Brush: Correction"
                                      : "Better Brush: Remove Background");
        opened = true;
      }
      for (var i = 0; i < commands.length; i++) {
        var c = commands[i];
        try {
          var res = app.run(c.id, c.params);
          if (c.id === "file.import" || c.id === "file.replaceFootage") {
            var errs = (res && res.errors) ? res.errors : [];
            if (errs.length) {
              /* The footage did not land, so every index after it points at
               * the wrong layer — stop rather than matte something unrelated. */
              failures.push(c.id + ": " + errs.join("; "));
              break;
            }
          }
        } catch (e) {
          if (c.critical) {
            failures.push((c.label || c.id) + ": " + String(e.message || e));
            break;
          }
          failures.push((c.label || c.id) + " skipped: " + String(e.message || e));
        }
      }
    } finally {
      if (opened) {
        try {
          app.endUndoGroup();
        } catch (e2) {
          /* nothing else to do */
        }
      }
    }
    return failures;
  }

  function setEnabled(on, allowCancel) {
    if (state.btn) {
      state.btn.enabled = on;
    }
    if (state.cancelBtn) {
      state.cancelBtn.enabled = !!allowCancel;
    }
    /* The dock settings belong to the *next* job, so they stay out of the way
     * only while one is running; the brush controls follow the same gate. */
    var dock = [state.qualityDd, state.mattingChk, state.postChk,
                state.applyBrushBtn, state.refreshBtn, state.markAddBtn,
                state.markRemoveBtn, state.undoStrokeBtn, state.clearStrokesBtn,
                state.brushSizeSlider, state.removeForceSlider,
                state.openAppBtn, state.appPathBtn];
    for (var i = 0; i < dock.length; i++) {
      if (dock[i]) {
        dock[i].enabled = on;
      }
    }
    if (on) {
      updateBrushButtons();
    }
  }

  function cancelJob() {
    if (!state.job) {
      return;
    }
    /* The result may already be on disk — cancelling then would throw away a
     * finished job (which is exactly what happened in the field: the adapter
     * had finished, the panel had not noticed, and Cancel dropped it). */
    if (poll()) {
      return;
    }
    try {
      writeFile(state.bridge, "cancel-" + state.job, "1");
    } catch (e) {
      say("Could not cancel: " + String(e.message || e));
      return;
    }
    say("Cancel requested — waiting for the adapter to stop.");
    state.job = null;
    state.jobDoc = null;
    setEnabled(true, false);
  }


  function qualityLabel(id) {
    for (var i = 0; i < QUALITIES.length; i++) {
      if (QUALITIES[i].id === id) {
        return QUALITIES[i].label;
      }
    }
    return id || "";
  }

  /* One line under the image: enough to tell two runs apart without a trip
   * to panel.log. */
  function previewCaption(result, opts) {
    opts = opts || {};
    if (!result) {
      return "";
    }
    var bits = [];
    if (opts.prefix) {
      bits.push(opts.prefix);
    }
    if (result.frame_count) {
      bits.push(result.frame_count + " frames");
    }
    if (result.model) {
      bits.push(result.model);
    }
    if (result.quality) {
      bits.push(qualityLabel(result.quality));
    }
    if (result.simulated) {
      bits.push("simulated");
    }
    if (result.post_processed === false) {
      bits.push("no post-processing");
    }
    if (opts.suffix) {
      bits.push(opts.suffix);
    }
    return bits.join(" · ");
  }

  /* ---------------------------------------------------------------- */
  /* UI                                                                */
  /* ---------------------------------------------------------------- */
  /* Three titled sections — what to run, how to fix it, and how it is
   * going — instead of one flat stack of controls. ScriptUI gives no access
   * to the host theme, so hierarchy comes from grouping, type size and the
   * accent/muted colours above rather than from styling that is not there. */
  function buildUI(host) {
    var ui = host;
    var owned = false;
    if (!(ui instanceof Panel)) {
      ui = new Window("palette", "Better Brush", undefined, { resizeable: true });
      owned = true;
    }
    ui.orientation = "column";
    ui.alignChildren = ["fill", "top"];
    ui.spacing = 10;
    ui.margins = 14;

    /* Header: brand, version pushed to the right by a stretching spacer. */
    var head = ui.add("group");
    head.orientation = "row";
    head.alignChildren = ["left", "center"];
    head.alignment = ["fill", "top"];
    var brand = head.add("statictext", undefined, "Better Brush");
    styleText(brand, 15, ACCENT, true);
    var headSpace = head.add("group");
    headSpace.alignment = ["fill", "fill"];
    var ver = head.add("statictext", undefined, "v" + VERSION);
    styleText(ver, 9, MUTED);

    /* A hairline under the brand. ScriptUI has no border styling, but a
     * two-pixel custom-drawn group reads as one. */
    var rule = ui.add("group");
    rule.alignment = ["fill", "top"];
    rule.preferredSize = [220, 4];
    rule.onDraw = function () {
      var graphics = this.graphics;
      var bounds = this.bounds;
      graphics.newPath();
      graphics.rectPath(0, 1, bounds[2] - bounds[0], 1);
      graphics.fillPath(graphics.newBrush(graphics.BrushType.SOLID_COLOR,
          [0.15, 0.58, 0.45, 0.5]));
    };

    /* ---------------- Result ---------------- */
    var render = ui.add("panel", undefined, "Result");
    render.orientation = "column";
    render.alignChildren = ["fill", "top"];
    render.spacing = 6;
    render.margins = [12, 16, 12, 10];

    /* The result image. The slot is a fixed-size group so that swapping the
     * image control (there is no way to re-point one) does not move the
     * controls below it. */
    state.previewSlot = render.add("group");
    state.previewSlot.orientation = "column";
    state.previewSlot.alignChildren = ["center", "center"];
    state.previewSlot.alignment = ["center", "top"];
    state.previewSlot.preferredSize = [PREVIEW_W, PREVIEW_H];
    state.previewPlaceholder = state.previewSlot.add("statictext", undefined,
        "No render yet.", { multiline: true });
    state.previewPlaceholder.preferredSize = [PREVIEW_W, 40];
    styleText(state.previewPlaceholder, 9, MUTED);

    state.previewCap = render.add("statictext", undefined,
        "No render yet — select a layer, then Remove Background.",
        { multiline: true });
    state.previewCap.preferredSize = [PREVIEW_W, 26];
    styleText(state.previewCap, 9, MUTED);

    var qRow = render.add("group");
    qRow.orientation = "row";
    qRow.alignChildren = ["left", "center"];
    qRow.alignment = ["fill", "top"];
    var qLabel = qRow.add("statictext", undefined, "Quality");
    styleText(qLabel, 11, null, true);
    var qSpace = qRow.add("group");
    qSpace.alignment = ["fill", "fill"];
    var labels = [];
    for (var q = 0; q < QUALITIES.length; q++) {
      labels.push(QUALITIES[q].label);
    }
    state.qualityDd = qRow.add("dropdownlist", undefined, labels);
    state.qualityDd.preferredSize = [104, 22];
    state.qualityDd.selection = 2; /* Balanced, as the AE panel defaults */
    state.qualityDd.helpTip = "Speed against edge quality — this is the checkpoint the run loads.";

    state.modelNote = render.add("statictext", undefined, QUALITIES[2].model);
    styleText(state.modelNote, 9, MUTED);

    var tRow = render.add("group");
    tRow.orientation = "row";
    tRow.alignChildren = ["left", "center"];
    tRow.alignment = ["fill", "top"];
    state.mattingChk = tRow.add("checkbox", undefined, "Matting");
    state.mattingChk.value = true;
    state.mattingChk.helpTip =
        "MatAnyone 2: refine the outline into a soft alpha (hair, partial edges). Slower.";
    var tSpace = tRow.add("group");
    tSpace.alignment = ["fill", "fill"];
    state.postChk = tRow.add("checkbox", undefined, "Post-processing");
    state.postChk.value = true;
    state.postChk.helpTip =
        "Smooth flicker between frames and clean the alpha's edge band.";
    var mattingNote = render.add("statictext", undefined,
        "Matting = MatAnyone 2 · Post-processing = temporal + edge cleanup");
    styleText(mattingNote, 9, MUTED);

    /* ---------------- Brush ---------------- */
    var brush = ui.add("panel", undefined, "Brush \u2014 fix a frame");
    brush.orientation = "column";
    brush.alignChildren = ["fill", "top"];
    brush.spacing = 6;
    brush.margins = [12, 16, 12, 10];

    state.brushHint = brush.add("statictext", undefined,
        "Paint on the frame above: + adds pixels back in, \u2212 removes " +
        "them, Shift swaps.", { multiline: true });
    state.brushHint.preferredSize = [220, 42];
    styleText(state.brushHint, 9, MUTED);

    state.brushList = brush.add("listbox", undefined, [], { numberOfRows: 4 });
    state.brushList.preferredSize = [220, 66];
    state.brushList.helpTip =
        "Strokes painted on this frame. + keeps pixels, - removes them.";

    var bRow = brush.add("group");
    bRow.orientation = "row";
    bRow.alignChildren = ["fill", "center"];
    bRow.alignment = ["fill", "top"];
    state.markAddBtn = bRow.add("button", undefined, "\uFF0B Add");
    state.markAddBtn.helpTip =
        "The next stroke paints pixels back in. With a stroke selected, marks that one instead.";
    state.markRemoveBtn = bRow.add("button", undefined, "\u2212 Remove");
    state.markRemoveBtn.helpTip =
        "The next stroke paints pixels out. With a stroke selected, marks that one instead.";

    var bRow2 = brush.add("group");
    bRow2.orientation = "row";
    bRow2.alignChildren = ["fill", "center"];
    bRow2.alignment = ["fill", "top"];
    state.undoStrokeBtn = bRow2.add("button", undefined, "Undo stroke");
    state.undoStrokeBtn.helpTip = "Remove the stroke painted last.";
    state.clearStrokesBtn = bRow2.add("button", undefined, "Clear");
    state.clearStrokesBtn.helpTip = "Remove every stroke and start over.";
    state.refreshBtn = bRow2.add("button", undefined, "Refresh");
    state.refreshBtn.helpTip = "Re-read the strokes on the layer.";

    var sizeRow = brush.add("group");
    sizeRow.orientation = "row";
    sizeRow.alignChildren = ["left", "center"];
    sizeRow.alignment = ["fill", "top"];
    var sizeLabel = sizeRow.add("statictext", undefined, "Brush");
    styleText(sizeLabel, 10, null, true);
    state.brushSizeSlider = sizeRow.add("slider", undefined, state.brushSize, 4, 160);
    state.brushSizeSlider.alignment = ["fill", "center"];
    state.brushSizeSlider.helpTip = "Brush size, in pixels of the clip.";
    state.brushSizeLabel = sizeRow.add("statictext", undefined,
        state.brushSize + " px");
    state.brushSizeLabel.preferredSize = [44, 18];
    styleText(state.brushSizeLabel, 9, MUTED);

    var forceRow = brush.add("group");
    forceRow.orientation = "row";
    forceRow.alignChildren = ["left", "center"];
    forceRow.alignment = ["fill", "top"];
    var forceLabel = forceRow.add("statictext", undefined, "Remove force");
    styleText(forceLabel, 10, null, true);
    state.removeForceSlider = forceRow.add("slider", undefined, state.removeForce, 1, 5);
    state.removeForceSlider.alignment = ["fill", "center"];
    state.removeForceSlider.helpTip =
        "How hard a red stroke bites: 1 is exactly what you painted, 5 takes " +
        "a margin around it and erases it from every frame the matte still covers.";
    state.removeForceLabel = forceRow.add("statictext", undefined,
        state.removeForce + " / 5");
    state.removeForceLabel.preferredSize = [44, 18];
    styleText(state.removeForceLabel, 9, MUTED);

    state.frameNote = brush.add("statictext", undefined, "Frame: \u2014");
    styleText(state.frameNote, 9, MUTED);

    state.applyBrushBtn = brush.add("button", undefined, "Apply fix & re-run");
    state.applyBrushBtn.preferredSize = [220, 26];
    state.applyBrushBtn.helpTip =
        "Re-track the clip from your strokes and swap the new frames into the existing matte layer.";

    /* ---------------- Better Brush app ---------------- */
    var appPanel = ui.add("panel", undefined, "Better Brush app");
    appPanel.orientation = "column";
    appPanel.alignChildren = ["fill", "top"];
    appPanel.spacing = 6;
    appPanel.margins = [12, 16, 12, 10];

    state.openAppBtn = appPanel.add("button", undefined, "Open in Better Brush");
    state.openAppBtn.helpTip =
        "Open this clip in the Better Brush desktop app, on the frame the time indicator sits on.";

    var pathRow = appPanel.add("group");
    pathRow.orientation = "row";
    pathRow.alignChildren = ["fill", "center"];
    pathRow.alignment = ["fill", "top"];
    state.appPathEdit = pathRow.add("edittext", undefined, "");
    state.appPathEdit.preferredSize = [150, 22];
    state.appPathEdit.helpTip =
        "Path to BetterBrush.exe. Set it once \u2014 the bridge remembers it.";
    state.appPathBtn = pathRow.add("button", undefined, "Save");
    state.appPathBtn.helpTip = "Remember this path for next time.";

    /* ---------------- Job ---------------- */
    var job = ui.add("panel", undefined, "Run");
    job.orientation = "column";
    job.alignChildren = ["fill", "top"];
    job.spacing = 6;
    job.margins = [12, 16, 12, 10];

    state.btn = job.add("button", undefined, "Remove Background");
    state.btn.helpTip = "Cuts the selected footage layer out of its background.";

    /* Cancel only — the status line under it updates itself (see "The tick"). */
    var row = job.add("group");
    row.orientation = "row";
    row.alignChildren = ["fill", "center"];
    row.alignment = ["fill", "top"];
    state.cancelBtn = row.add("button", undefined, "Cancel");
    state.cancelBtn.helpTip = "Ask the adapter to stop this job.";
    state.cancelBtn.enabled = false;

    state.bar = job.add("progressbar", undefined, 0, 100);
    state.bar.preferredSize = [220, 8];

    state.status = job.add("statictext", undefined, "Starting…", { multiline: true });
    state.status.preferredSize = [220, 54];
    styleText(state.status, 10, TONE.neutral);

    /* Deliberately *not* part of the status text: that string is written to
     * panel.log on every change, and a counter changes every tick. */
    state.elapsedEl = job.add("statictext", undefined, "");
    state.elapsedEl.preferredSize = [220, 14];
    styleText(state.elapsedEl, 9, MUTED);

    /* Every interaction also pumps one tick: if the timer chain is missing
     * (some builds never fire scheduleTask) the panel still keeps itself
     * current whenever the user touches it, and with a chain running wake()
     * is a no-op. */
    state.btn.onClick = function () { wake(); submit(); };
    state.cancelBtn.onClick = function () { wake(); cancelJob(); };
    state.refreshBtn.onClick = function () { wake(); refreshBrush(); };
    state.markAddBtn.onClick = function () { wake(); markStroke("add"); };
    state.markRemoveBtn.onClick = function () { wake(); markStroke("remove"); };
    state.undoStrokeBtn.onClick = function () { wake(); undoStroke(); };
    state.clearStrokesBtn.onClick = function () { wake(); clearStrokes(); };
    state.brushSizeSlider.onChanging = function () {
      state.brushSize = Math.round(this.value) || 1;
      if (state.brushSizeLabel) {
        state.brushSizeLabel.text = state.brushSize + " px";
      }
    };
    state.brushSizeSlider.onChange = state.brushSizeSlider.onChanging;
    state.removeForceSlider.onChanging = function () {
      state.removeForce = Math.round(this.value) || 1;
      if (state.removeForceLabel) {
        state.removeForceLabel.text = state.removeForce + " / 5";
      }
    };
    state.removeForceSlider.onChange = state.removeForceSlider.onChanging;
    state.applyBrushBtn.onClick = function () { wake(); submitCorrection(); };
    state.openAppBtn.onClick = function () { wake(); openInApp(); };
    state.appPathBtn.onClick = function () { wake(); saveAppPath(); };

    state.qualityDd.onChange = function () {
      wake();
      var index = this.selection ? this.selection.index : 2;
      state.opts.quality = QUALITIES[index].id;
      state.modelNote.text = QUALITIES[index].model;
    };
    state.mattingChk.onClick = function () {
      wake();
      state.opts.matting = !!this.value;
    };
    state.postChk.onClick = function () {
      wake();
      state.opts.post = !!this.value;
    };
    /* Both spellings exist across ScriptUI builds; setting an unknown one is
     * harmless, missing the real one would leave the buttons dead. */
    state.brushList.onSelectionChanged = function () {
      wake();
      updateBrushButtons();
    };
    state.brushList.onSelect = function () {
      wake();
      updateBrushButtons();
    };

    ui.layout.layout(true);
    state.ui = ui;
    updateBrushButtons();
    /* The hover pump (see onPanelMove): a listener on the root receives every
     * control's mousemove, because ScriptUI bubbles mouse events to their
     * containers — one listener covers the whole panel. */
    try {
      ui.addEventListener("mousemove", onPanelMove);
    } catch (e) {
      /* No mouse pump on this host: clicks still pump the panel. */
    }
    return owned ? ui : null;
  }

  /* ---------------------------------------------------------------- */
  /* Startup                                                           */
  /* ---------------------------------------------------------------- */
  function start() {
    if (!hasJSON()) {
      setBlocked("This EffectCraft build has no JSON support; the panel cannot run.");
      setEnabled(false, false);
      return;
    }
    state.bridge = resolveBridge();
    if (!state.bridge) {
      var where = "";
      try {
        where = (typeof $ === "object" && $.fileName) ? String($.fileName) : "(no $.fileName)";
      } catch (e) {
        where = "(unreadable)";
      }
      setBlocked("No bridge folder: this copy has no install stamp and " +
          "$.fileName gave \"" + where + "\". Re-run scripts/install.ps1 " +
          "(never run the .jsx straight from the repository).");
      setEnabled(false, false);
      return;
    }
    state.writable = checkWritable();
    if (!state.writable) {
      setBlocked("Script file writes are blocked: " +
          (state.writeError || "the write probe failed") +
          " — enable Preferences > Scripting & Expressions > \"Allow Scripts to " +
          "Write Files and Access Network\", then reopen this panel.");
      setEnabled(false, false);
      return;
    }
    state.blocked = "";
    /* The brush context survives a panel reload: it is what turns "run again"
     * into "correct this frame" after EffectCraft has been restarted — and it
     * is also where the last result image is remembered. */
    state.ctx = readContext();
    state.appPath = readAppPath();
    if (state.appPathEdit) {
      state.appPathEdit.text = state.appPath;
    }
    /* One look for the adapter, after the boot grace period (idleTick). */
    state.adapterCheckAt = nowMs();
    if (adapterIsAlive()) {
      writeLog("adapter alive (heartbeat " +
          Math.round((nowMs() - adapterAlive()) / 1000) + "s ago)");
    }
    say("Ready — select a footage layer, then click Remove Background.");
    setBar(0);
    if (state.ctx && state.ctx.preview) {
      showPreview(state.ctx.preview, state.ctx.preview_cap || "");
    }
    /* Strokes belong to the matte on screen, so a reload starts with none
     * and re-reads the current frame from the rendered sequence. */
    state.strokes = [];
    state.painting = null;
    state.shownFrame = null;
    refreshBrush();
    armTick();
  }

  var owned = buildUI(thisObj);

  /* Publish a hook the scheduled task can reach: scheduleTask takes source
   * text, not a closure, so tick() has to live on a global object. Publish to
   * *every* global this engine might evaluate the task against ($.global,
   * window, the bare global) — and the task probes all three as well. An
   * earlier version picked the first resolver that existed, which was not the
   * one the task's scope saw, so the task silently evaluated to false and the
   * panel never noticed a finished job. Unconditional: a hook left behind by
   * a previous load of this panel must be replaced, not kept. */
  var hook = { poll: poll, tick: tick, version: VERSION };
  var published = 0;
  try {
    if (typeof $.global === "object" && $.global) {
      $.global.BetterBrushPanel = hook;
      published++;
    }
  } catch (e1) {
    /* try the next one */
  }
  try {
    if (typeof window === "object" && window) {
      window.BetterBrushPanel = hook;
      published++;
    }
  } catch (e2) {
    /* try the next one */
  }
  try {
    var bare = (function () { return this; })();
    if (bare) {
      bare.BetterBrushPanel = hook;
      published++;
    }
  } catch (e3) {
    /* wake() pumps the panel from its own interactions instead */
  }
  /* After the hook: armTick() schedules work that needs it to exist. */
  start();
  writeLog("panel loaded v" + VERSION + " bridge=" + state.bridge +
      " writable=" + state.writable + " hook@" + published);

  if (owned) {
    owned.show();
  } else {
    /* Docked inside a panel host: there is no window to show, only the
     * layout we just filled — re-run it against the host, never against
     * `owned`, which is null here. */
    thisObj.layout.layout(true);
  }
})(this);
