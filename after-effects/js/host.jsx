/*
 * Better Brush — After Effects host script (ExtendScript / ES3).
 *
 * Everything the panel needs from the AE object model lives here. The panel is
 * a thin client: it talks HTTP to the local Better Brush engine for the AI
 * work, and calls these functions only to read the selection and to apply the
 * finished alpha movie back to the timeline.
 *
 * Each entry point returns a JSON string. ES3 has no guaranteed JSON, so a
 * small serialiser is included below.
 */

/* ------------------------------------------------------------------ */
/* JSON (with fallback)                                                */
/* ------------------------------------------------------------------ */
function bbEsc(s) {
    s = String(s);
    var out = "";
    for (var i = 0; i < s.length; i++) {
        var c = s.charAt(i);
        var code = s.charCodeAt(i);
        if (c === '"') {
            out += '\\"';
        } else if (c === '\\') {
            out += '\\\\';
        } else if (c === '\n') {
            out += '\\n';
        } else if (c === '\r') {
            out += '\\r';
        } else if (c === '\t') {
            out += '\\t';
        } else if (code < 32) {
            var hex = code.toString(16);
            while (hex.length < 4) { hex = "0" + hex; }
            out += "\\u" + hex;
        } else {
            out += c;
        }
    }
    return '"' + out + '"';
}

function bbJsonStringify(v) {
    if (v === null || v === undefined) { return "null"; }
    var t = typeof v;
    if (t === "number") { return isFinite(v) ? String(v) : "null"; }
    if (t === "boolean") { return v ? "true" : "false"; }
    if (t === "string") { return bbEsc(v); }
    if (v instanceof Array) {
        var parts = [];
        for (var i = 0; i < v.length; i++) { parts.push(bbJsonStringify(v[i])); }
        return "[" + parts.join(",") + "]";
    }
    var kv = [];
    for (var k in v) {
        if (v.hasOwnProperty && !v.hasOwnProperty(k)) { continue; }
        if (typeof v[k] === "function") { continue; }
        kv.push(bbEsc(k) + ":" + bbJsonStringify(v[k]));
    }
    return "{" + kv.join(",") + "}";
}

function bbJson(o) {
    try {
        if (typeof JSON !== "undefined" && JSON && JSON.stringify) {
            return JSON.stringify(o);
        }
    } catch (e) { /* fall through to the local serialiser */ }
    return bbJsonStringify(o);
}

/* ------------------------------------------------------------------ */
/* Diagnostics                                                         */
/* ------------------------------------------------------------------ */
function bbPing() {
    return "pong";
}

/* ------------------------------------------------------------------ */
/* Selection                                                           */
/* ------------------------------------------------------------------ */
function bbSelectedLayer() {
    try {
        if (!app.project) {
            return bbJson({ ok: false, reason: "No project is open." });
        }
        var comp = app.project.activeItem;
        if (!comp || !(comp instanceof CompItem)) {
            return bbJson({ ok: false, reason: "Open a composition and select a layer." });
        }
        var layer = null;
        var selected = comp.selectedLayers;
        if (selected && selected.length > 0) { layer = selected[0]; }
        if (!layer) {
            return bbJson({ ok: false, reason: "Select a layer in the timeline." });
        }

        var compFps = comp.frameRate > 0 ? comp.frameRate : 24;
        var info = {
            ok: false,
            comp: comp.name,
            compId: comp.id,
            layer: layer.name,
            layerId: layer.id,
            fps: compFps
        };

        if (layer.timeRemapEnabled) {
            info.reason = "Time-remapped layers aren't supported by the one-button flow — use the full app.";
            return bbJson(info);
        }

        var src = layer.source;
        if (!src) {
            info.reason = "This layer has no footage (solid, shape, text or adjustment). Use the full app.";
            return bbJson(info);
        }
        if (!(src instanceof FootageItem)) {
            info.reason = "Select a footage layer (not a pre-comp). Use the full app for comps.";
            return bbJson(info);
        }
        if (!src.file) {
            info.reason = "This footage isn't linked to a file on disk.";
            return bbJson(info);
        }

        var isStill = false;
        try { isStill = !!(src.mainSource && src.mainSource.isStill); } catch (eStill) {}

        var fw = src.width > 0 ? src.width : comp.width;
        var fh = src.height > 0 ? src.height : comp.height;
        var ffps = (src.frameRate && src.frameRate > 0) ? src.frameRate : compFps;
        var dur = src.duration ? src.duration : 0;
        var frames = isStill ? 1 : Math.max(1, Math.round(dur * ffps));

        info.ok = true;
        info.file = src.file.fsName;
        info.width = fw;
        info.height = fh;
        info.fps = ffps;
        info.frames = frames;
        info.isStill = isStill;
        info.kind = isStill ? "still" : "video";
        return bbJson(info);
    } catch (e) {
        return bbJson({ ok: false, reason: "Selection check failed: " + e.toString() });
    }
}

/* ------------------------------------------------------------------ */
/* Apply result                                                        */
/* ------------------------------------------------------------------ */
function bbFindComp(compId) {
    var proj = app.project;
    for (var i = 1; i <= proj.numItems; i++) {
        var item = proj.item(i);
        if (item instanceof CompItem && item.id === compId) { return item; }
    }
    if (proj.activeItem instanceof CompItem) { return proj.activeItem; }
    return null;
}

function bbApplyInner(comp, layer, file, mode, fps) {
    var undoOpen = false;
    try {
        app.beginUndoGroup("Better Brush - Remove Background");
        undoOpen = true;

        var originalFolder = null;
        try {
            if (layer.source && layer.source.parentFolder) {
                originalFolder = layer.source.parentFolder;
            }
        } catch (eFolder) {}

        var footage = app.project.importFile(new ImportOptions(file));
        if (originalFolder) {
            try { footage.parentFolder = originalFolder; } catch (eMove) {}
        }
        try { if (fps > 0) { footage.frameRate = fps; } } catch (eFps) {}
        try {
            if (footage.mainSource && footage.mainSource.alphaMode !== undefined) {
                footage.mainSource.alphaMode = AlphaMode.STRAIGHT;
            }
        } catch (eAlpha) {}

        if (String(mode) === "add") {
            var newLayer = comp.layers.add(footage);
            try {
                newLayer.startTime = layer.startTime;
                newLayer.inPoint = layer.inPoint;
                newLayer.outPoint = layer.outPoint;
            } catch (eTimes) {}
            return bbJson({
                ok: true,
                footage: footage.name,
                mode: "add",
                layerId: newLayer.id,
                compId: comp.id
            });
        }

        var originalName = layer.name;
        layer.replaceSource(footage, false);
        try { layer.name = originalName; } catch (eName) {}
        return bbJson({
            ok: true,
            footage: footage.name,
            mode: "replace",
            layerId: layer.id,
            compId: comp.id
        });
    } finally {
        if (undoOpen) { app.endUndoGroup(); }
    }
}

function bbApplyResult(outputPath, mode, fps, compId, layerId) {
    try {
        var file = new File(String(outputPath));
        if (!file.exists) {
            return bbJson({ ok: false, reason: "Result file not found: " + outputPath });
        }
        var comp = bbFindComp(compId);
        if (!comp) {
            return bbJson({ ok: false, reason: "Could not find the composition." });
        }
        var layer = null;
        for (var j = 1; j <= comp.numLayers; j++) {
            if (comp.layer(j).id === layerId) { layer = comp.layer(j); break; }
        }
        if (!layer) {
            return bbJson({ ok: false, reason: "The original layer no longer exists." });
        }
        return bbApplyInner(comp, layer, file, mode, fps);
    } catch (e) {
        return bbJson({ ok: false, reason: "Apply failed: " + e.toString() });
    }
}

/* ------------------------------------------------------------------ */
/* Process launching                                                   */
/* ------------------------------------------------------------------ */
function bbRunDetached(inner) {
    // `start "" /b` keeps the launched process in the background; pythonw.exe
    // means there is no console window to close.
    system.callSystem('cmd.exe /c ' + inner);
}

function bbLaunchEngine(pythonw, root, port) {
    try {
        pythonw = String(pythonw || "");
        root = String(root || "");
        port = parseInt(port, 10);
        if (isNaN(port) || port <= 0) { port = 8765; }
        if (!pythonw || !root) {
            return bbJson({
                ok: false,
                reason: "config.json is missing next to the panel - run after-effects/install.ps1 first."
            });
        }
        var inner = 'cd /d "' + root + '" && start "" /b "' + pythonw + '" -m src.engine --port ' + port + ' --quiet';
        bbRunDetached(inner);
        return bbJson({ ok: true });
    } catch (e) {
        return bbJson({ ok: false, reason: "Could not start the engine: " + e.toString() });
    }
}

function bbOpenFullApp(pythonPath, root) {
    try {
        pythonPath = String(pythonPath || "");
        root = String(root || "");
        if (!pythonPath || !root) {
            return bbJson({
                ok: false,
                reason: "config.json is missing next to the panel - run after-effects/install.ps1 first."
            });
        }
        var inner = 'cd /d "' + root + '" && start "" "' + pythonPath + '" -m src.main';
        bbRunDetached(inner);
        return bbJson({ ok: true });
    } catch (e) {
        return bbJson({ ok: false, reason: "Could not open Better Brush: " + e.toString() });
    }
}
