/*
 * Better Brush — Premiere Pro host script (ExtendScript / ES3).
 *
 * The panel is a thin client: it talks HTTP to the local Better Brush engine
 * for the AI work, and calls these functions only to read the timeline
 * selection and to put the finished alpha movie back on the timeline.
 *
 * Premiere's object model is not After Effects', so this file is the whole of
 * the port. Two differences are worth knowing before reading on:
 *
 *   1. Nothing in Premiere has an id. A sequence is found by name, and a clip
 *      is found by name on a track, so the selection is remembered in
 *      `bbTarget` while the job runs and re-resolved when the result comes
 *      back.
 *   2. There is no `Layer.replaceSource`. The Premiere equivalent is
 *      `ProjectItem.changeMediaPath`, which repoints a project item at new
 *      media — every clip cut from that item keeps its trims, effects and
 *      position, exactly like replacing a layer's source does in AE.
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
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */
/* Premiere counts time in ticks; there is no constant for it, so here it is. */
var BB_TICKS_PER_SECOND = 254016000000;

/* The clip the panel is working on, remembered between the selection call and
 * the apply call. Premiere objects are not addressable by id, and the panel
 * runs for minutes in between, so the target is re-resolved by name. */
var bbTarget = null;

function bbTicks(time) {
    if (!time) { return 0; }
    try { return parseFloat(time.ticks) || 0; } catch (e) { return 0; }
}

function bbNum(value) {
    var n = parseFloat(value);
    return isFinite(n) ? n : 0;
}

/* Paths from Premiere and from the file system differ in case and separators. */
function bbNormPath(path) {
    return String(path || "").replace(/\\/g, "/").toLowerCase();
}

function bbPing() {
    return "pong";
}

/* ------------------------------------------------------------------ */
/* Timeline lookups                                                    */
/* ------------------------------------------------------------------ */
function bbIsVideoClip(clip) {
    if (!clip) { return false; }
    var kind = "";
    try { kind = String(clip.type || ""); } catch (e1) {}
    if (!kind) {
        try { kind = String(clip.mediaType || ""); } catch (e2) {}
    }
    if (kind) { return kind === "Video"; }
    /* No usable type: assume video, and let the media path check below sort it
     * out — an audio clip has no video project item to speak of. */
    return true;
}

function bbVideoSelection(seq) {
    var picked = [];
    try {
        var selection = seq.getSelection();
        if (selection) {
            for (var i = 0; i < selection.length; i++) {
                if (bbIsVideoClip(selection[i])) { picked.push(selection[i]); }
            }
        }
    } catch (eSel) { /* very old builds throw instead of returning an empty list */ }
    return picked;
}

function bbTrackIndexOf(seq, clip) {
    for (var i = 0; i < seq.videoTracks.numTracks; i++) {
        var track = seq.videoTracks[i];
        for (var j = 0; j < track.clips.numItems; j++) {
            if (track.clips[j] === clip) { return i; }
        }
    }
    return -1;
}

/* Find the clip the panel last resolved. Premiere gives us no ids, and
 * `changeMediaPath` can rename the project item a clip is built from, so the
 * search is anchored on what a replace cannot move: the track and the clip's
 * start time. The name is only used to break a tie. */
function bbFindClip(seq, target) {
    if (!target) { return null; }
    var wanted = target.startTicks;
    var byPosition = null;
    for (var i = 0; i < seq.videoTracks.numTracks; i++) {
        var track = seq.videoTracks[i];
        for (var j = 0; j < track.clips.numItems; j++) {
            var clip = track.clips[j];
            if (Math.abs(bbTicks(clip.start) - wanted) > 1) { continue; }
            if (i === target.trackIndex || target.trackIndex < 0) { return clip; }
            if (!byPosition) { byPosition = clip; }
        }
    }
    if (byPosition) { return byPosition; }
    /* Last resort: the clip was moved, but kept its name. */
    for (var k = 0; k < seq.videoTracks.numTracks; k++) {
        var t = seq.videoTracks[k];
        for (var m = 0; m < t.clips.numItems; m++) {
            if (String(t.clips[m].name) === target.clipName) { return t.clips[m]; }
        }
    }
    return null;
}

function bbFindProjectItemByPath(path) {
    var wanted = bbNormPath(path);
    if (!wanted || !app.project) { return null; }
    try {
        var found = app.project.rootItem.findItemsMatchingMediaPath(path, false);
        if (found && found.length) { return found[0]; }
    } catch (eFast) { /* not available on older builds; fall back to a walk */ }
    var stack = [app.project.rootItem];
    while (stack.length) {
        var folder = stack.pop();
        if (!folder || !folder.children) { continue; }
        for (var i = 0; i < folder.children.numItems; i++) {
            var child = folder.children[i];
            if (bbIsBin(child)) { stack.push(child); continue; }
            var media = "";
            try { media = child.getMediaPath(); } catch (eMedia) {}
            if (media && bbNormPath(media) === wanted) { return child; }
        }
    }
    return null;
}

function bbIsBin(item) {
    try {
        if (typeof ProjectItemType !== "undefined" && item.type === ProjectItemType.BIN) {
            return true;
        }
    } catch (eType) {}
    try { return item.children !== undefined && !item.getMediaPath(); } catch (eBin) {}
    return false;
}

/* Import the alpha movie and hand back its project item. The path is settled
 * before importing and re-found afterwards, because importing an already-known
 * path is a no-op and returns nothing useful. */
function bbImportMedia(path) {
    var file = new File(String(path));
    if (!file.exists) { return null; }
    var known = bbFindProjectItemByPath(file.fsName);
    if (known) { return known; }
    try {
        app.project.importFiles([file.fsName], true, app.project.rootItem, false);
    } catch (eImport) {
        return null;
    }
    return bbFindProjectItemByPath(file.fsName);
}

/* ------------------------------------------------------------------ */
/* Selection                                                           */
/* ------------------------------------------------------------------ */
function bbSelectedLayer() {
    try {
        if (!app.project) {
            return bbJson({ ok: false, reason: "No project is open." });
        }
        var seq = app.project.activeSequence;
        if (!seq) {
            return bbJson({ ok: false, reason: "Open a sequence and select a clip." });
        }

        var picked = bbVideoSelection(seq);
        if (picked.length > 1) {
            return bbJson({
                ok: false,
                reason: "Several clips are selected. Select just the one to cut out."
            });
        }
        if (!picked.length) {
            return bbJson({ ok: false, reason: "Select a video clip in the timeline." });
        }
        var clip = picked[0];

        var item = null;
        try { item = clip.projectItem; } catch (eItem) {}
        if (!item) {
            return bbJson({
                ok: false,
                reason: "That clip has no media of its own (a colour matte or adjustment layer). Use the full app."
            });
        }

        var path = "";
        try { path = String(item.getMediaPath() || ""); } catch (ePath) {}
        if (!path) {
            return bbJson({ ok: false, reason: "This clip isn't linked to a file on disk." });
        }
        var file = new File(path);
        if (!file.exists) {
            return bbJson({ ok: false, reason: "The clip's media is offline: " + path });
        }

        var timebase = bbNum(seq.timebase);
        if (timebase <= 0) { timebase = BB_TICKS_PER_SECOND / 24; }
        var startTicks = bbTicks(clip.start);
        var endTicks = bbTicks(clip.end);
        var frames = Math.max(1, Math.round((endTicks - startTicks) / timebase));

        /* Remember the target for bbApplyResult — the panel may run for
         * minutes, and Premiere gives us no id to look it up by later. */
        bbTarget = {
            seqName: String(seq.name),
            clipName: String(clip.name),
            startTicks: startTicks,
            trackIndex: bbTrackIndexOf(seq, clip),
            itemName: String(item.name),
            resolved: true
        };

        var width = 0;
        var height = 0;
        try {
            width = bbNum(seq.frameSizeHorizontal);
            height = bbNum(seq.frameSizeVertical);
        } catch (eSize) {}

        /* The panel shows the clip's length; the engine reads the real frame
         * size, rate and length straight from the media file, so a sequence
         * that doesn't match its footage costs nothing but a slightly odd
         * label. */
        return bbJson({
            ok: true,
            comp: String(seq.name),
            compId: 0,
            layer: String(clip.name),
            layerId: 0,
            fps: BB_TICKS_PER_SECOND / timebase,
            width: width,
            height: height,
            frames: frames,
            isStill: false,
            kind: "video",
            file: String(file.fsName)
        });
    } catch (e) {
        return bbJson({ ok: false, reason: "Selection check failed: " + e.toString() });
    }
}

/* ------------------------------------------------------------------ */
/* Apply result                                                        */
/* ------------------------------------------------------------------ */
function bbTrackFreeBetween(track, startTicks, endTicks) {
    for (var i = 0; i < track.clips.numItems; i++) {
        var clip = track.clips[i];
        var s = bbTicks(clip.start);
        var e = bbTicks(clip.end);
        if (s < endTicks && e > startTicks) { return false; }
    }
    return true;
}

/* Mode "add": put the cut-out on the video track directly above the original
 * and switch the original off, so the edit underneath is untouched. Premiere's
 * scripting cannot create a track, so if there is nothing above — or something
 * already occupies that span — this says so instead of scribbling over it. */
function bbPlaceAbove(seq, clip, item) {
    var above = bbTrackIndexOf(seq, clip) + 1;
    if (above <= 0 || above >= seq.videoTracks.numTracks) {
        return bbJson({
            ok: false,
            reason: "There is no empty video track above this clip — add one and try again, or use Replace."
        });
    }
    var startTicks = bbTicks(clip.start);
    var endTicks = bbTicks(clip.end);
    var track = seq.videoTracks[above];
    if (!bbTrackFreeBetween(track, startTicks, endTicks)) {
        return bbJson({
            ok: false,
            reason: "The video track above this clip already has footage there — use Replace, or move the clip down a track."
        });
    }
    track.overwriteClip(item, clip.start);
    try { clip.disabled = true; } catch (eDisable) {}
    return bbJson({
        ok: true,
        footage: String(item.name),
        mode: "add",
        compId: 0,
        layerId: 0
    });
}

/* Mode "replace": repoint the clip's media at the cut-out. This is the
 * Premiere counterpart of AE's Layer.replaceSource — trims, effects and
 * position on every clip cut from that item are left exactly as they were. */
function bbReplaceMedia(item, path) {
    if (typeof item.changeMediaPath !== "function") { return false; }
    try {
        item.changeMediaPath(path, true);
        return true;
    } catch (eChecked) {
        try {
            item.changeMediaPath(path);
            return true;
        } catch (ePlain) {
            return false;
        }
    }
}

function bbApplyResult(outputPath, mode, fps, compId, layerId) {
    var undoOpen = false;
    try {
        var file = new File(String(outputPath));
        if (!file.exists) {
            return bbJson({ ok: false, reason: "Result file not found: " + outputPath });
        }
        if (!app.project) {
            return bbJson({ ok: false, reason: "No project is open." });
        }
        var seq = app.project.activeSequence;
        if (!seq) {
            return bbJson({ ok: false, reason: "No sequence is open." });
        }
        var target = bbTarget;
        var clip = bbFindClip(seq, target);
        if (!clip) {
            return bbJson({
                ok: false,
                reason: "The clip moved or was renamed while the job ran. Select it again and it will be applied on the next run."
            });
        }

        if (app.beginUndoGroup) {
            try {
                app.beginUndoGroup("Better Brush - Remove Background");
                undoOpen = true;
            } catch (eUndo) { undoOpen = false; }
        }

        var item = bbImportMedia(file.fsName);
        if (!item) {
            return bbJson({ ok: false, reason: "Could not import the cut-out into the project." });
        }

        if (String(mode) === "add") {
            return bbPlaceAbove(seq, clip, item);
        }

        var original = null;
        try { original = clip.projectItem; } catch (eOrig) {}
        if (original && bbReplaceMedia(original, file.fsName)) {
            return bbJson({
                ok: true,
                footage: String(item.name),
                mode: "replace",
                compId: 0,
                layerId: 0
            });
        }

        /* changeMediaPath isn't available (or refused): fall back to dropping
         * the cut-out where the clip was. That loses the clip's effects but
         * still leaves the sequence looking right. */
        var track = seq.videoTracks[bbTrackIndexOf(seq, clip) >= 0 ? bbTrackIndexOf(seq, clip) : 0];
        track.overwriteClip(item, clip.start);
        return bbJson({
            ok: true,
            footage: String(item.name),
            mode: "replace",
            compId: 0,
            layerId: 0
        });
    } catch (e) {
        return bbJson({ ok: false, reason: "Apply failed: " + e.toString() });
    } finally {
        if (undoOpen) {
            try { app.endUndoGroup(); } catch (eEndUndo) {}
        }
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
                reason: "config.json is missing next to the panel - run premiere-pro/install.ps1 first."
            });
        }
        var inner = 'cd /d "' + root + '" && start "" /b "' + pythonw + '" -m src.engine --port ' + port + ' --quiet';
        bbRunDetached(inner);
        return bbJson({ ok: true });
    } catch (e) {
        return bbJson({ ok: false, reason: "Could not start the engine: " + e.toString() });
    }
}

function bbOpenFullApp(python, root, file) {
    try {
        python = String(python || "");
        root = String(root || "");
        file = String(file || "");
        if (!python || !root) {
            return bbJson({
                ok: false,
                reason: "config.json is missing next to the panel - run premiere-pro/install.ps1 first."
            });
        }
        // Launched exactly like the engine: a detached pythonw.exe with no
        // console. A console child started from inside Premiere shares
        // Premiere's process group, and closing it can take Premiere down too.
        var inner = 'cd /d "' + root + '" && start "" /b "' + python + '" -m src.main';
        if (file) {
            // Hand the clip straight to the app so it opens on the footage the
            // panel was working on, instead of the welcome launcher.
            inner += ' "' + file + '"';
        }
        bbRunDetached(inner);
        return bbJson({ ok: true });
    } catch (e) {
        return bbJson({ ok: false, reason: "Could not open Better Brush: " + e.toString() });
    }
}
