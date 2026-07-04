// canvas.js — drop zone, reorder, save, export.
//
// Behavior:
//   - Accept drops from fragment cards (data-fragment-ref).
//   - On drop, fetch full fragment body via /api/fragment/<ref>,
//     then render a canvas item.
//   - Reorder items via native HTML5 drag (dragstart sets a marker,
//     dragover calculates the insertion point, drop moves the item).
//   - Save posts the current canvas state to /api/canvas with the
//     canvas name from the input. Export hits /api/canvas/<slug>/
//     export and triggers a file download.
//   - Per-item delete button removes from the in-memory state.
//
// No framework, no drag-drop library. Vanilla DOM only.

(function () {
  "use strict";

  var surface = document.getElementById("canvas-surface");
  var nameInput = document.getElementById("canvas-name");
  var saveBtn = document.getElementById("canvas-save");
  var exportBtn = document.getElementById("canvas-export");
  var clearBtn = document.getElementById("canvas-clear");
  var status = document.getElementById("canvas-status");
  if (!surface) return;

  /** Current canvas items, in display order. */
  var items = [];
  /** Slug of the last saved canvas. Drives Export. */
  var currentSlug = null;

  function setStatus(text, kind) {
    if (!status) return;
    status.textContent = text || "";
    status.className = "canvas-status";
    if (kind === "ok") status.classList.add("canvas-status--ok");
    if (kind === "error") status.classList.add("canvas-status--error");
  }

  function clearEmptyState() {
    var empty = surface.querySelector(".canvas-empty");
    if (empty) empty.remove();
  }

  function rebuildEmptyState() {
    if (items.length === 0 && !surface.querySelector(".canvas-empty")) {
      var p = document.createElement("p");
      p.className = "canvas-empty";
      p.textContent = "Drag fragment cards from the chat panel here.";
      surface.appendChild(p);
    }
  }

  function findItem(ref) {
    for (var i = 0; i < items.length; i++) {
      if (items[i].ref === ref) return items[i];
    }
    return null;
  }

  function renderItem(item) {
    var el = document.createElement("article");
    el.className = "canvas-item";
    el.setAttribute("draggable", "true");
    el.dataset.fragmentRef = item.ref;
    el.dataset.order = String(item.order);

    var header = document.createElement("header");
    header.className = "canvas-item__header";

    var typeEl = document.createElement("span");
    typeEl.className =
      "canvas-item__type canvas-item__type--" + (item.type || "unknown");
    typeEl.textContent = item.type || "unknown";
    header.appendChild(typeEl);

    var src = document.createElement("span");
    src.className = "canvas-item__source";
    src.textContent = item.ref;
    header.appendChild(src);

    var del = document.createElement("button");
    del.type = "button";
    del.className = "canvas-item__delete";
    del.setAttribute("aria-label", "Remove from canvas");
    del.title = "Remove from canvas";
    del.textContent = "\u00d7";  // ×
    del.addEventListener("click", function () {
      items = items.filter(function (it) { return it.ref !== item.ref; });
      el.remove();
      reindex();
      rebuildEmptyState();
    });
    header.appendChild(del);
    el.appendChild(header);

    var body = document.createElement("div");
    body.className = "canvas-item__body";
    body.textContent = item.body || "";
    el.appendChild(body);

    if (item.annotation) {
      var ann = document.createElement("p");
      ann.className = "canvas-item__annotation";
      ann.textContent = item.annotation;
      el.appendChild(ann);
    }

    return el;
  }

  function reindex() {
    items.forEach(function (it, idx) { it.order = idx; });
    Array.prototype.forEach.call(
      surface.querySelectorAll(".canvas-item"),
      function (node) {
        var ref = node.dataset.fragmentRef;
        var it = findItem(ref);
        if (it) node.dataset.order = String(it.order);
      }
    );
  }

  function appendItem(item) {
    clearEmptyState();
    // Drop the fragment preview fetched lazily so we don't store
    // bodies in two places (chat preview vs canvas body).
    var fresh = Object.assign({}, item, { order: items.length });
    items.push(fresh);
    surface.appendChild(renderItem(fresh));
  }

  // ---------------------- Drop handlers ----------------------

  surface.addEventListener("dragover", function (ev) {
    ev.preventDefault();
    surface.classList.add("canvas-surface--over");
  });
  surface.addEventListener("dragleave", function (ev) {
    if (ev.target === surface) {
      surface.classList.remove("canvas-surface--over");
    }
  });
  surface.addEventListener("drop", function (ev) {
    ev.preventDefault();
    surface.classList.remove("canvas-surface--over");
    var ref = ev.dataTransfer.getData("text/fragment-ref");
    if (!ref) return;
    if (findItem(ref)) {
      setStatus("Fragment already on canvas.", "error");
      return;
    }
    fetch("/api/fragment/" + encodeURI(ref))
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (out) {
        if (!out.ok) {
          setStatus(
            (out.body && out.body.detail) || "Failed to fetch fragment",
            "error"
          );
          return;
        }
        appendItem({
          ref: out.body.ref,
          type: out.body.type,
          body: out.body.content,
          annotation: "",
        });
        setStatus("Added " + ref, "ok");
      })
      .catch(function (err) {
        setStatus("Drop failed: " + err.message, "error");
      });
  });

  // ---------------------- Reorder via HTML5 drag ----------------------

  // Vanilla HTML5 drag between siblings. We track which item is
  // being dragged via a module-local ref (canvas.js is the only
  // drag source on the canvas side; chat cards set their own
  // dataTransfer payload).
  var dragRef = null;
  surface.addEventListener("dragstart", function (ev) {
    var node = ev.target.closest(".canvas-item");
    if (!node) return;
    dragRef = node.dataset.fragmentRef;
    ev.dataTransfer.effectAllowed = "move";
    ev.dataTransfer.setData("text/canvas-ref", dragRef);
  });
  surface.addEventListener("dragover", function (ev) {
    // Allow drop for move operations as well.
    var node = ev.target.closest(".canvas-item");
    if (node) {
      ev.preventDefault();
      ev.dataTransfer.dropEffect = "move";
    }
  });
  surface.addEventListener("drop", function (ev) {
    if (!dragRef) return;
    var target = ev.target.closest(".canvas-item");
    if (!target) { dragRef = null; return; }
    var targetRef = target.dataset.fragmentRef;
    if (targetRef === dragRef) { dragRef = null; return; }
    ev.preventDefault();
    var srcIdx = items.findIndex(function (i) { return i.ref === dragRef; });
    var dstIdx = items.findIndex(function (i) { return i.ref === targetRef; });
    if (srcIdx < 0 || dstIdx < 0) { dragRef = null; return; }
    var moved = items.splice(srcIdx, 1)[0];
    items.splice(dstIdx, 0, moved);
    reindex();
    // Re-render the surface to reflect new order. Cheap because v0
    // canvases are small.
    Array.prototype.forEach.call(
      surface.querySelectorAll(".canvas-item"),
      function (n) { n.remove(); }
    );
    items.forEach(function (it) { surface.appendChild(renderItem(it)); });
    dragRef = null;
  });

  // ---------------------- Save / export / clear ----------------------

  function payload() {
    return {
      name: nameInput ? nameInput.value.trim() || "Untitled" : "Untitled",
      slug: currentSlug || undefined,
      items: items.map(function (it, idx) {
        return {
          ref: it.ref,
          order: idx,
          x: it.x || 0,
          y: it.y || 0,
          annotation: it.annotation || "",
        };
      }),
    };
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", function () {
      var name = nameInput ? nameInput.value.trim() : "";
      if (!name) {
        setStatus("Canvas name is required.", "error");
        return;
      }
      saveBtn.disabled = true;
      fetch("/api/canvas", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload()),
      })
        .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
        .then(function (out) {
          if (!out.ok) {
            setStatus(
              (out.body && out.body.detail) || "Save failed",
              "error"
            );
            return;
          }
          currentSlug = out.body.slug;
          setStatus("Saved (" + out.body.slug + ")", "ok");
        })
        .catch(function (err) {
          setStatus("Save failed: " + err.message, "error");
        })
        .finally(function () {
          saveBtn.disabled = false;
        });
    });
  }

  if (exportBtn) {
    exportBtn.addEventListener("click", function () {
      if (!currentSlug) {
        setStatus("Save the canvas first.", "error");
        return;
      }
      window.location.href = "/api/canvas/" + encodeURIComponent(currentSlug) + "/export";
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      items = [];
      currentSlug = null;
      Array.prototype.forEach.call(
        surface.querySelectorAll(".canvas-item"),
        function (n) { n.remove(); }
      );
      if (nameInput) nameInput.value = "";
      rebuildEmptyState();
      setStatus("Cleared.", "ok");
    });
  }
})();