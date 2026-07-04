// chat.js — chat panel submit + render.
//
// Behavior:
//   - Intercept form submit, POST to /api/chat, render the response.
//   - Empty-state placeholder is replaced on first successful response.
//   - Errors are surfaced as a styled inline message; the chat log
//     remains usable (the user can retry).
//
// No framework, no drag-drop library. Vanilla DOM only.

(function () {
  "use strict";

  var form = document.getElementById("chat-form");
  var input = document.getElementById("chat-input");
  var messages = document.getElementById("chat-messages");
  if (!form || !input || !messages) return;

  function clearEmptyState() {
    var empty = messages.querySelector(".chat-empty");
    if (empty) empty.remove();
  }

  function appendError(text) {
    var p = document.createElement("p");
    p.className = "chat-message__meta";
    p.style.color = "var(--err-fg)";
    p.textContent = text;
    messages.appendChild(p);
    messages.scrollTop = messages.scrollHeight;
  }

  function buildFragmentCard(frag) {
    var card = document.createElement("div");
    card.className = "fragment-card";
    card.setAttribute("draggable", "true");
    card.dataset.fragmentRef = frag.id;

    var typeEl = document.createElement("div");
    typeEl.className =
      "fragment-card__type fragment-card__type--" + (frag.type || "unknown");
    typeEl.textContent = frag.type || "unknown";
    card.appendChild(typeEl);

    var preview = document.createElement("div");
    preview.className = "fragment-card__preview";
    preview.textContent = frag.preview || "";
    card.appendChild(preview);

    var source = document.createElement("div");
    source.className = "fragment-card__source";
    source.textContent = frag.id;
    card.appendChild(source);
    return card;
  }

  function appendAssistantTurn(data) {
    var wrap = document.createElement("div");
    wrap.className = "chat-message chat-message--assistant";

    if (data.reply) {
      var text = document.createElement("p");
      text.className = "chat-message__text";
      text.textContent = data.reply;
      wrap.appendChild(text);
    }

    if (data.fragments && data.fragments.length) {
      var list = document.createElement("ul");
      list.className = "fragment-list";
      list.setAttribute("aria-label", "Fragments surfaced by the LLM");
      data.fragments.forEach(function (frag) {
        var li = document.createElement("li");
        li.appendChild(buildFragmentCard(frag));
        list.appendChild(li);
      });
      wrap.appendChild(list);
    }

    if (data.tool_calls) {
      var meta = document.createElement("p");
      meta.className = "chat-message__meta";
      meta.textContent =
        data.tool_calls + " tool call" + (data.tool_calls === 1 ? "" : "s");
      wrap.appendChild(meta);
    }

    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;
  }

  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    var text = input.value.trim();
    if (!text) return;
    clearEmptyState();

    // Render the user message immediately so the chat log feels
    // responsive even on a slow Ollama turn.
    var userMsg = document.createElement("div");
    userMsg.className = "chat-message chat-message--user";
    var userText = document.createElement("p");
    userText.className = "chat-message__text";
    userText.textContent = text;
    userMsg.appendChild(userText);
    messages.appendChild(userMsg);

    input.value = "";
    input.disabled = true;
    form.querySelector("button[type=submit]").disabled = true;

    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history: [] }),
    })
      .then(function (resp) {
        return resp.json().then(function (body) {
          return { ok: resp.ok, status: resp.status, body: body };
        });
      })
      .then(function (out) {
        if (!out.ok) {
          appendError(
            (out.body && out.body.detail) ||
              "Chat failed (HTTP " + out.status + ")"
          );
          return;
        }
        appendAssistantTurn(out.body);
      })
      .catch(function (err) {
        appendError("Network error: " + err.message);
      })
      .finally(function () {
        input.disabled = false;
        form.querySelector("button[type=submit]").disabled = false;
        input.focus();
      });
  });
})();