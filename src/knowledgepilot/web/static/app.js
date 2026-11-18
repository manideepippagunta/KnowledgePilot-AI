/**
 * KnowledgePilot AI — Claude-Style & Agentic Frontend Controller (with Live SSE Token Streaming)
 * Compatible with FastAPI endpoints:
 *   GET    /api/status
 *   GET    /api/documents
 *   POST   /api/documents/upload
 *   DELETE /api/documents
 *   POST   /api/generator
 *   POST   /api/chat/stream  (Primary live SSE token stream)
 *   POST   /api/chat         (Automatic fallback)
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const uploadStatus = document.getElementById("upload-status");
  const documentList = document.getElementById("document-list");
  const docCountBadge = document.getElementById("doc-count-badge");
  const statDocs = document.getElementById("stat-docs");
  const statChunks = document.getElementById("stat-chunks");
  const ollamaDot = document.getElementById("ollama-dot");
  const ollamaStatusText = document.getElementById("ollama-status-text");
  const activeProviderPill = document.getElementById("active-provider-pill");
  const resetIndexBtn = document.getElementById("reset-index-btn");

  const chatScroll = document.getElementById("chat-scroll");
  const chatFeed = document.getElementById("chat-feed");
  const welcomeState = document.getElementById("welcome-state");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");

  const generatorSelect = document.getElementById("generator-select");
  const modelInput = document.getElementById("model-input");
  const topkSlider = document.getElementById("topk-slider");
  const topkValue = document.getElementById("topk-value");

  const evidenceDrawer = document.getElementById("evidence-drawer");
  const drawerBody = document.getElementById("drawer-body");
  const toggleInspectorBtn = document.getElementById("toggle-inspector-btn");
  const closeDrawerBtn = document.getElementById("close-drawer-btn");
  const inspectorCountBadge = document.getElementById("inspector-count-badge");
  const toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
  const sidebar = document.getElementById("sidebar");

  let activeEvidenceList = [];

  // ---------------------------------------------------------------------------
  // 1. Utilities & Safe Markdown / Citation Formatter
  // ---------------------------------------------------------------------------
  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatBytes(bytes) {
    if (!bytes || bytes <= 0) return "";
    const kb = bytes / 1024;
    return kb < 1024 ? `${kb.toFixed(1)} KB` : `${(kb / 1024).toFixed(2)} MB`;
  }

  /**
   * Converts plain / markdown text into clean editorial HTML and transforms
   * [1], [Document 1], or [Doc 1 p.49] markers into interactive citation pills.
   */
  function renderEditorialMarkdown(rawText, evidenceItems = []) {
    const evidenceCount = Array.isArray(evidenceItems) ? evidenceItems.length : Number(evidenceItems) || 0;
    let safe = escapeHtml(rawText);

    // Code blocks ```...```
    safe = safe.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);

    // Inline code `...`
    safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold **...**
    safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Convert [Document N], [Doc N], [Source N], or [N] into clickable citation pills
    safe = safe.replace(
      /\[(?:Document\s+|Doc\s+|Source\s+|Chunk\s+)?(\d+)(?:\s*[,:]?\s*(?:p\.?|page\s*)(\d+))?\]/gi,
      (match, numStr, pageStr) => {
        const idx = parseInt(numStr, 10);
        if (idx >= 1 && idx <= Math.max(evidenceCount, 10)) {
          const ev = Array.isArray(evidenceItems) ? evidenceItems[idx - 1] : null;
          const pg = pageStr || (ev && ev.page_number ? ev.page_number : null);
          const label = pg ? `Doc ${idx} p.${pg}` : `Doc ${idx}`;
          return `<button type="button" class="cite-pill" data-cite-index="${idx - 1}" title="Inspect Source [${idx}]">[${label}]</button>`;
        }
        return match;
      }
    );

    // Split paragraphs & bullet lists cleanly
    const blocks = safe.split(/\n{2,}/);
    return blocks
      .map((block) => {
        const lines = block.trim().split("\n");
        if (lines.every((l) => /^[-*•]\s+/.test(l.trim()))) {
          const items = lines
            .map((l) => `<li>${l.trim().replace(/^[-*•]\s+/, "")}</li>`)
            .join("");
          return `<ul>${items}</ul>`;
        }
        return `<p>${lines.join("<br/>")}</p>`;
      })
      .join("");
  }

  // ---------------------------------------------------------------------------
  // 2. Status & Document List Polling
  // ---------------------------------------------------------------------------
  async function refreshStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) return;
      const data = await res.json();

      const docCount = data.total_documents ?? data.document_count ?? data.documents_count ?? 0;
      const chunkCount = data.total_chunks ?? data.chunk_count ?? data.chunks_count ?? 0;
      const provider = data.provider ?? data.generator_provider ?? generatorSelect.value;
      const model = data.model ?? data.ollama_model ?? modelInput.value;
      const ollamaOk = Boolean(data.ollama_available ?? data.ollama_connected ?? true);

      statDocs.textContent = docCount;
      statChunks.textContent = chunkCount;
      docCountBadge.textContent = `${docCount} ${docCount === 1 ? "Doc" : "Docs"}`;
      activeProviderPill.textContent = `Provider: ${provider}${provider === "ollama" ? ` (${model})` : ""}`;

      if (ollamaOk) {
        ollamaDot.className = "status-dot online";
        ollamaStatusText.textContent = `Ollama Online (${model})`;
      } else {
        ollamaDot.className = "status-dot offline";
        ollamaStatusText.textContent = "Ollama Offline (Mock Ready)";
      }
    } catch (err) {
      console.error("Failed to fetch /api/status:", err);
    }
  }

  async function refreshDocuments() {
    try {
      const res = await fetch("/api/documents");
      if (!res.ok) return;
      const data = await res.json();
      const docs = Array.isArray(data) ? data : data.documents || [];

      if (docs.length === 0) {
        documentList.innerHTML = `
          <div class="empty-docs-state">
            No documents indexed yet. Upload a PDF or TXT file to ground the agent.
          </div>`;
        return;
      }

      documentList.innerHTML = docs
        .map((doc) => {
          const name = doc.file_name || doc.filename || doc.name || "Document";
          const ext = (doc.file_type || name.split(".").pop() || "DOC").replace(/^\./, "");
          const pages = doc.page_count ?? doc.pages ?? "";
          const chunks = doc.chunk_count ?? doc.chunks ?? 0;
          const size = formatBytes(doc.file_size_bytes ?? doc.size_bytes ?? doc.file_size);

          return `
            <div class="doc-item">
              <div class="doc-item-top">
                <span class="doc-ext">${escapeHtml(ext)}</span>
                <span class="doc-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
              </div>
              <div class="doc-meta">
                ${pages ? `<span>${pages} pages</span>` : ""}
                <span>• ${chunks} chunks</span>
                ${size ? `<span>• ${size}</span>` : ""}
              </div>
            </div>`;
        })
        .join("");
    } catch (err) {
      console.error("Failed to fetch /api/documents:", err);
    }
  }

  // ---------------------------------------------------------------------------
  // 3. Document Upload & Reset Index
  // ---------------------------------------------------------------------------
  async function uploadFiles(files) {
    if (!files || files.length === 0) return;

    uploadStatus.classList.remove("hidden");
    for (const file of files) {
      uploadStatus.textContent = `Indexing ${file.name}...`;
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("/api/documents/upload", {
          method: "POST",
          body: formData,
        });
        const payload = await res.json();
        if (!res.ok) {
          uploadStatus.textContent = `Error: ${payload.detail || "Upload failed"}`;
        } else {
          uploadStatus.textContent = `Indexed ${file.name}`;
        }
      } catch (err) {
        uploadStatus.textContent = `Upload error: ${err.message}`;
      }
    }

    await Promise.all([refreshStatus(), refreshDocuments()]);
    setTimeout(() => uploadStatus.classList.add("hidden"), 2500);
  }

  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => uploadFiles(e.target.files));

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    uploadFiles(e.dataTransfer.files);
  });

  resetIndexBtn.addEventListener("click", async () => {
    if (!confirm("Clear all indexed documents and vector chunks?")) return;
    await fetch("/api/documents", { method: "DELETE" });
    activeEvidenceList = [];
    renderEvidenceDrawer([]);
    await Promise.all([refreshStatus(), refreshDocuments()]);
  });

  // ---------------------------------------------------------------------------
  // 4. Generator Switching & Top-K Controls
  // ---------------------------------------------------------------------------
  topkSlider.addEventListener("input", () => {
    topkValue.textContent = topkSlider.value;
  });

  async function syncGeneratorConfig() {
    const provider = generatorSelect.value;
    const model = modelInput.value.trim() || "llama3.2:3b";
    try {
      await fetch("/api/generator", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, model }),
      });
      await refreshStatus();
    } catch (err) {
      console.error("Generator switch error:", err);
    }
  }

  generatorSelect.addEventListener("change", syncGeneratorConfig);
  modelInput.addEventListener("change", syncGeneratorConfig);

  // ---------------------------------------------------------------------------
  // 5. Right-Hand Evidence Artifact Drawer
  // ---------------------------------------------------------------------------
  function renderEvidenceDrawer(evidenceItems, highlightIndex = null) {
    activeEvidenceList = evidenceItems || [];
    inspectorCountBadge.textContent = activeEvidenceList.length;

    if (activeEvidenceList.length === 0) {
      drawerBody.innerHTML = `
        <div class="empty-drawer-state">
          No source passages for this turn. Ask a grounded question to inspect retrieved vector chunks.
        </div>`;
      return;
    }

    drawerBody.innerHTML = activeEvidenceList
      .map((item, idx) => {
        const fileName = item.file_name || item.document_name || item.source || "Document";
        const page = item.page_number ?? item.page ?? null;
        const score =
          typeof item.score === "number"
            ? item.score.toFixed(4)
            : typeof item.similarity === "number"
            ? item.similarity.toFixed(4)
            : null;
        const text = item.text || item.content || item.snippet || "";

        return `
          <div class="evidence-card ${highlightIndex === idx ? "highlighted" : ""}" id="evidence-card-${idx}">
            <div class="evidence-card-header">
              <span class="evidence-badge">[Doc ${idx + 1}${page ? ` · p.${page}` : ""}]</span>
              ${score !== null ? `<span class="evidence-score">Similarity: ${score}</span>` : ""}
            </div>
            <div class="evidence-meta">
              ${escapeHtml(fileName)}${page ? ` · Page ${page}` : ""}
            </div>
            <div class="evidence-passage">${escapeHtml(text)}</div>
          </div>`;
      })
      .join("");

    if (highlightIndex !== null) {
      const target = document.getElementById(`evidence-card-${highlightIndex}`);
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  }

  function openDrawer(evidenceItems, highlightIndex = null) {
    evidenceDrawer.classList.remove("closed");
    renderEvidenceDrawer(evidenceItems, highlightIndex);
  }

  toggleInspectorBtn.addEventListener("click", () => {
    evidenceDrawer.classList.toggle("closed");
  });
  closeDrawerBtn.addEventListener("click", () => {
    evidenceDrawer.classList.add("closed");
  });
  if (toggleSidebarBtn) {
    toggleSidebarBtn.addEventListener("click", () => sidebar.classList.toggle("open"));
  }

  // ---------------------------------------------------------------------------
  // 6. Agentic Chat Flow with Real-Time SSE Streaming (+ Automatic Fallback)
  // ---------------------------------------------------------------------------
  function autoResizeTextarea() {
    chatInput.style.height = "auto";
    chatInput.style.height = `${Math.min(chatInput.scrollHeight, 160)}px`;
  }

  chatInput.addEventListener("input", autoResizeTextarea);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.requestSubmit();
    }
  });

  document.querySelectorAll(".starter-card").forEach((card) => {
    card.addEventListener("click", () => {
      const prompt = card.getAttribute("data-prompt");
      if (prompt) {
        chatInput.value = prompt;
        chatForm.requestSubmit();
      }
    });
  });

  function finalizeAssistantTurn(assistantWrap, answerText, rawEvidence, provider, model, topK, elapsedSec) {
    const evidenceItems = (rawEvidence || []).map((item, i) => ({
      file_name: item.file_name || item.document_name || item.source || `Source ${i + 1}`,
      page_number: item.page_number ?? item.page ?? null,
      score: item.score ?? item.similarity ?? null,
      text: item.text || item.snippet || item.content || "",
    }));

    const uniqueDocs = new Set(evidenceItems.map((e) => e.file_name)).size;
    const traceSummaryText =
      evidenceItems.length > 0
        ? `Grounded in ${evidenceItems.length} ${evidenceItems.length === 1 ? "passage" : "passages"} across ${uniqueDocs} ${uniqueDocs === 1 ? "document" : "documents"} · ${elapsedSec}s`
        : `Completed grounded check · ${elapsedSec}s`;

    const sourceChipsHtml = evidenceItems
      .map((ev, idx) => {
        const pg = ev.page_number ? ` p.${ev.page_number}` : "";
        return `
          <button type="button" class="source-chip-btn" data-source-index="${idx}">
            <span>[Doc ${idx + 1}${pg}]</span>
            <span>${escapeHtml(ev.file_name)}</span>
          </button>`;
      })
      .join("");

    assistantWrap.innerHTML = `
      <details class="agent-trace">
        <summary class="trace-summary">
          <div class="trace-left">
            <span class="trace-pulse"></span>
            <span class="trace-title">${escapeHtml(traceSummaryText)}</span>
          </div>
          <span>▾</span>
        </summary>
        <div class="trace-steps">
          <div class="trace-step"><span class="trace-step-ok">✓</span> Semantic search retrieved ${evidenceItems.length} evidence chunk(s) (Top-K = ${topK})</div>
          <div class="trace-step"><span class="trace-step-ok">✓</span> Generator: ${escapeHtml(provider)} (${escapeHtml(model)})</div>
          <div class="trace-step"><span class="trace-step-ok">✓</span> Citations mapped in ${elapsedSec}s</div>
        </div>
      </details>
      <div class="assistant-prose" style="margin-top: 14px;">
        ${renderEditorialMarkdown(answerText, evidenceItems)}
      </div>
      <div class="answer-footer">
        ${sourceChipsHtml}
        <button type="button" class="ghost-action-btn copy-btn">Copy Answer</button>
      </div>
    `;

    assistantWrap.querySelectorAll(".cite-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        const idx = parseInt(pill.getAttribute("data-cite-index"), 10);
        openDrawer(evidenceItems, idx);
      });
    });

    assistantWrap.querySelectorAll(".source-chip-btn").forEach((chip) => {
      chip.addEventListener("click", () => {
        const idx = parseInt(chip.getAttribute("data-source-index"), 10);
        openDrawer(evidenceItems, idx);
      });
    });

    const copyBtn = assistantWrap.querySelector(".copy-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(answerText);
        copyBtn.textContent = "Copied";
        setTimeout(() => (copyBtn.textContent = "Copy Answer"), 1500);
      });
    }

    renderEvidenceDrawer(evidenceItems);
  }

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query) return;

    if (welcomeState) {
      welcomeState.style.display = "none";
    }

    chatInput.value = "";
    autoResizeTextarea();
    sendBtn.disabled = true;

    const turnEl = document.createElement("div");
    turnEl.className = "msg-turn";
    turnEl.innerHTML = `<div class="user-bubble">${escapeHtml(query)}</div>`;

    const topK = parseInt(topkSlider.value, 10) || 3;
    const generator = generatorSelect.value;
    const model = modelInput.value.trim() || "llama3.2:3b";

    const assistantWrap = document.createElement("div");
    assistantWrap.className = "assistant-card";
    assistantWrap.innerHTML = `
      <details class="agent-trace" open>
        <summary class="trace-summary">
          <div class="trace-left">
            <span class="trace-pulse running"></span>
            <span class="trace-title">Streaming grounded synthesis (Top-${topK})...</span>
          </div>
          <span>▾</span>
        </summary>
        <div class="trace-steps">
          <div class="trace-step"><span class="trace-step-ok">✓</span> Encoded query with SentenceTransformer</div>
          <div class="trace-step"><span class="trace-step-ok">✓</span> Searching vector index (Top-K = ${topK})</div>
          <div class="trace-step"><span>⋯</span> Streaming tokens from ${escapeHtml(generator)} (${escapeHtml(model)})</div>
        </div>
      </details>
      <div class="assistant-prose live-stream-prose" style="margin-top: 12px;"></div>
    `;

    turnEl.appendChild(assistantWrap);
    chatFeed.appendChild(turnEl);
    chatScroll.scrollTop = chatScroll.scrollHeight;

    const liveProseEl = assistantWrap.querySelector(".live-stream-prose");
    const requestBody = JSON.stringify({
      query,
      top_k: topK,
      generator,
      model,
    });

    const t0 = performance.now();

    try {
      // Try live SSE streaming endpoint first
      const streamRes = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestBody,
      });

      if (streamRes.ok && streamRes.body) {
        const reader = streamRes.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let streamedAnswer = "";
        let finalPayload = null;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const events = buffer.split("\n\n");
          buffer = events.pop() || "";

          for (const evt of events) {
            const dataLine = evt
              .split("\n")
              .find((line) => line.startsWith("data:"));
            if (!dataLine) continue;

            const jsonStr = dataLine.replace(/^data:\s*/, "").trim();
            if (!jsonStr) continue;

            const msg = JSON.parse(jsonStr);
            if (msg.type === "error") {
              throw new Error(msg.detail || "Streaming error");
            }
            if (msg.type === "token" && msg.token) {
              streamedAnswer += msg.token;
              liveProseEl.innerHTML = renderEditorialMarkdown(streamedAnswer, []);
              chatScroll.scrollTop = chatScroll.scrollHeight;
            } else if (msg.type === "done" || msg.done) {
              finalPayload = msg;
            }
          }
        }

        const elapsedSec = ((performance.now() - t0) / 1000).toFixed(2);
        const finalAnswer = (finalPayload && finalPayload.answer) || streamedAnswer || "No answer returned.";
        const rawEvidence = (finalPayload && (finalPayload.evidence || finalPayload.citations)) || [];
        const usedProvider = (finalPayload && finalPayload.provider) || generator;
        const usedModel = (finalPayload && finalPayload.model) || model;

        finalizeAssistantTurn(
          assistantWrap,
          finalAnswer,
          rawEvidence,
          usedProvider,
          usedModel,
          topK,
          elapsedSec
        );
        return;
      }

      // Automatic fallback to standard POST /api/chat if /api/chat/stream is not mounted yet
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestBody,
      });
      const data = await res.json();
      const elapsedSec = ((performance.now() - t0) / 1000).toFixed(2);

      if (!res.ok) {
        throw new Error(data.detail || "Failed to generate answer");
      }

      finalizeAssistantTurn(
        assistantWrap,
        data.answer || "No answer returned.",
        data.evidence || data.citations || [],
        data.provider || generator,
        data.model || model,
        topK,
        elapsedSec
      );
    } catch (err) {
      assistantWrap.innerHTML = `
        <div class="assistant-prose" style="color: var(--danger);">
          <strong>Error:</strong> ${escapeHtml(err.message)}
        </div>`;
    } finally {
      sendBtn.disabled = false;
      chatScroll.scrollTop = chatScroll.scrollHeight;
    }
  });

  // Initial load
  refreshStatus();
  refreshDocuments();
});