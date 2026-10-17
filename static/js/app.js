/* ============================================================
   刹车盘孔位检测分析系统 - 前端交互逻辑
   ============================================================ */

(function () {
  "use strict";

  // ---------- DOM 引用 ----------
  const uploadArea = document.getElementById("upload-area");
  const fileInput = document.getElementById("file-input");
  const pickFileBtn = document.getElementById("pick-file-btn");
  const imageStage = document.getElementById("image-stage");
  const previewImage = document.getElementById("preview-image");
  const imageLoader = document.getElementById("image-loader");
  const imageMeta = document.getElementById("image-meta");
  const analyzeBtn = document.getElementById("analyze-btn");
  const drawAllCheckbox = document.getElementById("draw-all");
  const paramsGrid = document.getElementById("params-grid");
  const resetParamsBtn = document.getElementById("reset-params");
  const resultsEmpty = document.getElementById("results-empty");
  const resultsContent = document.getElementById("results-content");
  const downloadImgBtn = document.getElementById("download-img");
  const downloadJsonBtn = document.getElementById("download-json");
  const statusBadge = document.getElementById("status-badge");

  const toggleBtns = document.querySelectorAll(".toggle-btn");
  const batchGallery = document.getElementById("batch-gallery");
  const compareOverlay = document.getElementById("compare-overlay");
  const compareAnnotated = document.getElementById("compare-annotated");
  const compareSlider = document.getElementById("compare-slider");
  const stepsGallery = document.getElementById("steps-gallery");
  const zoomToolbar = document.getElementById("zoom-toolbar");
  const zoomLevelEl = document.getElementById("zoom-level");

  // ---------- 状态 ----------
  let currentFile = null;
  let originalDataUrl = null;
  let lastResult = null;
  let defaultConfig = {};
  let currentConfig = {};
  let currentView = "annotated";
  let batchFiles = []; // [{file, dataUrl, result, status}]
  let activeBatchIndex = 0;

  // 缩放状态
  let zoomScale = 1;
  let zoomPanX = 0;
  let zoomPanY = 0;
  let isPanning = false;
  let panStartX = 0;
  let panStartY = 0;

  // ---------- 参数定义 ----------
  const PARAMS = [
    { key: "min_circularity", label: "最小圆度", type: "range", min: 0.3, max: 0.95, step: 0.05 },
    { key: "min_aspect_ratio", label: "最小长宽比", type: "range", min: 0.5, max: 1.0, step: 0.05 },
    { key: "hough_param2", label: "Hough 阈值", type: "range", min: 10, max: 60, step: 1 },
    { key: "hough_min_radius", label: "Hough 最小半径", type: "number", min: 0, max: 100 },
    { key: "hough_max_radius", label: "Hough 最大半径", type: "number", min: 0, max: 200 },
    { key: "min_bolt_radius_ratio", label: "螺栓孔最小半径比", type: "range", min: 0.01, max: 0.08, step: 0.005 },
    { key: "max_bolt_radius_ratio", label: "螺栓孔最大半径比", type: "range", min: 0.02, max: 0.15, step: 0.005 },
    { key: "pcd_tolerance", label: "PCD 容差", type: "range", min: 0.05, max: 0.3, step: 0.01 },
    { key: "use_hough", label: "启用 Hough", type: "checkbox" },
    { key: "use_blob", label: "启用 Blob", type: "checkbox" },
  ];

  const STEP_LABELS = {
    grayscale: "灰度图",
    blurred: "高斯模糊",
    binary: "二值化",
    edges: "边缘检测",
    candidates: "候选圆",
    final: "最终结果",
  };

  // ---------- 初始化 ----------
  async function init() {
    await loadDefaultConfig();
    buildParamsUI();
    bindEvents();
  }

  async function loadDefaultConfig() {
    try {
      const res = await fetch("/api/config");
      defaultConfig = await res.json();
      currentConfig = { ...defaultConfig };
    } catch (e) {
      console.error("加载默认配置失败", e);
      showToast("无法连接到后端服务", "error");
    }
  }

  // ---------- 参数面板 ----------
  function buildParamsUI() {
    paramsGrid.innerHTML = "";
    PARAMS.forEach((p) => {
      const field = document.createElement("div");
      field.className = "param-field";

      const label = document.createElement("label");
      label.innerHTML = `<span>${p.label}</span>`;

      let input;
      if (p.type === "checkbox") {
        input = document.createElement("input");
        input.type = "checkbox";
        input.checked = !!currentConfig[p.key];
        input.addEventListener("change", () => {
          currentConfig[p.key] = input.checked;
        });
        field.appendChild(label);
        field.appendChild(input);
      } else if (p.type === "range") {
        const valSpan = document.createElement("span");
        valSpan.className = "val";
        valSpan.textContent = currentConfig[p.key];
        label.appendChild(valSpan);

        input = document.createElement("input");
        input.type = "range";
        input.min = p.min;
        input.max = p.max;
        input.step = p.step;
        input.value = currentConfig[p.key];
        input.addEventListener("input", () => {
          const v = parseFloat(input.value);
          currentConfig[p.key] = v;
          valSpan.textContent = v;
        });
        field.appendChild(label);
        field.appendChild(input);
      } else {
        input = document.createElement("input");
        input.type = "number";
        input.min = p.min || "";
        input.max = p.max || "";
        input.value = currentConfig[p.key];
        input.addEventListener("change", () => {
          currentConfig[p.key] = parseFloat(input.value);
        });
        field.appendChild(label);
        field.appendChild(input);
      }

      paramsGrid.appendChild(field);
    });
  }

  function resetParams() {
    currentConfig = { ...defaultConfig };
    buildParamsUI();
    showToast("参数已重置", "success");
  }

  // ---------- 事件绑定 ----------
  function bindEvents() {
    pickFileBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });

    uploadArea.addEventListener("click", () => fileInput.click());

    uploadArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", () => {
      uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadArea.classList.remove("dragover");
      const files = Array.from(e.dataTransfer.files);
      if (files.length) handleFiles(files);
    });

    fileInput.addEventListener("change", (e) => {
      const files = Array.from(e.target.files);
      if (files.length) handleFiles(files);
    });

    analyzeBtn.addEventListener("click", analyze);

    resetParamsBtn.addEventListener("click", resetParams);

    toggleBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        toggleBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        setView(btn.dataset.view);
      });
    });

    downloadImgBtn.addEventListener("click", downloadAnnotated);
    downloadJsonBtn.addEventListener("click", downloadJson);

    // 缩放控制
    document.getElementById("zoom-in").addEventListener("click", () => setZoom(zoomScale * 1.2));
    document.getElementById("zoom-out").addEventListener("click", () => setZoom(zoomScale / 1.2));
    document.getElementById("zoom-reset").addEventListener("click", resetZoom);

    // 滚轮缩放
    const container = document.getElementById("image-container");
    container.addEventListener("wheel", (e) => {
      if (currentView === "compare" || currentView === "steps") return;
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      setZoom(zoomScale * factor);
    }, { passive: false });

    // 拖拽平移
    container.addEventListener("mousedown", (e) => {
      if (currentView === "compare" || currentView === "steps") return;
      if (zoomScale <= 1) return;
      isPanning = true;
      panStartX = e.clientX - zoomPanX;
      panStartY = e.clientY - zoomPanY;
      container.style.cursor = "grabbing";
    });

    window.addEventListener("mousemove", (e) => {
      if (!isPanning) return;
      zoomPanX = e.clientX - panStartX;
      zoomPanY = e.clientY - panStartY;
      applyZoom();
    });

    window.addEventListener("mouseup", () => {
      isPanning = false;
      document.getElementById("image-container").style.cursor =
        zoomScale > 1 ? "grab" : "default";
    });

    // 对比滑块
    let isSliding = false;
    compareSlider.addEventListener("mousedown", (e) => {
      isSliding = true;
      e.preventDefault();
    });
    compareOverlay.addEventListener("click", (e) => {
      moveSlider(e);
    });
    window.addEventListener("mousemove", (e) => {
      if (isSliding) moveSlider(e);
    });
    window.addEventListener("mouseup", () => {
      isSliding = false;
    });
  }

  function moveSlider(e) {
    const rect = compareOverlay.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    compareSlider.style.left = x + "px";
    compareAnnotated.style.clipPath = `inset(0 ${rect.width - x}px 0 0)`;
  }

  // ---------- 缩放 ----------
  function setZoom(scale) {
    zoomScale = Math.max(0.5, Math.min(scale, 8));
    zoomLevelEl.textContent = Math.round(zoomScale * 100) + "%";
    applyZoom();
  }

  function resetZoom() {
    zoomScale = 1;
    zoomPanX = 0;
    zoomPanY = 0;
    zoomLevelEl.textContent = "100%";
    applyZoom();
  }

  function applyZoom() {
    previewImage.style.transform =
      `translate(${zoomPanX}px, ${zoomPanY}px) scale(${zoomScale})`;
  }

  // ---------- 文件处理（支持多文件） ----------
  function handleFiles(files) {
    const images = files.filter((f) => f.type.startsWith("image/"));
    if (!images.length) {
      showToast("请选择图像文件", "error");
      return;
    }

    batchFiles = [];
    activeBatchIndex = 0;

    const loadPromises = images.map((file) => {
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          batchFiles.push({
            file,
            dataUrl: e.target.result,
            result: null,
            status: "pending",
          });
          resolve();
        };
        reader.readAsDataURL(file);
      });
    });

    Promise.all(loadPromises).then(() => {
      renderBatchGallery();
      loadBatchItem(0);
    });
  }

  function renderBatchGallery() {
    batchGallery.innerHTML = "";
    if (batchFiles.length <= 1) {
      batchGallery.hidden = true;
      return;
    }
    batchGallery.hidden = false;
    batchFiles.forEach((item, idx) => {
      const thumb = document.createElement("div");
      thumb.className = "batch-thumb" + (idx === activeBatchIndex ? " active" : "") +
        (item.status === "done" ? " done" : item.status === "error" ? " error" : "");
      thumb.innerHTML = `
        <img src="${item.dataUrl}" alt="${item.file.name}" />
        <div class="thumb-status">${item.status === "done" ? "✓ 完成" : item.status === "error" ? "✗ 失败" : idx + 1}</div>
      `;
      thumb.addEventListener("click", () => loadBatchItem(idx));
      batchGallery.appendChild(thumb);
    });
  }

  function loadBatchItem(idx) {
    if (idx < 0 || idx >= batchFiles.length) return;
    activeBatchIndex = idx;
    const item = batchFiles[idx];
    currentFile = item.file;
    originalDataUrl = item.dataUrl;

    uploadArea.hidden = true;
    imageStage.hidden = false;
    analyzeBtn.disabled = false;
    setStatus("已加载图像", "");
    imageMeta.textContent = `${item.file.name} · ${(item.file.size / 1024).toFixed(1)} KB`;
    resetZoom();
    previewImage.src = originalDataUrl;
    stepsGallery.hidden = true;
    stepsGallery.innerHTML = "";
    compareOverlay.hidden = true;

    if (item.result) {
      lastResult = item.result;
      previewImage.src = item.result.annotated;
      updateResults(item.result.result);
      renderSteps(item.result.steps);
    } else {
      resultsEmpty.hidden = false;
      resultsContent.hidden = true;
      lastResult = null;
    }

    renderBatchGallery();
  }

  // ---------- 分析 ----------
  async function analyze() {
    if (!currentFile) return;

    setStatus("分析中...", "busy");
    imageLoader.hidden = false;
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", currentFile);
    formData.append("config", JSON.stringify(currentConfig));
    formData.append("draw_all", drawAllCheckbox.checked ? "true" : "false");
    formData.append("steps", "true");

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok || !data.ok) {
        throw new Error(data.error || "分析失败");
      }

      lastResult = data;
      batchFiles[activeBatchIndex].result = data;
      batchFiles[activeBatchIndex].status = "done";

      previewImage.src = data.annotated;
      updateResults(data.result);
      renderSteps(data.steps);
      setStatus("分析完成", "success");
      showToast("分析完成", "success");
      renderBatchGallery();
    } catch (e) {
      console.error(e);
      batchFiles[activeBatchIndex].status = "error";
      setStatus("分析失败", "error");
      showToast(e.message || "分析失败", "error");
      renderBatchGallery();
    } finally {
      imageLoader.hidden = true;
      analyzeBtn.disabled = false;
    }
  }

  // ---------- 视图切换 ----------
  function setView(view) {
    currentView = view;
    const container = document.getElementById("image-container");

    // 隐藏所有特殊层
    compareOverlay.hidden = true;
    stepsGallery.hidden = true;
    container.style.display = "flex";
    zoomToolbar.hidden = true;

    if (view === "annotated") {
      previewImage.src = lastResult ? lastResult.annotated : originalDataUrl;
      zoomToolbar.hidden = false;
    } else if (view === "original") {
      previewImage.src = originalDataUrl;
      zoomToolbar.hidden = false;
    } else if (view === "compare") {
      if (!lastResult) {
        showToast("请先完成分析", "error");
        // 回退到 annotated
        toggleBtns.forEach((b) => b.classList.remove("active"));
        document.querySelector('[data-view="annotated"]').classList.add("active");
        setView("annotated");
        return;
      }
      previewImage.src = originalDataUrl;
      compareAnnotated.src = lastResult.annotated;
      compareOverlay.hidden = false;
      // 初始化滑块位置到中间
      requestAnimationFrame(() => {
        const rect = compareOverlay.getBoundingClientRect();
        const mid = rect.width / 2;
        compareSlider.style.left = mid + "px";
        compareAnnotated.style.clipPath = `inset(0 ${mid}px 0 0)`;
      });
    } else if (view === "steps") {
      if (!lastResult || !lastResult.steps) {
        showToast("请先完成分析", "error");
        toggleBtns.forEach((b) => b.classList.remove("active"));
        document.querySelector('[data-view="annotated"]').classList.add("active");
        setView("annotated");
        return;
      }
      container.style.display = "none";
      stepsGallery.hidden = false;
    }

    resetZoom();
  }

  // ---------- 检测过程画廊 ----------
  function renderSteps(steps) {
    if (!steps) return;
    stepsGallery.innerHTML = "";
    const order = ["grayscale", "blurred", "binary", "edges", "candidates", "final"];
    order.forEach((key) => {
      if (!steps[key]) return;
      const card = document.createElement("div");
      card.className = "step-card";
      card.innerHTML = `
        <div class="step-img-wrap"><img src="${steps[key]}" alt="${key}" /></div>
        <div class="step-label">${STEP_LABELS[key] || key}</div>
      `;
      card.addEventListener("click", () => {
        stepsGallery.querySelectorAll(".step-card").forEach((c) => c.classList.remove("active"));
        card.classList.add("active");
        previewImage.src = steps[key];
        currentView = "annotated";
        document.getElementById("image-container").style.display = "flex";
        stepsGallery.hidden = true;
        zoomToolbar.hidden = false;
        // 切回 annotated 按钮
        toggleBtns.forEach((b) => b.classList.remove("active"));
        document.querySelector('[data-view="annotated"]').classList.add("active");
        resetZoom();
      });
      stepsGallery.appendChild(card);
    });
  }

  // ---------- 结果展示 ----------
  function updateResults(r) {
    resultsEmpty.hidden = true;
    resultsContent.hidden = false;

    setText("m-od", r.outer_diameter ? (r.outer_diameter.radius * 2).toFixed(1) : "—");
    setText("m-bore", r.central_bore ? (r.central_bore.radius * 2).toFixed(1) : "—");
    setText("m-bolt-count", r.num_bolt_holes);
    setText("m-pcd", r.pcd_diameter ? r.pcd_diameter.toFixed(1) : "—");

    const tbody = document.getElementById("holes-tbody");
    tbody.innerHTML = "";
    (r.bolt_holes || []).forEach((h, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>#${i + 1}</td>
        <td>${h.center[0].toFixed(1)}</td>
        <td>${h.center[1].toFixed(1)}</td>
        <td>${h.radius.toFixed(1)}</td>
        <td>${(h.radius * 2).toFixed(1)}</td>
      `;
      tbody.appendChild(tr);
    });

    setText("s-avg-dia", r.bolt_hole_avg_diameter ? r.bolt_hole_avg_diameter.toFixed(1) : "—");
    setText("s-pcd-std", r.pcd_std ? r.pcd_std.toFixed(2) : "—");
    const pc = r.pcd_center;
    setText("s-pcd-center", pc ? `(${pc[0].toFixed(1)}, ${pc[1].toFixed(1)})` : "—");
  }

  function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  // ---------- 下载 ----------
  function downloadAnnotated() {
    if (!lastResult) return;
    const a = document.createElement("a");
    a.href = lastResult.annotated;
    a.download = "brake_disc_annotated.jpg";
    a.click();
  }

  function downloadJson() {
    if (!lastResult) return;
    const blob = new Blob([JSON.stringify(lastResult.result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "brake_disc_result.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  // ---------- 工具 ----------
  function setStatus(text, type) {
    statusBadge.textContent = text;
    statusBadge.className = "badge" + (type ? " " + type : "");
  }

  function showToast(msg, type) {
    let toast = document.querySelector(".toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.className = "toast";
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.className = "toast " + (type || "");
    requestAnimationFrame(() => toast.classList.add("show"));
    clearTimeout(toast._t);
    toast._t = setTimeout(() => toast.classList.remove("show"), 2500);
  }

  // ---------- 启动 ----------
  init();
})();
