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

  // ---------- 状态 ----------
  let currentFile = null;
  let originalDataUrl = null;
  let lastResult = null;
  let defaultConfig = {};
  let currentConfig = {};

  // ---------- 参数定义（与 DetectorConfig 对应） ----------
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
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    });

    fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) handleFile(file);
    });

    analyzeBtn.addEventListener("click", analyze);

    resetParamsBtn.addEventListener("click", resetParams);

    toggleBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        toggleBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        updateImageView(btn.dataset.view);
      });
    });

    downloadImgBtn.addEventListener("click", downloadAnnotated);
    downloadJsonBtn.addEventListener("click", downloadJson);
  }

  // ---------- 文件处理 ----------
  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      showToast("请选择图像文件", "error");
      return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      originalDataUrl = e.target.result;
      previewImage.src = originalDataUrl;
      imageStage.hidden = false;
      uploadArea.hidden = true;
      analyzeBtn.disabled = false;
      setStatus("已加载图像", "");
      imageMeta.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
      // 重置结果区
      resultsEmpty.hidden = false;
      resultsContent.hidden = true;
      lastResult = null;
    };
    reader.readAsDataURL(file);
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

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok || !data.ok) {
        throw new Error(data.error || "分析失败");
      }

      lastResult = data;
      previewImage.src = data.annotated;
      updateResults(data.result);
      setStatus("分析完成", "success");
      showToast("分析完成", "success");
    } catch (e) {
      console.error(e);
      setStatus("分析失败", "error");
      showToast(e.message || "分析失败", "error");
    } finally {
      imageLoader.hidden = true;
      analyzeBtn.disabled = false;
    }
  }

  // ---------- 结果展示 ----------
  function updateResults(r) {
    resultsEmpty.hidden = true;
    resultsContent.hidden = false;

    // 指标卡片
    setText("m-od", r.outer_diameter ? r.outer_diameter.radius * 2 : "—");
    setText("m-bore", r.central_bore ? r.central_bore.radius * 2 : "—");
    setText("m-bolt-count", r.num_bolt_holes);
    setText("m-pcd", r.pcd_diameter ? r.pcd_diameter.toFixed(1) : "—");

    // 螺栓孔表格
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

    // 统计
    setText("s-avg-dia", r.bolt_hole_avg_diameter ? r.bolt_hole_avg_diameter.toFixed(1) : "—");
    setText("s-pcd-std", r.pcd_std ? r.pcd_std.toFixed(2) : "—");
    const pc = r.pcd_center;
    setText("s-pcd-center", pc ? `(${pc[0].toFixed(1)}, ${pc[1].toFixed(1)})` : "—");
  }

  function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  // ---------- 视图切换 ----------
  function updateImageView(view) {
    if (!lastResult) {
      if (originalDataUrl) previewImage.src = originalDataUrl;
      return;
    }
    if (view === "annotated") {
      previewImage.src = lastResult.annotated;
    } else if (view === "original") {
      previewImage.src = originalDataUrl;
    } else if (view === "compare") {
      // 简单拼接对比
      previewImage.src = lastResult.annotated;
      showToast("对比视图：请在标注与原始之间切换查看", "success");
    }
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
