// NeuroScan AI - Interactive Brain Tumor Radiomics Platform
document.addEventListener("DOMContentLoaded", () => {
  // Application State
  const state = {
    currentFile: null,
    currentSampleId: null,
    currentResult: null,
    splitActive: true,
    showBbox: true,
    autoContrast: false,
    opacity: 0.7,
    colormap: "jet",
    smartMode: true,
    soundEnabled: true,
    samples: []
  };

  // DOM Elements
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const emptyState = document.getElementById("empty-state");
  const mriStage = document.getElementById("mri-stage");
  const baseMriImg = document.getElementById("base-mri-img");
  const heatmapOverlayImg = document.getElementById("heatmap-overlay-img");
  const splitOverlayContainer = document.getElementById("split-overlay-container");
  const splitSliderDivider = document.getElementById("split-slider-divider");
  const roiBbox = document.getElementById("roi-bbox");
  const bboxTag = document.getElementById("bbox-tag");
  const bboxArea = document.getElementById("bbox-area");

  const laserScanner = document.getElementById("laser-scanner");
  const telemetryStep = document.getElementById("telemetry-step");
  const telemetrySub = document.getElementById("telemetry-sub");
  const telemetryProgress = document.getElementById("telemetry-progress");

  const opacitySlider = document.getElementById("opacity-slider");
  const opacityVal = document.getElementById("opacity-val");
  const colormapSelect = document.getElementById("colormap-select");
  const btnToggleSplit = document.getElementById("btn-toggle-split");
  const btnToggleBbox = document.getElementById("btn-toggle-bbox");
  const btnToggleContrast = document.getElementById("btn-toggle-contrast");
  const modeToggle = document.getElementById("mode-toggle");
  const soundBtn = document.getElementById("sound-btn");
  const soundIcon = document.getElementById("sound-icon");

  const sampleCardsGrid = document.getElementById("sample-cards-grid");

  // Findings Elements
  const diagnosisBanner = document.getElementById("diagnosis-banner");
  const diagnosisCategory = document.getElementById("diagnosis-category");
  const severityBadge = document.getElementById("severity-badge");
  const predictedClassTitle = document.getElementById("predicted-class-title");
  const confidenceCircleBar = document.getElementById("confidence-circle-bar");
  const confidencePercent = document.getElementById("confidence-percent");
  const detectionStatusText = document.getElementById("detection-status-text");
  const probabilityBarsList = document.getElementById("probability-bars-list");

  const valCentroid = document.getElementById("val-centroid");
  const valArea = document.getElementById("val-area");
  const valLatency = document.getElementById("val-latency");
  const valLocation = document.getElementById("val-location");

  const badgeUrgency = document.getElementById("badge-urgency");
  const clinicalNotes = document.getElementById("clinical-notes");
  const clinicalFollowup = document.getElementById("clinical-followup");

  const patientMrn = document.getElementById("patient-mrn");
  const patientName = document.getElementById("patient-name");
  const patientAgeSex = document.getElementById("patient-age-sex");
  const patientScanType = document.getElementById("patient-scan-type");
  const patientDate = document.getElementById("patient-date");
  const patientHistory = document.getElementById("patient-history");

  const btnExportReport = document.getElementById("btn-export-report");
  const reportModal = document.getElementById("report-modal");
  const btnCloseModal = document.getElementById("btn-close-modal");
  const btnPrintReport = document.getElementById("btn-print-report");

  // Audio Synthesizer (Web Audio API)
  let audioCtx = null;
  function playAudioTone(freq, duration = 0.08, type = "sine") {
    if (!state.soundEnabled) return;
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === "suspended") {
        audioCtx.resume();
      }
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
      gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
      // Audio might be blocked by browser policy until interaction
    }
  }

  function playSuccessChime() {
    if (!state.soundEnabled) return;
    setTimeout(() => playAudioTone(523.25, 0.12), 0);
    setTimeout(() => playAudioTone(659.25, 0.12), 100);
    setTimeout(() => playAudioTone(783.99, 0.22), 200);
  }

  // Fetch Samples on Startup
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      if (!res.ok) throw new Error("Failed to load samples");
      state.samples = await res.json();
      renderSampleCards();
      // Auto-load first sample for immediate wow factor!
      if (state.samples.length > 0) {
        selectSample(state.samples[0].id);
      }
    } catch (err) {
      console.error(err);
    }
  }

  function renderSampleCards() {
    sampleCardsGrid.innerHTML = state.samples.map(s => `
      <div class="sample-card ${state.currentSampleId === s.id ? 'active' : ''}" data-id="${s.id}">
        <div class="sample-thumb-wrapper">
          <img src="${s.image_url}" alt="${s.title}">
        </div>
        <div class="sample-meta">
          <span class="sample-type">${s.type}</span>
          <span class="sample-sub">${s.title}</span>
        </div>
      </div>
    `).join("");

    sampleCardsGrid.querySelectorAll(".sample-card").forEach(card => {
      card.addEventListener("click", () => {
        const id = card.dataset.id;
        selectSample(id);
      });
    });
  }

  function selectSample(id) {
    state.currentSampleId = id;
    state.currentFile = null;
    renderSampleCards();

    const sample = state.samples.find(s => s.id === id);
    if (!sample) return;

    // Show image immediately in stage
    baseMriImg.src = sample.image_url;
    emptyState.style.display = "none";
    mriStage.classList.remove("hidden");

    // Execute holographic neural inference
    runInference({ sampleId: id });
  }

  // File Upload Handlers
  browseBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUploadedFile(e.target.files[0]);
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUploadedFile(e.dataTransfer.files[0]);
    }
  });

  function handleUploadedFile(file) {
    state.currentFile = file;
    state.currentSampleId = null;
    renderSampleCards();

    const reader = new FileReader();
    reader.onload = (e) => {
      baseMriImg.src = e.target.result;
      emptyState.style.display = "none";
      mriStage.classList.remove("hidden");
      runInference({ file: state.currentFile });
    };
    reader.readAsDataURL(file);
  }

  // Holographic Scanning Animation & Model Execution
  async function runInference(options = {}) {
    laserScanner.classList.add("active");
    btnExportReport.disabled = true;

    // Animation telemetry sequence
    const telemetrySteps = [
      { step: "VOXEL DATA ACQUISITION", sub: "NORMALIZING T1/T2 AXIAL DENSITIES...", prog: "20%", freq: 380 },
      { step: "DEEP FEATURE EXTRACTION", sub: "4x CONVOLUTION BLOCKS (conv2d_3)...", prog: "50%", freq: 440 },
      { step: "GRADIENT BACKPROPAGATION", sub: "COMPUTING ACTIVATION TENSOR FIELD...", prog: "75%", freq: 520 },
      { step: "RADIOMIC CLASSIFICATION", sub: "SYNTHESIZING GRAD-CAM HEATMAP...", prog: "95%", freq: 620 }
    ];

    let stepIdx = 0;
    const interval = setInterval(() => {
      if (stepIdx < telemetrySteps.length) {
        const item = telemetrySteps[stepIdx];
        telemetryStep.textContent = item.step;
        telemetrySub.textContent = item.sub;
        telemetryProgress.style.width = item.prog;
        playAudioTone(item.freq, 0.05, "triangle");
        stepIdx++;
      }
    }, 280);

    const formData = new FormData();
    if (options.file) {
      formData.append("file", options.file);
    } else if (options.sampleId) {
      formData.append("sample_id", options.sampleId);
    }
    formData.append("colormap", state.colormap);
    formData.append("enhance_contrast", state.autoContrast);
    formData.append("mode", state.smartMode ? "smart" : "raw");

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        throw new Error("Inference failed with status " + res.status);
      }

      const result = await res.json();
      clearInterval(interval);

      // Finish telemetry bar
      telemetryStep.textContent = "DIAGNOSTIC PIPELINE COMPLETE";
      telemetrySub.textContent = "ACTIVATION MAP GENERATED";
      telemetryProgress.style.width = "100%";

      setTimeout(() => {
        laserScanner.classList.remove("active");
        playSuccessChime();
        applyInferenceResult(result);
      }, 350);

    } catch (err) {
      clearInterval(interval);
      laserScanner.classList.remove("active");
      alert("Inference error: " + err.message);
      console.error(err);
    }
  }

  // Update UI with Model Results
  function applyInferenceResult(result) {
    state.currentResult = result;
    btnExportReport.disabled = false;

    // Heatmap Overlay
    heatmapOverlayImg.src = result.overlay_data_uri;
    heatmapOverlayImg.style.opacity = state.opacity;

    // Predicted Class & Confidence
    predictedClassTitle.textContent = result.predicted_class;
    confidencePercent.textContent = `${result.confidence.toFixed(1)}%`;
    confidenceCircleBar.setAttribute("stroke-dasharray", `${result.confidence}, 100`);

    // Banner & Badges Styling based on Pathology
    diagnosisBanner.className = "diagnosis-banner";
    severityBadge.className = "severity-badge";

    if (result.predicted_class === "Glioma") {
      diagnosisBanner.classList.add("severity-critical");
      severityBadge.classList.add("critical");
      severityBadge.textContent = "CRITICAL LESION";
      confidenceCircleBar.style.stroke = "var(--crimson)";
    } else if (result.predicted_class === "Meningioma" || result.predicted_class === "Pituitary") {
      diagnosisBanner.classList.add("severity-elevated");
      severityBadge.classList.add("elevated");
      severityBadge.textContent = "NEOPLASM IDENTIFIED";
      confidenceCircleBar.style.stroke = "var(--amber)";
    } else {
      diagnosisBanner.classList.add("severity-normal");
      severityBadge.classList.add("normal");
      severityBadge.textContent = "BENIGN / NORMAL";
      confidenceCircleBar.style.stroke = "var(--emerald)";
    }

    diagnosisCategory.textContent = result.clinical_details.category.toUpperCase();
    detectionStatusText.textContent = result.is_tumor_detected 
      ? `Focal hyperintense abnormality detected with ${result.confidence}% confidence.`
      : `No pathological mass or structural deviation identified.`;

    // Probability Spectrum Bars
    probabilityBarsList.innerHTML = result.probabilities.map(p => {
      const isTop = (p.class_name === result.predicted_class);
      return `
        <div class="prob-row ${isTop ? 'highest' : ''}" data-class="${p.class_name}">
          <div class="prob-meta">
            <span class="class-name">${p.class_name === 'No Tumor' ? 'No Tumor (Healthy)' : p.class_name}</span>
            <span class="class-pct">${p.percentage.toFixed(1)}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width: ${p.percentage}%;"></div>
          </div>
        </div>
      `;
    }).join("");

    // Spatial & Radiomics Metrics
    if (result.roi_bounding_box) {
      valCentroid.textContent = `(${result.roi_bounding_box.centroid_x_pct}%, ${result.roi_bounding_box.centroid_y_pct}%)`;
      valArea.textContent = `${result.roi_bounding_box.lesion_area_pct}% cranial area`;
    } else {
      valCentroid.textContent = "None (Diffuse)";
      valArea.textContent = "0.0%";
    }
    valLatency.textContent = `${result.inference_time_ms} ms`;
    valLocation.textContent = result.clinical_details.anatomical_location;

    // Recommendations
    badgeUrgency.textContent = result.clinical_details.severity.toUpperCase();
    clinicalNotes.textContent = result.clinical_details.pathology_notes;
    clinicalFollowup.textContent = result.clinical_details.recommended_followup;

    // Patient Metadata
    const p = result.patient_metadata || {};
    patientName.textContent = p.name || "Anonymous Patient";
    patientMrn.textContent = p.mrn || "MRN-PENDING";
    patientAgeSex.textContent = `${p.age || 'N/A'} / ${p.gender || 'N/A'}`;
    patientScanType.textContent = p.scan_type || "Axial T1 MRI";
    patientDate.textContent = p.study_date || "Today";
    patientHistory.textContent = result.clinical_history || "No prior clinical history recorded.";

    // Bounding Box Update
    updateBoundingBox(result.roi_bounding_box);
  }

  function updateBoundingBox(roi) {
    if (!roi || !state.showBbox || !state.currentResult || !state.currentResult.is_tumor_detected) {
      roiBbox.classList.add("hidden");
      return;
    }

    roiBbox.classList.remove("hidden");
    roiBbox.style.top = `${roi.ymin_pct}%`;
    roiBbox.style.left = `${roi.xmin_pct}%`;
    roiBbox.style.width = `${roi.width_pct}%`;
    roiBbox.style.height = `${roi.height_pct}%`;

    bboxTag.textContent = `${state.currentResult.predicted_class.toUpperCase()} ROI`;
    bboxArea.textContent = `${roi.lesion_area_pct}%`;
  }

  // Split-Screen Interactive Slider Logic
  let isDraggingSplit = false;

  function setSplitPosition(percent) {
    const clamped = Math.max(0, Math.min(100, percent));
    splitOverlayContainer.style.setProperty("--split-pos", `${clamped}%`);
    splitSliderDivider.style.left = `${clamped}%`;
  }

  splitSliderDivider.addEventListener("mousedown", (e) => {
    e.preventDefault();
    isDraggingSplit = true;
  });

  window.addEventListener("mouseup", () => {
    isDraggingSplit = false;
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDraggingSplit || !state.splitActive) return;
    const rect = mriStage.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = (x / rect.width) * 100;
    setSplitPosition(pct);
  });

  // Touch support for split slider
  splitSliderDivider.addEventListener("touchstart", (e) => {
    isDraggingSplit = true;
  }, { passive: true });

  window.addEventListener("touchend", () => {
    isDraggingSplit = false;
  });

  window.addEventListener("touchmove", (e) => {
    if (!isDraggingSplit || !state.splitActive || !e.touches[0]) return;
    const rect = mriStage.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    const pct = (x / rect.width) * 100;
    setSplitPosition(pct);
  }, { passive: true });

  // Toolbar Controls
  opacitySlider.addEventListener("input", (e) => {
    state.opacity = e.target.value / 100;
    opacityVal.textContent = `${e.target.value}%`;
    heatmapOverlayImg.style.opacity = state.opacity;
  });

  colormapSelect.addEventListener("change", (e) => {
    state.colormap = e.target.value;
    if (state.currentResult) {
      // Re-run inference or update with selected colormap
      if (state.currentFile) {
        runInference({ file: state.currentFile });
      } else if (state.currentSampleId) {
        runInference({ sampleId: state.currentSampleId });
      }
    }
  });

  btnToggleSplit.addEventListener("click", () => {
    state.splitActive = !state.splitActive;
    btnToggleSplit.classList.toggle("active", state.splitActive);
    if (state.splitActive) {
      splitOverlayContainer.classList.remove("no-split");
      splitSliderDivider.classList.remove("hidden");
      setSplitPosition(50);
    } else {
      splitOverlayContainer.classList.add("no-split");
      splitSliderDivider.classList.add("hidden");
    }
  });

  btnToggleBbox.addEventListener("click", () => {
    state.showBbox = !state.showBbox;
    btnToggleBbox.classList.toggle("active", state.showBbox);
    if (state.currentResult) {
      updateBoundingBox(state.currentResult.roi_bounding_box);
    }
  });

  btnToggleContrast.addEventListener("click", () => {
    state.autoContrast = !state.autoContrast;
    btnToggleContrast.classList.toggle("active", state.autoContrast);
    if (state.currentResult) {
      if (state.currentFile) {
        runInference({ file: state.currentFile });
      } else if (state.currentSampleId) {
        runInference({ sampleId: state.currentSampleId });
      }
    }
  });

  modeToggle.addEventListener("change", (e) => {
    state.smartMode = e.target.checked;
    if (state.currentResult) {
      if (state.currentFile) {
        runInference({ file: state.currentFile });
      } else if (state.currentSampleId) {
        runInference({ sampleId: state.currentSampleId });
      }
    }
  });

  soundBtn.addEventListener("click", () => {
    state.soundEnabled = !state.soundEnabled;
    soundIcon.textContent = state.soundEnabled ? "🔊" : "🔇";
  });

  // Clinical Report Modal & Export
  btnExportReport.addEventListener("click", () => {
    if (!state.currentResult) return;
    populateReportModal(state.currentResult);
    reportModal.classList.remove("hidden");
  });

  btnCloseModal.addEventListener("click", () => {
    reportModal.classList.add("hidden");
  });

  btnPrintReport.addEventListener("click", () => {
    window.print();
  });

  function populateReportModal(res) {
    const p = res.patient_metadata || {};
    document.getElementById("rep-id").textContent = `REP-${Date.now().toString().slice(-6)}`;
    document.getElementById("rep-date").textContent = new Date().toISOString().split("T")[0];
    document.getElementById("rep-patient-name").textContent = p.name || "Anonymous Patient";
    document.getElementById("rep-mrn").textContent = p.mrn || "MRN-PENDING";
    document.getElementById("rep-age-sex").textContent = `${p.age || 'N/A'} / ${p.gender || 'N/A'}`;
    document.getElementById("rep-modality").textContent = p.scan_type || "Axial Brain MRI";

    document.getElementById("rep-orig-img").src = baseMriImg.src;
    document.getElementById("rep-cam-img").src = res.overlay_data_uri;

    document.getElementById("rep-primary-finding").textContent = res.predicted_class;
    document.getElementById("rep-confidence").textContent = `${res.confidence.toFixed(1)}%`;
    document.getElementById("rep-category").textContent = res.clinical_details.category;
    document.getElementById("rep-urgency").textContent = res.clinical_details.urgency;
    document.getElementById("rep-location").textContent = res.clinical_details.anatomical_location;
    document.getElementById("rep-footprint").textContent = res.roi_bounding_box 
      ? `${res.roi_bounding_box.lesion_area_pct}% of cranial slice`
      : "No localized mass footprint";

    document.getElementById("rep-notes").textContent = res.clinical_details.pathology_notes;
    document.getElementById("rep-recs").textContent = res.clinical_details.recommended_followup;
  }

  // Initialize
  setSplitPosition(50);
  loadSamples();
});
