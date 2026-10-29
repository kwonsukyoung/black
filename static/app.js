/**
 * 블랙박스 범인 잡기 - 프론트엔드 제어 및 하이브리드(Python OpenCV + 브라우저 Canvas) 분석 엔진
 */

const state = {
  engineMode: "browser", // "python" | "browser"
  selectedFiles: [],     // File 객체 배열
  objectUrls: [],        // 로컬 미리보기용 Blob URL 배열
  currentJobId: null,
  isAnalyzing: false,
  cancelRequested: false,
  pollTimer: null,

  // 기본 ROI: 왼쪽 뒷 필러/좌측 집중 프리셋
  roi: { x: 0.0, y: 0.12, w: 0.58, h: 0.80 },
  activePreset: "left_pillar",

  // 감지 결과 및 필터 상태
  events: [],
  activeFilter: "all",
  sortOrder: "score_desc",
  selectedEventId: null,

  // 비디오 줌 상태
  zoomLevel: 1,
  panX: 0,
  panY: 0,
};

const ROI_PRESETS = {
  left_pillar: { x: 0.0, y: 0.12, w: 0.58, h: 0.80, name: "좌측 집중 (왼쪽 뒷 필러/측면)" },
  full:        { x: 0.0, y: 0.06, w: 1.00, h: 0.88, name: "전체 화면 (시계 자막 제외)" },
  right_side:  { x: 0.42, y: 0.12, w: 0.58, h: 0.80, name: "우측 집중 (조수석 측면)" },
  rear_close:  { x: 0.08, y: 0.25, w: 0.84, h: 0.70, name: "후방/양측 근접 구역" },
};

// DOM 요소 참조
const els = {
  engineBadge: document.getElementById("engineBadge"),
  engineText: document.getElementById("engineText"),
  exportReportBtn: document.getElementById("exportReportBtn"),

  dropZone: document.getElementById("dropZone"),
  fileInput: document.getElementById("fileInput"),
  folderInput: document.getElementById("folderInput"),
  localFolderBox: document.getElementById("localFolderBox"),
  localFolderPath: document.getElementById("localFolderPath"),
  fileSummary: document.getElementById("fileSummary"),
  fileCountText: document.getElementById("fileCountText"),
  clearFilesBtn: document.getElementById("clearFilesBtn"),
  fileListPreview: document.getElementById("fileListPreview"),

  presetBtns: document.querySelectorAll(".preset-btn"),
  sensitivityRange: document.getElementById("sensitivityRange"),
  sensitivityVal: document.getElementById("sensitivityVal"),
  sampleFpsSelect: document.getElementById("sampleFpsSelect"),
  mergeGapSelect: document.getElementById("mergeGapSelect"),
  detectShakeCheck: document.getElementById("detectShakeCheck"),

  startAnalyzeBtn: document.getElementById("startAnalyzeBtn"),
  stopAnalyzeBtn: document.getElementById("stopAnalyzeBtn"),
  progressPanel: document.getElementById("progressPanel"),
  progressStatusText: document.getElementById("progressStatusText"),
  progressPctText: document.getElementById("progressPctText"),
  progressBarFill: document.getElementById("progressBarFill"),
  progressFileSub: document.getElementById("progressFileSub"),
  progressFoundSub: document.getElementById("progressFoundSub"),

  currentVideoTitle: document.getElementById("currentVideoTitle"),
  roiCoordsBadge: document.getElementById("roiCoordsBadge"),
  videoStageWrapper: document.getElementById("videoStageWrapper"),
  videoZoomContainer: document.getElementById("videoZoomContainer"),
  mainVideo: document.getElementById("mainVideo"),
  roiOverlayCanvas: document.getElementById("roiOverlayCanvas"),
  emptyVideoPlaceholder: document.getElementById("emptyVideoPlaceholder"),

  stepBack2sBtn: document.getElementById("stepBack2sBtn"),
  stepBackFrameBtn: document.getElementById("stepBackFrameBtn"),
  stepFwdFrameBtn: document.getElementById("stepFwdFrameBtn"),
  stepFwd2sBtn: document.getElementById("stepFwd2sBtn"),
  speedBtns: document.querySelectorAll(".speed-btn"),
  zoomBtns: document.querySelectorAll(".zoom-btn"),
  drawRoiModeCheck: document.getElementById("drawRoiModeCheck"),
  captureFrameBtn: document.getElementById("captureFrameBtn"),

  totalEventsBadge: document.getElementById("totalEventsBadge"),
  filterTabs: document.querySelectorAll(".filter-tab"),
  sortOrderSelect: document.getElementById("sortOrderSelect"),
  eventsGrid: document.getElementById("eventsGrid"),

  hiddenWorkerVideo: document.getElementById("hiddenWorkerVideo"),
  hiddenWorkerCanvas: document.getElementById("hiddenWorkerCanvas"),
};

// ============================================================================
// 초기화 및 백엔드 연결 확인
// ============================================================================
async function initApp() {
  await checkBackendEngine();
  bindEvents();
  updateRoiBadgeText();
  syncOverlayCanvas();
}

async function checkBackendEngine() {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    if (res.ok) {
      const data = await res.json();
      if (data.engine === "python-opencv") {
        state.engineMode = "python";
        els.engineBadge.className = "engine-badge online";
        els.engineText.textContent = "Python OpenCV 정밀 분석 엔진 연결됨";
        els.localFolderBox.classList.remove("hidden");
        return;
      }
    }
  } catch (_) {
    // 로컬 서버 없이 HTML을 직접 연 경우 브라우저 고속 엔진으로 자동 전환
  }
  state.engineMode = "browser";
  els.engineBadge.className = "engine-badge browser-mode";
  els.engineText.textContent = "브라우저 내장 고속 분석 모드 (설치 불필요)";
}

// ============================================================================
// 이벤트 바인딩
// ============================================================================
function bindEvents() {
  // 드래그 앤 드롭
  els.dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.dropZone.classList.add("dragover");
  });
  els.dropZone.addEventListener("dragleave", () => {
    els.dropZone.classList.remove("dragover");
  });
  els.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.dropZone.classList.remove("dragover");
    if (e.dataTransfer?.files?.length) {
      handleAddedFiles(Array.from(e.dataTransfer.files));
    }
  });

  els.fileInput.addEventListener("change", (e) => {
    if (e.target.files?.length) handleAddedFiles(Array.from(e.target.files));
  });
  els.folderInput.addEventListener("change", (e) => {
    if (e.target.files?.length) handleAddedFiles(Array.from(e.target.files));
  });

  els.localFolderPath.addEventListener("input", () => {
    updateStartButtonState();
  });

  els.clearFilesBtn.addEventListener("click", clearSelectedFiles);

  // ROI 프리셋 버튼
  els.presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.preset;
      const preset = ROI_PRESETS[key];
      if (!preset) return;
      state.activePreset = key;
      state.roi = { x: preset.x, y: preset.y, w: preset.w, h: preset.h };
      els.presetBtns.forEach((b) => b.classList.toggle("active", b === btn));
      updateRoiBadgeText(preset.name);
      drawRoiOverlay();
    });
  });

  // 민감도 슬라이더
  els.sensitivityRange.addEventListener("input", () => {
    const v = parseInt(els.sensitivityRange.value, 10);
    let label = "보통";
    if (v >= 78) label = "매우 민감 (미세 움직임)";
    else if (v >= 62) label = "권장 (차량/보행자/접촉)";
    else if (v < 45) label = "둔감 (큰 변화만)";
    els.sensitivityVal.textContent = `${v} (${label})`;
  });

  // 분석 시작 / 중지
  els.startAnalyzeBtn.addEventListener("click", startBatchAnalysis);
  els.stopAnalyzeBtn.addEventListener("click", stopBatchAnalysis);

  // 비디오 로드 및 리사이즈 시 ROI 캔버스 동기화
  els.mainVideo.addEventListener("loadedmetadata", () => {
    els.emptyVideoPlaceholder.classList.add("hidden");
    setTimeout(syncOverlayCanvas, 60);
  });
  window.addEventListener("resize", syncOverlayCanvas);

  // ROI 마우스 드래그 그리기
  initRoiCanvasDrag();

  // 비디오 판독 컨트롤 (프레임 이동, 배속, 줌, 캡처)
  els.stepBack2sBtn.addEventListener("click", () => seekRelative(-2.0));
  els.stepFwd2sBtn.addEventListener("click", () => seekRelative(2.0));
  els.stepBackFrameBtn.addEventListener("click", () => {
    els.mainVideo.pause();
    seekRelative(-1 / 30);
  });
  els.stepFwdFrameBtn.addEventListener("click", () => {
    els.mainVideo.pause();
    seekRelative(1 / 30);
  });

  els.speedBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const spd = parseFloat(btn.dataset.speed);
      els.mainVideo.playbackRate = spd;
      els.speedBtns.forEach((b) => b.classList.toggle("active", b === btn));
    });
  });

  els.zoomBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      state.zoomLevel = parseInt(btn.dataset.zoom, 10);
      if (state.zoomLevel === 1) {
        state.panX = 0;
        state.panY = 0;
      }
      applyVideoZoom();
      els.zoomBtns.forEach((b) => b.classList.toggle("active", b === btn));
    });
  });

  els.drawRoiModeCheck.addEventListener("change", () => {
    els.roiOverlayCanvas.style.pointerEvents = els.drawRoiModeCheck.checked ? "auto" : "none";
  });

  els.captureFrameBtn.addEventListener("click", captureCurrentVideoFrame);
  els.exportReportBtn.addEventListener("click", exportHtmlReport);

  // 필터 탭 및 정렬
  els.filterTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      state.activeFilter = tab.dataset.filter;
      els.filterTabs.forEach((t) => t.classList.toggle("active", t === tab));
      renderEventsGrid();
    });
  });

  els.sortOrderSelect.addEventListener("change", () => {
    state.sortOrder = els.sortOrderSelect.value;
    renderEventsGrid();
  });
}

// ============================================================================
// 파일 관리 및 미리보기
// ============================================================================
function handleAddedFiles(fileList) {
  const validExts = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"];
  const filtered = fileList.filter((f) => {
    const lower = f.name.toLowerCase();
    return f.type.startsWith("video/") || validExts.some((ext) => lower.endsWith(ext));
  });

  if (filtered.length === 0) {
    alert("지원하는 동영상 파일(MP4, AVI, MOV, MKV)을 선택해 주세요.");
    return;
  }

  // 이름순 정렬
  filtered.sort((a, b) => a.name.localeCompare(b.name));

  // 기존 URL 해제 후 재등록
  state.objectUrls.forEach((u) => URL.revokeObjectURL(u));
  state.selectedFiles = filtered;
  state.objectUrls = filtered.map((f) => URL.createObjectURL(f));

  renderFileListPreview();
  updateStartButtonState();

  // 첫 번째 영상 즉시 미리보기 로드
  loadVideoToMainPlayer(state.objectUrls[0], filtered[0].name, 0);
}

function clearSelectedFiles() {
  state.objectUrls.forEach((u) => URL.revokeObjectURL(u));
  state.selectedFiles = [];
  state.objectUrls = [];
  els.fileInput.value = "";
  els.folderInput.value = "";
  els.fileSummary.classList.add("hidden");
  els.fileListPreview.innerHTML = "";
  updateStartButtonState();
}

function renderFileListPreview() {
  els.fileSummary.classList.remove("hidden");
  const totalBytes = state.selectedFiles.reduce((acc, f) => acc + f.size, 0);
  const mb = (totalBytes / (1024 * 1024)).toFixed(1);
  els.fileCountText.innerHTML = `<strong>${state.selectedFiles.length}</strong>개 영상 준비됨 (${mb} MB)`;

  els.fileListPreview.innerHTML = "";
  state.selectedFiles.forEach((file, idx) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span>🎬 ${escapeHtml(file.name)}</span>
      <span>${(file.size / (1024 * 1024)).toFixed(1)}MB</span>
    `;
    li.addEventListener("click", () => {
      loadVideoToMainPlayer(state.objectUrls[idx], file.name, 0);
      Array.from(els.fileListPreview.children).forEach((el, i) =>
        el.classList.toggle("active-file", i === idx)
      );
    });
    if (idx === 0) li.classList.add("active-file");
    els.fileListPreview.appendChild(li);
  });
}

function updateStartButtonState() {
  const hasFolder = state.engineMode === "python" && els.localFolderPath.value.trim().length > 0;
  const hasFiles = state.selectedFiles.length > 0;
  els.startAnalyzeBtn.disabled = !(hasFiles || hasFolder) || state.isAnalyzing;
}

function loadVideoToMainPlayer(srcUrl, titleText, startTimeSec = 0) {
  els.emptyVideoPlaceholder.classList.add("hidden");
  els.currentVideoTitle.textContent = titleText;

  if (els.mainVideo.src !== srcUrl) {
    els.mainVideo.src = srcUrl;
    els.mainVideo.load();
    els.mainVideo.onloadeddata = () => {
      syncOverlayCanvas();
      els.mainVideo.currentTime = Math.max(0, startTimeSec);
      if (startTimeSec > 0) {
        els.mainVideo.play().catch(() => {});
      }
    };
  } else {
    els.mainVideo.currentTime = Math.max(0, startTimeSec);
    if (startTimeSec > 0) {
      els.mainVideo.play().catch(() => {});
    }
  }
}

// ============================================================================
// ROI 캔버스 오버레이 & 마우스 드래그 지정
// ============================================================================
function syncOverlayCanvas() {
  const video = els.mainVideo;
  const canvas = els.roiOverlayCanvas;
  const rect = video.getBoundingClientRect();
  const parentRect = els.videoZoomContainer.getBoundingClientRect();

  if (rect.width <= 0 || rect.height <= 0) return;

  canvas.width = rect.width;
  canvas.height = rect.height;
  canvas.style.left = `${rect.left - parentRect.left}px`;
  canvas.style.top = `${rect.top - parentRect.top}px`;

  drawRoiOverlay();
}

function drawRoiOverlay() {
  const canvas = els.roiOverlayCanvas;
  const ctx = canvas.getContext("2d");
  if (!ctx || canvas.width === 0 || canvas.height === 0) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const rx = state.roi.x * canvas.width;
  const ry = state.roi.y * canvas.height;
  const rw = state.roi.w * canvas.width;
  const rh = state.roi.h * canvas.height;

  // 바깥 어두운 마스크
  ctx.fillStyle = "rgba(0, 0, 0, 0.42)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.clearRect(rx, ry, rw, rh);

  // ROI 테두리
  ctx.strokeStyle = "#22d3ee";
  ctx.lineWidth = 2.5;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(rx, ry, rw, rh);
  ctx.setLineDash([]);

  // 라벨 표시
  ctx.fillStyle = "rgba(6, 182, 212, 0.9)";
  ctx.fillRect(rx, Math.max(0, ry - 22), 185, 22);
  ctx.fillStyle = "#000";
  ctx.font = "bold 12px sans-serif";
  ctx.fillText("🎯 집중 감시 구역 (ROI)", rx + 6, Math.max(15, ry - 7));
}

function initRoiCanvasDrag() {
  const canvas = els.roiOverlayCanvas;
  let isDragging = false;
  let startX = 0;
  let startY = 0;

  canvas.addEventListener("mousedown", (e) => {
    if (!els.drawRoiModeCheck.checked) return;
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    isDragging = true;
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const rect = canvas.getBoundingClientRect();
    const curX = Math.max(0, Math.min(canvas.width, e.clientX - rect.left));
    const curY = Math.max(0, Math.min(canvas.height, e.clientY - rect.top));

    const x = Math.min(startX, curX);
    const y = Math.min(startY, curY);
    const w = Math.abs(curX - startX);
    const h = Math.abs(curY - startY);

    if (w > 15 && h > 15 && canvas.width > 0 && canvas.height > 0) {
      state.roi = {
        x: +(x / canvas.width).toFixed(3),
        y: +(y / canvas.height).toFixed(3),
        w: +(w / canvas.width).toFixed(3),
        h: +(h / canvas.height).toFixed(3),
      };
      els.presetBtns.forEach((b) => b.classList.remove("active"));
      updateRoiBadgeText("마우스 사용자 지정 구역");
      drawRoiOverlay();
    }
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
  });
}

function updateRoiBadgeText(customLabel) {
  const x1 = Math.round(state.roi.x * 100);
  const x2 = Math.round((state.roi.x + state.roi.w) * 100);
  const y1 = Math.round(state.roi.y * 100);
  const y2 = Math.round((state.roi.y + state.roi.h) * 100);
  const title = customLabel || ROI_PRESETS[state.activePreset]?.name || "사용자 지정";
  els.roiCoordsBadge.textContent = `감시 영역: ${title} (가로 ${x1}%~${x2}%, 세로 ${y1}%~${y2}%)`;
}

// ============================================================================
// 일괄 분석 실행 (Python 백엔드 모드 & 브라우저 내장 고속 모드 자동 분기)
// ============================================================================
async function startBatchAnalysis() {
  state.isAnalyzing = true;
  state.cancelRequested = false;
  state.events = [];
  renderEventsGrid();

  els.startAnalyzeBtn.classList.add("hidden");
  els.stopAnalyzeBtn.classList.remove("hidden");
  els.progressPanel.classList.remove("hidden");
  els.exportReportBtn.disabled = true;

  updateProgressUI("분석 준비 중...", 0, 0, state.selectedFiles.length, 0);

  if (state.engineMode === "python") {
    await runPythonBackendAnalysis();
  } else {
    await runBrowserStandaloneAnalysis();
  }
}

async function stopBatchAnalysis() {
  state.cancelRequested = true;
  if (state.engineMode === "python" && state.currentJobId) {
    try {
      await fetch(`/api/jobs/${state.currentJobId}/cancel`, { method: "POST" });
    } catch (_) {}
  }
  finishAnalysisUI("분석이 사용자에 의해 중지되었습니다.");
}

function finishAnalysisUI(statusMsg) {
  state.isAnalyzing = false;
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  els.startAnalyzeBtn.classList.remove("hidden");
  els.stopAnalyzeBtn.classList.add("hidden");
  els.progressStatusText.textContent = statusMsg;
  els.exportReportBtn.disabled = state.events.length === 0;
  updateStartButtonState();
}

function updateProgressUI(statusText, pct, doneFiles, totalFiles, foundCount) {
  els.progressStatusText.textContent = statusText;
  els.progressPctText.textContent = `${Math.round(pct)}%`;
  els.progressBarFill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
  els.progressFileSub.textContent = `${doneFiles} / ${totalFiles} 파일 완료`;
  els.progressFoundSub.textContent = `감지된 의심 장면: ${foundCount}건`;
}

// ----------------------------------------------------------------------------
// 1) Python OpenCV 백엔드 분석
// ----------------------------------------------------------------------------
async function runPythonBackendAnalysis() {
  const sensitivity = parseInt(els.sensitivityRange.value, 10);
  const sampleFps = parseFloat(els.sampleFpsSelect.value);
  const mergeGap = parseFloat(els.mergeGapSelect.value);
  const detectShake = els.detectShakeCheck.checked;
  const folderPath = els.localFolderPath.value.trim();

  try {
    let res;
    if (folderPath && state.selectedFiles.length === 0) {
      res = await fetch("/api/analyze-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          folder_path: folderPath,
          roi_x: state.roi.x,
          roi_y: state.roi.y,
          roi_w: state.roi.w,
          roi_h: state.roi.h,
          sample_fps: sampleFps,
          sensitivity: sensitivity,
          merge_gap_sec: mergeGap,
          detect_shake: detectShake,
        }),
      });
    } else {
      const formData = new FormData();
      state.selectedFiles.forEach((f) => formData.append("files", f));
      formData.append("roi_x", state.roi.x);
      formData.append("roi_y", state.roi.y);
      formData.append("roi_w", state.roi.w);
      formData.append("roi_h", state.roi.h);
      formData.append("sample_fps", sampleFps);
      formData.append("sensitivity", sensitivity);
      formData.append("merge_gap_sec", mergeGap);
      formData.append("detect_shake", detectShake);

      updateProgressUI("영상을 로컬 분석 엔진으로 전달 중...", 3, 0, state.selectedFiles.length, 0);
      res = await fetch("/api/analyze-upload", {
        method: "POST",
        body: formData,
      });
    }

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "분석 요청 실패");
    }

    const { job_id } = await res.json();
    state.currentJobId = job_id;

    state.pollTimer = setInterval(async () => {
      try {
        const pollRes = await fetch(`/api/jobs/${job_id}`);
        if (!pollRes.ok) return;
        const job = await pollRes.json();

        const prevLen = state.events.length;
        state.events = job.events || [];
        if (state.events.length !== prevLen) {
          renderEventsGrid();
          // 첫 감지 장면이 나오면 자동으로 플레이어에 표시
          if (prevLen === 0 && state.events.length > 0) {
            selectAndPlayEvent(state.events[0], false);
          }
        }

        updateProgressUI(
          job.status === "running"
            ? `분석 중: ${job.current_file_name} (${job.current_file_progress}%)`
            : "분석 완료!",
          job.overall_progress,
          job.completed_files,
          job.total_files,
          state.events.length
        );

        if (job.status === "completed" || job.status === "cancelled" || job.status === "error") {
          renderEventsGrid();
          finishAnalysisUI(
            job.status === "completed"
              ? `✅ 총 ${job.total_files}개 영상 분석 완료 (${job.elapsed_sec}초 소요) - ${state.events.length}건 감지`
              : "분석 중지됨"
          );
        }
      } catch (e) {
        console.error(e);
      }
    }, 450);
  } catch (err) {
    alert(`오류 발생: ${err.message}`);
    finishAnalysisUI("오류로 인해 분석이 중단되었습니다.");
  }
}

// ----------------------------------------------------------------------------
// 2) 브라우저 내장 HTML5 Canvas 고속 모션 & 흔들림 분석 엔진
// ----------------------------------------------------------------------------
async function runBrowserStandaloneAnalysis() {
  const files = state.selectedFiles;
  if (!files.length) {
    finishAnalysisUI("선택된 파일이 없습니다.");
    return;
  }

  const sensitivity = parseInt(els.sensitivityRange.value, 10);
  const sampleFps = parseFloat(els.sampleFpsSelect.value);
  const mergeGap = parseFloat(els.mergeGapSelect.value);
  const detectShake = els.detectShakeCheck.checked;

  const workerVideo = els.hiddenWorkerVideo;
  const workerCanvas = els.hiddenWorkerCanvas;
  const ctx = workerCanvas.getContext("2d", { willReadFrequently: true });

  const diffThresh = Math.max(10, Math.min(52, Math.round(56 - sensitivity * 0.48)));
  const minAreaPct = Math.max(0.12, 0.38 * ((110 - sensitivity) / 50));

  for (let fIdx = 0; fIdx < files.length; fIdx++) {
    if (state.cancelRequested) break;

    const file = files[fIdx];
    const fileUrl = state.objectUrls[fIdx];

    await new Promise((resolve) => {
      workerVideo.src = fileUrl;
      workerVideo.onloadedmetadata = () => resolve();
      workerVideo.onerror = () => resolve();
    });

    const duration = workerVideo.duration || 0;
    if (!duration || !isFinite(duration)) continue;

    const analysisW = 320;
    const scale = analysisW / Math.max(1, workerVideo.videoWidth || 640);
    const analysisH = Math.max(180, Math.round((workerVideo.videoHeight || 360) * scale));
    workerCanvas.width = analysisW;
    workerCanvas.height = analysisH;

    const rx = Math.floor(state.roi.x * analysisW);
    const ry = Math.floor(state.roi.y * analysisH);
    const rw = Math.max(16, Math.floor(state.roi.w * analysisW));
    const rh = Math.max(16, Math.floor(state.roi.h * analysisH));

    const stepSec = 1.0 / sampleFps;
    let prevPixels = null;
    let currentSeg = null;
    const fileSegments = [];

    for (let t = 0; t < duration; t += stepSec) {
      if (state.cancelRequested) break;

      await seekVideoAsync(workerVideo, t);
      ctx.drawImage(workerVideo, 0, 0, analysisW, analysisH);
      const imgData = ctx.getImageData(0, 0, analysisW, analysisH).data;

      if (!prevPixels) {
        prevPixels = new Uint8ClampedArray(imgData);
        continue;
      }

      let activeInRoi = 0;
      let activeGlobal = 0;
      let minBx = rw, minBy = rh, maxBx = 0, maxBy = 0;

      for (let y = 0; y < analysisH; y += 2) {
        for (let x = 0; x < analysisW; x += 2) {
          const idx = (y * analysisW + x) * 4;
          const dR = Math.abs(imgData[idx] - prevPixels[idx]);
          const dG = Math.abs(imgData[idx + 1] - prevPixels[idx + 1]);
          const dB = Math.abs(imgData[idx + 2] - prevPixels[idx + 2]);
          const diff = (dR + dG + dB) / 3;

          if (diff > diffThresh + 6) {
            activeGlobal++;
          }

          if (x >= rx && x < rx + rw && y >= ry && y < ry + rh && diff > diffThresh) {
            activeInRoi++;
            const lx = x - rx;
            const ly = y - ry;
            if (lx < minBx) minBx = lx;
            if (ly < minBy) minBy = ly;
            if (lx > maxBx) maxBx = lx;
            if (ly > maxBy) maxBy = ly;
          }
        }
      }

      prevPixels.set(imgData);

      const sampledRoiTotal = Math.max(1, (rw / 2) * (rh / 2));
      const sampledGlobalTotal = Math.max(1, (analysisW / 2) * (analysisH / 2));

      const areaPct = (activeInRoi / sampledRoiTotal) * 100;
      const globalRatio = activeGlobal / sampledGlobalTotal;
      const shakeScore =
        detectShake && globalRatio >= 0.18 && globalRatio <= 0.85
          ? Math.min(100, globalRatio * 180)
          : 0;

      const isMotion = areaPct >= minAreaPct || shakeScore >= 40;
      const score = Math.min(99, (areaPct / Math.max(0.2, minAreaPct)) * 22 + shakeScore * 0.65);

      if (isMotion) {
        const bbox =
          maxBx > minBx && maxBy > minBy
            ? {
                x: (rx + minBx) / analysisW,
                y: (ry + minBy) / analysisH,
                w: (maxBx - minBx) / analysisW,
                h: (maxBy - minBy) / analysisH,
              }
            : { ...state.roi };

        if (!currentSeg) {
          const thumbUrl = createAnnotatedBrowserThumbnail(workerVideo, bbox, shakeScore >= 38);
          currentSeg = {
            start_time: t,
            end_time: t,
            peak_time: t,
            max_score: score,
            max_area_pct: areaPct,
            max_shake: shakeScore,
            bbox,
            thumbUrl,
          };
        } else {
          currentSeg.end_time = t;
          currentSeg.max_shake = Math.max(currentSeg.max_shake, shakeScore);
          if (score > currentSeg.max_score) {
            currentSeg.max_score = score;
            currentSeg.max_area_pct = Math.max(currentSeg.max_area_pct, areaPct);
            currentSeg.peak_time = t;
            currentSeg.bbox = bbox;
            currentSeg.thumbUrl = createAnnotatedBrowserThumbnail(workerVideo, bbox, shakeScore >= 38);
          }
        }
      } else if (currentSeg && t - currentSeg.end_time > mergeGap) {
        fileSegments.push(currentSeg);
        currentSeg = null;
      }

      const filePct = Math.min(99, (t / duration) * 100);
      const overallPct = ((fIdx + filePct / 100) / files.length) * 100;
      updateProgressUI(
        `분석 중: ${file.name} (${Math.round(filePct)}%)`,
        overallPct,
        fIdx,
        files.length,
        state.events.length + fileSegments.length
      );
    }

    if (currentSeg) fileSegments.push(currentSeg);

    // 세그먼트를 이벤트 객체로 등록
    fileSegments.forEach((seg, sIdx) => {
      const isImpact = seg.max_shake >= 38 || seg.max_area_pct >= 14;
      const category =
        seg.max_shake >= 38
          ? "impact"
          : seg.max_area_pct >= 8 || seg.max_score >= 72
          ? "close_motion"
          : "motion";
      const label =
        category === "impact"
          ? "충격/접촉 의심 (화면 흔들림)"
          : category === "close_motion"
          ? "근접 큰 움직임 (밀착 통행)"
          : "주변 움직임 감지";

      const startT = Math.max(0, +(seg.start_time - 1.0).toFixed(2));
      const endT = +Math.min(duration, seg.end_time + 1.0).toFixed(2);

      state.events.push({
        event_id: `b_ev_${fIdx}_${sIdx + 1}`,
        file_name: file.name,
        file_index: fIdx,
        start_time: startT,
        end_time: endT,
        peak_time: +seg.peak_time.toFixed(2),
        duration: +Math.max(0.5, endT - startT).toFixed(2),
        score: +Math.max(18, seg.max_score).toFixed(1),
        area_pct: +seg.max_area_pct.toFixed(2),
        shake_score: +seg.max_shake.toFixed(1),
        is_impact_suspected: isImpact,
        category,
        label,
        bbox: seg.bbox,
        thumbnail_data_url: seg.thumbUrl,
        video_url: fileUrl,
      });
    });

    renderEventsGrid();
  }

  finishAnalysisUI(
    state.cancelRequested
      ? "분석 중지됨"
      : `✅ 총 ${files.length}개 영상 분석 완료 - 의심 장면 ${state.events.length}건 감지`
  );
}

function seekVideoAsync(videoEl, timeSec) {
  return new Promise((resolve) => {
    const onSeeked = () => {
      videoEl.removeEventListener("seeked", onSeeked);
      resolve();
    };
    videoEl.addEventListener("seeked", onSeeked);
    videoEl.currentTime = timeSec;
  });
}

function createAnnotatedBrowserThumbnail(videoEl, bbox, isImpact) {
  const c = document.createElement("canvas");
  c.width = 480;
  c.height = 270;
  const ctx = c.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, c.width, c.height);

  // ROI 표시
  ctx.strokeStyle = "#06b6d4";
  ctx.lineWidth = 2;
  ctx.strokeRect(
    state.roi.x * c.width,
    state.roi.y * c.height,
    state.roi.w * c.width,
    state.roi.h * c.height
  );

  // 움직임 바운딩 박스 표시
  ctx.strokeStyle = isImpact ? "#ef4444" : "#f59e0b";
  ctx.lineWidth = 3;
  ctx.strokeRect(
    bbox.x * c.width,
    bbox.y * c.height,
    Math.max(18, bbox.w * c.width),
    Math.max(18, bbox.h * c.height)
  );

  return c.toDataURL("image/jpeg", 0.82);
}

// ============================================================================
// 결과 카드 렌더링 및 클릭 시 해당 시점 재생
// ============================================================================
function renderEventsGrid() {
  els.totalEventsBadge.textContent = `${state.events.length}건`;

  let list = [...state.events];
  if (state.activeFilter !== "all") {
    list = list.filter((ev) => ev.category === state.activeFilter);
  }

  if (state.sortOrder === "score_desc") {
    list.sort((a, b) => b.score - a.score);
  } else {
    list.sort((a, b) => a.file_index - b.file_index || a.peak_time - b.peak_time);
  }

  if (list.length === 0) {
    els.eventsGrid.innerHTML = `
      <div class="empty-results">
        <p>${
          state.events.length === 0
            ? "감지된 의심 장면이 아직 없습니다. 좌측에서 영상을 추가하고 <strong>[일괄 분석 시작하기]</strong>를 눌러주세요."
            : "현재 선택한 필터에 해당하는 장면이 없습니다."
        }</p>
      </div>
    `;
    return;
  }

  els.eventsGrid.innerHTML = "";
  list.forEach((ev) => {
    const card = document.createElement("div");
    card.className = `event-card ${ev.category === "impact" ? "impact-card" : ""} ${
      state.selectedEventId === ev.event_id ? "selected" : ""
    }`;

    card.innerHTML = `
      <div class="thumb-wrap">
        <img src="${ev.thumbnail_data_url}" alt="${escapeHtml(ev.label)}" loading="lazy" />
        <span class="event-badge ${ev.category}">${escapeHtml(ev.label)}</span>
        <span class="score-pill">위험도 ${ev.score}점</span>
        <span class="time-overlay">${formatTime(ev.start_time)} ~ ${formatTime(ev.end_time)}</span>
      </div>
      <div class="event-body">
        <div class="event-file" title="${escapeHtml(ev.file_name)}">🎬 ${escapeHtml(ev.file_name)}</div>
        <div class="event-meta">
          <span>피크 시점: <strong>${formatTime(ev.peak_time)}</strong></span>
          <span>변화량: ${ev.area_pct}%</span>
        </div>
      </div>
    `;

    card.addEventListener("click", () => selectAndPlayEvent(ev, true));
    els.eventsGrid.appendChild(card);
  });
}

function selectAndPlayEvent(ev, autoPlay = true) {
  state.selectedEventId = ev.event_id;
  renderEventsGrid();

  // 로컬 Blob URL이 있으면 우선 사용하여 더욱 빠른 탐색 지원
  const localBlobUrl = state.objectUrls[ev.file_index];
  const playUrl = localBlobUrl || ev.video_url;
  const jumpTime = Math.max(0, ev.peak_time - 2.0);

  loadVideoToMainPlayer(
    playUrl,
    `${ev.file_name} — [${ev.label}] 피크 시점 ${formatTime(ev.peak_time)} (2초 전부터 재생)`,
    jumpTime
  );

  if (!autoPlay) {
    els.mainVideo.pause();
  }
}

// ============================================================================
// 비디오 정밀 판독 유틸리티 (탐색, 줌, 캡처, 리포트 다운로드)
// ============================================================================
function seekRelative(deltaSec) {
  if (!els.mainVideo.src) return;
  els.mainVideo.currentTime = Math.max(
    0,
    Math.min(els.mainVideo.duration || 99999, els.mainVideo.currentTime + deltaSec)
  );
}

function applyVideoZoom() {
  els.videoZoomContainer.style.transform = `scale(${state.zoomLevel}) translate(${state.panX}px, ${state.panY}px)`;
}

function captureCurrentVideoFrame() {
  const v = els.mainVideo;
  if (!v.src || v.videoWidth === 0) {
    alert("먼저 캡처할 영상을 재생하거나 선택해 주세요.");
    return;
  }

  const c = document.createElement("canvas");
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(v, 0, 0, c.width, c.height);

  // 하단 증거 워터마크 바
  ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
  ctx.fillRect(0, c.height - 44, c.width, 44);
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 20px sans-serif";
  const timeStr = formatTime(v.currentTime);
  ctx.fillText(
    `[증거 캡처] ${els.currentVideoTitle.textContent} | 시점: ${timeStr}`,
    16,
    c.height - 15
  );

  const link = document.createElement("a");
  link.download = `블랙박스_증거캡처_${timeStr.replace(":", "분")}초.png`;
  link.href = c.toDataURL("image/png");
  link.click();
}

function exportHtmlReport() {
  if (!state.events.length) return;

  const rows = state.events
    .map(
      (ev, i) => `
      <div style="border:1px solid #cbd5e1;border-radius:10px;padding:14px;margin-bottom:14px;display:flex;gap:16px;align-items:center;">
        <img src="${ev.thumbnail_data_url}" style="width:280px;border-radius:8px;border:1px solid #94a3b8;" />
        <div>
          <h3 style="margin:0 0 6px 0;">#${i + 1} [${escapeHtml(ev.label)}] - 위험도 ${ev.score}점</h3>
          <p style="margin:4px 0;"><strong>파일명:</strong> ${escapeHtml(ev.file_name)}</p>
          <p style="margin:4px 0;"><strong>감지 구간:</strong> ${formatTime(ev.start_time)} ~ ${formatTime(ev.end_time)} (최대 움직임 시점: <strong>${formatTime(ev.peak_time)}</strong>)</p>
          <p style="margin:4px 0;"><strong>영역 변화율:</strong> ${ev.area_pct}% / <strong>흔들림 지수:</strong> ${ev.shake_score}</p>
        </div>
      </div>`
    )
    .join("");

  const html = `<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <title>블랙박스 범인 잡기 - 감지 리포트</title>
</head>
<body style="font-family:sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#0f172a;">
  <h1>🚨 블랙박스 의심 장면 자동 분석 리포트</h1>
  <p>생성 일시: ${new Date().toLocaleString("ko-KR")} | 총 감지 건수: <strong>${state.events.length}건</strong></p>
  <hr style="margin:16px 0;" />
  ${rows}
</body>
</html>`;

  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const link = document.createElement("a");
  link.download = `블랙박스_범인감지_리포트_${new Date().toISOString().slice(0, 10)}.html`;
  link.href = URL.createObjectURL(blob);
  link.click();
}

function formatTime(sec) {
  if (!isFinite(sec) || sec < 0) return "00:00.0";
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(4, "0")}`;
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

document.addEventListener("DOMContentLoaded", initApp);
