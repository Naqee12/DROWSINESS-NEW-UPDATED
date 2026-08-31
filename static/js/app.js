/* ==========================================================================
   DriveGuard AI – Application JavaScript
   Handles UI state, API polling, accessibility, and interactions
   ========================================================================== */

(function () {
  'use strict';

  /* ---------- Configuration ---------- */
  const CONFIG = {
    pollInterval: 500,
    severityColors: {
      'NORMAL': '#43a047',
      'MILD FATIGUE': '#fdd835',
      'DROWSY': '#fb8c00',
      'CRITICAL': '#e53935'
    },
    severityOrder: ['NORMAL', 'MILD FATIGUE', 'DROWSY', 'CRITICAL'],
    sparklinePoints: 30
  };

  /* ---------- State ---------- */
  const state = {
    metrics: null,
    calibrating: true,
    sessionStart: Date.now(),
    sparklineHistory: {
      ear: [],
      mar: [],
      perclos: [],
      fusion: []
    },
    videoLoaded: false
  };

  /* ---------- DOM References ---------- */
  const els = {};

  function cacheElements() {
    Object.assign(els, {
      sidebar: document.getElementById('sidebar'),
      sidebarOverlay: document.getElementById('sidebar-overlay'),
      menuToggle: document.getElementById('menu-toggle'),
      navLinks: document.querySelectorAll('.nav-link'),
      pageTitle: document.getElementById('page-title'),
      globalStatus: document.getElementById('global-status'),
      statusDot: document.querySelector('.status-dot'),
      statusText: document.querySelector('.status-text'),
      safetyBanner: document.getElementById('safety-banner'),
      bannerTitle: document.getElementById('banner-title'),
      bannerMessage: document.getElementById('banner-message'),
      bannerDismiss: document.querySelector('.banner-dismiss'),
      videoFeed: document.getElementById('video-feed'),
      videoPlaceholder: document.getElementById('video-placeholder'),
      recalibrateBtn: document.getElementById('recalibrate-btn'),
      fullscreenBtn: document.getElementById('fullscreen-btn'),
      metricValues: {
        ear: document.getElementById('metric-ear'),
        mar: document.getElementById('metric-mar'),
        perclos: document.getElementById('metric-perclos'),
        fusion: document.getElementById('metric-fusion'),
        blinks: document.getElementById('metric-blinks'),
        yawns: document.getElementById('metric-yawns')
      },
      sparklines: {
        ear: document.querySelector('[data-metric="ear"] .trend-sparkline polyline'),
        mar: document.querySelector('[data-metric="mar"] .trend-sparkline polyline'),
        perclos: document.querySelector('[data-metric="perclos"] .trend-sparkline polyline'),
        fusion: document.querySelector('[data-metric="fusion"] .trend-sparkline polyline')
      },
      details: {
        sessionTime: document.getElementById('detail-session-time'),
        calibration: document.getElementById('detail-calibration'),
        pitch: document.getElementById('detail-pitch'),
        phone: document.getElementById('detail-phone'),
        drowsyEvents: document.getElementById('detail-drowsy-events'),
        fatigue: document.getElementById('detail-fatigue')
      },
      phoneStatusIndicator: document.getElementById('phone-status-indicator'),
      phoneStatusText: document.getElementById('phone-status-text'),
      driverName: document.getElementById('driver-name'),
      saveProfileBtn: document.getElementById('save-profile-btn'),
      downloadCsv: document.getElementById('download-csv'),
      toastContainer: document.getElementById('toast-container')
    });
  }

  /* ---------- Utility Functions ---------- */
  function fmtNumber(val, decimals = 3) {
    return typeof val === 'number' && !isNaN(val) ? val.toFixed(decimals) : '--';
  }

  function fmtTime(ms) {
    const s = Math.floor(ms / 1000);
    const h = Math.floor(s / 3600).toString().padStart(2, '0');
    const m = Math.floor((s % 3600) / 60).toString().padStart(2, '0');
    const sec = (s % 60).toString().padStart(2, '0');
    return `${h}:${m}:${sec}`;
  }

  function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

  function mapToHeight(val, min, max, height) {
    const clamped = clamp((val - min) / (max - min), 0, 1);
    return height - clamped * height;
  }

  function updateSparkline(polyline, history, min, max) {
    if (!polyline) return;
    const width = 60;
    const height = 30;
    const step = width / (CONFIG.sparklinePoints - 1);
    const points = history.slice(-CONFIG.sparklinePoints).map((v, i) => {
      const x = i * step;
      const y = mapToHeight(v, min, max, height);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    polyline.setAttribute('points', points.join(' '));
  }

  function showToast(message, type = 'info', duration = 4000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'polite');
    const icons = {
      success: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
      error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
      warning: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
      info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
    };
    toast.innerHTML = `
      <span class="toast-icon" aria-hidden="true">${icons[type]}</span>
      <span class="toast-message">${message}</span>
      <button class="toast-close" aria-label="Dismiss">&times;</button>
    `;
    toast.querySelector('.toast-close').addEventListener('click', () => toast.remove());
    els.toastContainer.appendChild(toast);
    setTimeout(() => { if (toast.parentNode) toast.remove(); }, duration);
  }

  function speak(text) {
    if ('speechSynthesis' in window) {
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 1;
      utter.pitch = 1;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utter);
    }
  }

  function playAlertTone(freq = 880, duration = 400) {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      gain.gain.value = 0.1;
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration / 1000);
      setTimeout(() => { osc.stop(); ctx.close(); }, duration + 50);
    } catch (_) {}
  }

  /* ---------- Sidebar Navigation ---------- */
  function initSidebar() {
    els.menuToggle.addEventListener('click', toggleSidebar);
    els.sidebarOverlay.addEventListener('click', closeSidebar);

    els.navLinks.forEach(link => {
      link.addEventListener('click', e => {
        e.preventDefault();
        const page = link.dataset.page;
        activateNav(page);
        closeSidebar();
      });
    });

    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') closeSidebar();
    });
  }

  function toggleSidebar() {
    const open = els.sidebar.classList.toggle('open');
    els.sidebarOverlay.classList.toggle('visible', open);
    els.menuToggle.setAttribute('aria-expanded', open);
  }

  function closeSidebar() {
    els.sidebar.classList.remove('open');
    els.sidebarOverlay.classList.remove('visible');
    els.menuToggle.setAttribute('aria-expanded', 'false');
  }

  function activateNav(page) {
    els.navLinks.forEach(l => {
      const active = l.dataset.page === page;
      l.classList.toggle('active', active);
      l.setAttribute('aria-current', active ? 'page' : 'false');
    });
    const titles = {
      dashboard: 'Dashboard',
      analytics: 'Analytics',
      sessions: 'Sessions',
      reports: 'Reports',
      settings: 'Settings'
    };
    els.pageTitle.textContent = titles[page] || 'Dashboard';
  }

  /* ---------- Safety Banner ---------- */
  function showSafetyBanner(severity, message) {
    const titles = {
      'MILD FATIGUE': 'Caution',
      'DROWSY': 'Warning',
      'CRITICAL': 'Critical Alert'
    };
    els.bannerTitle.textContent = titles[severity] || 'Alert';
    els.bannerMessage.textContent = message;
    els.safetyBanner.className = 'safety-banner ' + severity.toLowerCase().replace(' ', '-');
    els.safetyBanner.hidden = false;
  }

  function hideSafetyBanner() {
    els.safetyBanner.hidden = true;
  }

  /* ---------- Global Status Indicator ---------- */
  function updateGlobalStatus(severity) {
    const dot = els.statusDot;
    const text = els.statusText;
    const container = els.globalStatus;
    dot.style.background = CONFIG.severityColors[severity] || CONFIG.severityColors.NORMAL;
    dot.style.boxShadow = `0 0 8px ${CONFIG.severityColors[severity] || CONFIG.severityColors.NORMAL}`;
    
    const emojis = {
      'NORMAL': '🟢',
      'MILD FATIGUE': '🟡',
      'DROWSY': '🟠',
      'CRITICAL': '🔴'
    };
    text.textContent = `${emojis[severity] || '🟢'} ${severity}`;
    container.className = 'status-indicator';
    if (severity === 'MILD FATIGUE') container.classList.add('warn');
    else if (severity === 'DROWSY' || severity === 'CRITICAL') container.classList.add('danger');
  }

  /* ---------- Metric Cards ---------- */
  function updateMetrics(metrics) {
    const m = metrics;
    els.metricValues.ear.textContent = fmtNumber(m.ear);
    els.metricValues.mar.textContent = fmtNumber(m.mar);
    els.metricValues.perclos.textContent = fmtNumber(m.perclos, 1);
    els.metricValues.fusion.textContent = fmtNumber(m.fusion, 2);
    els.metricValues.blinks.textContent = m.blink_count ?? '--';
    els.metricValues.yawns.textContent = m.yawn_count ?? '--';

    // Sparkline history
    if (typeof m.ear === 'number') state.sparklineHistory.ear.push(m.ear);
    if (typeof m.mar === 'number') state.sparklineHistory.mar.push(m.mar);
    if (typeof m.perclos === 'number') state.sparklineHistory.perclos.push(m.perclos);
    if (typeof m.fusion === 'number') state.sparklineHistory.fusion.push(m.fusion);

    updateSparkline(els.sparklines.ear, state.sparklineHistory.ear, 0, 0.4);
    updateSparkline(els.sparklines.mar, state.sparklineHistory.mar, 0, 1);
    updateSparkline(els.sparklines.perclos, state.sparklineHistory.perclos, 0, 100);
    updateSparkline(els.sparklines.fusion, state.sparklineHistory.fusion, 0, 1);
  }

  /* ---------- Details Panel ---------- */
  function updateDetails(metrics) {
    const m = metrics;
    const elapsed = Date.now() - state.sessionStart;
    els.details.sessionTime.textContent = fmtTime(elapsed);
    els.details.calibration.textContent = m.calibrating ? `Calibrating… ${m.calib_remaining}s` : 'Complete';
    els.details.pitch.textContent = typeof m.pitch === 'number' ? `${m.pitch.toFixed(1)}°` : '--°';
    els.details.drowsyEvents.textContent = m.drowsy_events ?? 0;
    els.details.fatigue.textContent = m.fatigue_warning || '—';

    // Phone detection status
    if (m.phone_detected) {
      els.phoneStatusIndicator.classList.add('on');
      els.phoneStatusText.textContent = 'ON — Phone detected';
    } else {
      els.phoneStatusIndicator.classList.remove('on');
      els.phoneStatusText.textContent = 'OFF — No phone detected';
    }
  }

  /* ---------- Video Feed ---------- */
  function initVideo() {
    // Hide placeholder by default; show only on genuine error
    els.videoFeed.addEventListener('error', () => {
      els.videoPlaceholder.classList.add('show');
    });
    els.fullscreenBtn.addEventListener('click', toggleFullscreen);
  }

  function toggleFullscreen() {
    const container = els.videoFeed.parentElement;
    if (!document.fullscreenElement) {
      container.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen();
    }
  }

  /* ---------- Controls ---------- */
  function initControls() {
    els.recalibrateBtn.addEventListener('click', requestRecalibrate);
    els.saveProfileBtn.addEventListener('click', saveProfile);
  }

  async function requestRecalibrate() {
    try {
      const res = await fetch('/recalibrate', { method: 'POST' });
      const data = await res.json();
      if (data.ok) {
        showToast('Recalibration started', 'success');
        state.sessionStart = Date.now();
      } else {
        showToast('Recalibration failed', 'error');
      }
    } catch (_) { showToast('Network error', 'error'); }
  }

  async function saveProfile() {
    const name = els.driverName.value.trim();
    if (!name) { showToast('Enter a driver name', 'warning'); return; }
    try {
      const res = await fetch('/save_profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });
      const data = await res.json();
      if (data.ok) showToast('Profile saved', 'success');
      else showToast('Error: ' + (data.error || 'Unknown'), 'error');
    } catch (_) { showToast('Network error', 'error'); }
  }

  /* ---------- Alert Handling ---------- */
  function handleAlert(metrics) {
    const event = metrics.new_alert_event;
    const severity = metrics.severity;
    if (!event || (event !== 'DROWSY' && event !== 'CRITICAL')) return;
    if (severity !== 'DROWSY' && severity !== 'CRITICAL') return;

    const messages = {
      'DROWSY': 'Drowsiness detected',
      'CRITICAL': 'Critical drowsiness detected'
    };
    const titles = {
      'DROWSY': 'Warning',
      'CRITICAL': 'Critical Warning'
    };
    const msg = messages[event];
    const title = titles[event];
    if (msg) {
      speak(msg);
      playAlertTone(event === 'CRITICAL' ? 1000 : 880, 400);
      showSafetyBanner(event, msg, title);
      showToast(msg, event === 'CRITICAL' ? 'error' : 'warning');
    }
  }

  function clearAlertIfNeeded(severity) {
    if (severity === 'NORMAL' || severity === 'MILD FATIGUE') {
      hideSafetyBanner();
    }
  }

  function showSafetyBanner(severity, message, title) {
    els.bannerTitle.textContent = title;
    els.bannerMessage.textContent = message;
    els.safetyBanner.className = 'safety-banner ' + severity.toLowerCase().replace(' ', '-');
    els.safetyBanner.hidden = false;
  }

  /* ---------- Polling Loop ---------- */
  async function pollMetrics() {
    try {
      const res = await fetch('/metrics');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const m = await res.json();

      state.metrics = m;

      if (m.calibrating) {
        state.calibrating = true;
        els.pageTitle.textContent = `Calibrating… ${m.calib_remaining}s`;
        updateGlobalStatus('NORMAL');
        hideSafetyBanner();
      } else {
        if (state.calibrating) {
          state.calibrating = false;
          state.sessionStart = Date.now();
          els.pageTitle.textContent = 'Dashboard';
        }
        updateMetrics(m);
        updateDetails(m);
        updateGlobalStatus(m.severity);
        handleAlert(m);
        clearAlertIfNeeded(m.severity);
      }
    } catch (err) {
      console.warn('Metrics poll failed:', err);
      // Optionally show connection warning
    }
  }

  /* ---------- Initialization ---------- */
  function init() {
    cacheElements();
    initSidebar();
    initVideo();
    initControls();
    els.bannerDismiss.addEventListener('click', hideSafetyBanner);
    hideSafetyBanner();

    // Initial poll
    pollMetrics();
    setInterval(pollMetrics, CONFIG.pollInterval);

    // Handle visibility change to reduce polling when hidden
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) return;
      pollMetrics();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();