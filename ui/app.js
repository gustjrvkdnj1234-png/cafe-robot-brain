// Page controller: case picker, replay animation, live mode, exports.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const cv = $("stage"), ctx = cv.getContext("2d");
  const params = new URLSearchParams(location.search);
  const DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));

  // replay captions: still image (final PNG) vs animation (page, GIF, PNG frames); source notes go in the blog caption
  const FOOT_STILL = ["응답 시간은 같은 노트북에서 측정한 중앙값입니다."];
  const FOOT_ANIM = [...FOOT_STILL, "애니메이션은 실제보다 빠르게 재생됩니다."];
  const FOOT_LIVE = ["직접 주문: 시간은 이번 주문의 실제 값(모델 불러오기 포함), 메모리는 응답 직후 값입니다.",
    "배포용 배치(6c): 주문 내역을 프롬프트 맨 끝에 둡니다. 내보낸 GIF는 애니메이션 속도를 보기 좋게 줄였습니다."];
  let robots = [], scene = null, L = null, t0 = 0, fixedT = null, raf = 0;
  let live = { robot: "64e", history: [], state: [], stateKo: [], busy: false };

  const status = (html) => { $("status").innerHTML = html; };

  function relayout() {
    scene = Stage.timeline(scene);
    const nL = Stage.layout(ctx, scene);
    if (!L || nL.H !== L.H || cv.width !== Stage.W * DPR) {
      cv.width = Stage.W * DPR; cv.height = nL.H * DPR;
      cv.style.width = Stage.W + "px"; cv.style.height = nL.H + "px";
    }
    L = nL;
  }
  function setScene(s, startNow = true) {
    scene = s; L = null; relayout();
    if (startNow) t0 = performance.now();
    loop();
  }
  function now() { return fixedT != null ? fixedT : (performance.now() - t0) / 1000; }
  function frame() {
    if (!scene) return;
    if (scene.live) relayout();
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    Stage.draw(ctx, scene, Math.min(now(), scene.live ? 1e9 : scene.tEnd), L);
  }
  function loop() {
    cancelAnimationFrame(raf);
    const step = () => {
      frame();
      const t = now();
      if (fixedT == null && (scene.live ? (live.busy || t < scene.tEnd + 1) : t < scene.tEnd)) raf = requestAnimationFrame(step);
    };
    step();
  }

  // ---------- replay ----------
  async function loadCases() {
    const d = await (await fetch("/api/cases")).json();
    robots = d.robots;
    const sel = $("casePick");
    const groups = [["pick", "★ 블로그용 추천"], ["live", "실전 시험 (사람이 쓴 20문항)"], ["cafe", "카페 시험 (120문항)"]];
    for (const [g, label] of groups) {
      const og = document.createElement("optgroup"); og.label = label;
      for (const c of d.cases.filter((c) => (g === "pick" ? c.pick : !c.pick && c.exam === g))) {
        const o = document.createElement("option"); o.value = c.id;
        const txt = c.text.length > 30 ? c.text.slice(0, 30) + "…" : c.text;
        o.textContent = `${c.pick ? "★ " : ""}${c.id} · ${c.category} — ${txt}   ${c.marks}`;
        og.appendChild(o);
      }
      sel.appendChild(og);
    }
    const lr = $("liveRobot");
    for (const r of robots) { const o = document.createElement("option"); o.value = r.key; o.textContent = `${r.name} (${r.sub})`; lr.appendChild(o); }
    lr.value = live.robot;
  }

  async function replay(id) {
    const d = await (await fetch("/api/replay?case=" + encodeURIComponent(id))).json();
    if (d.error) { status(d.error); return; }
    const conv = d.conversation;
    const lastUser = conv.map((m) => m.role).lastIndexOf("user");
    const s = {
      live: false, tag: `${d.exam_ko} ${d.id} · ${d.category}`, footer: FOOT_ANIM, caseId: d.id,
      badgeLabel: "주문당 응답 시간(중앙값)",
      history: conv.slice(0, lastUser), say: conv[lastUser].content, orderState: d.order_state, expected: d.expected,
      robots: d.robots.map((r) => ({ ...r })),
    };
    await Stage.loadFonts(s);
    setScene(s);
    status("");
  }

  // ---------- live ----------
  function liveScene(say, pendingKey) {
    return {
      live: true, tag: "직접 주문 · 배포용 프롬프트", footer: FOOT_LIVE, caseId: "live",
      badgeLabel: "이번 주문 실측 · 배포용 배치",
      history: live.history.slice(-4), say: say || "(아래 칸에 주문을 입력해 보세요)", orderState: live.stateKo, expected: null,
      robots: robots.map((r) => ({ ...r, off: r.key !== pendingKey, done: false, wall_s: 0, lines: null, correct: null })),
    };
  }
  function showLiveOrder() {
    $("liveOrder").textContent = live.stateKo.length ? "현재 주문: " + live.stateKo.join(", ") : "현재 주문: 없음";
  }
  async function liveSend() {
    const text = $("liveText").value.trim();
    if (!text || live.busy) return;
    if (!$("liveKeep").checked) { live.history = []; live.state = []; live.stateKo = []; }
    live.robot = $("liveRobot").value;
    live.busy = true; $("liveSend").disabled = true;
    const s = liveScene(text, live.robot);
    await Stage.loadFonts(s);
    setScene(s);
    const r = s.robots.find((x) => x.key === live.robot);
    status(`${r.name}에게 보내는 중… (모델을 처음 불러오면 오래 걸려요)`);
    try {
      const res = await (await fetch("/api/live", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ robot: live.robot, mode: $("liveMode").value || null, text, history: live.history, order_state: live.state }),
      })).json();
      if (res.error) throw new Error(res.error);
      Object.assign(r, { done: true, wall_s: res.wall_s, lines: res.lines, memory_gb: res.memory_gb });
      await Stage.loadFonts(scene);
      relayout();
      live.history.push({ role: "user", content: text }, { role: "assistant", content: res.assistant_text });
      live.state = res.order_state; live.stateKo = res.order_state_ko;
      const extra = res.load_s != null ? ` · 불러오기 ${res.load_s}초 · 프롬프트 ${res.prompt_tokens}토큰 · 생성 ${res.gen_tokens}토큰` : "";
      status(`${r.name}: ${res.wall_s.toFixed(2)}초 (${res.mode})${extra}`);
      $("liveText").value = "";
    } catch (e) {
      r.off = true; status("오류: " + e.message);
    } finally {
      live.busy = false; $("liveSend").disabled = false; showLiveOrder(); loop();
    }
  }

  // ---------- export ----------
  function exportScene(still = false) {
    // live scenes are exported with the same compressed timing as replays
    const s = JSON.parse(JSON.stringify(scene));
    if (s.live) { s.live = false; s.robots.forEach((r) => { if (!r.done) r.off = true; }); }
    else s.footer = still ? FOOT_STILL : FOOT_ANIM;
    return Stage.timeline(s);
  }
  function renderAt(s, t, scale) {
    const c = document.createElement("canvas"), g = c.getContext("2d");
    const lay = Stage.layout(g, s);
    c.width = Math.round(Stage.W * scale); c.height = Math.round(lay.H * scale);
    g.setTransform(scale, 0, 0, scale, 0, 0);
    Stage.draw(g, s, t, lay);
    return c;
  }
  const tick = () => new Promise((r) => setTimeout(r, 0));
  function b64(bytes) {
    let s = ""; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }
  async function save(files) {
    const res = await (await fetch("/api/export", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ case: scene.caseId, files }),
    })).json();
    if (res.error) throw new Error(res.error);
    return res;
  }
  function stamp() { const d = new Date(); return `${d.getHours()}${String(d.getMinutes()).padStart(2, "0")}${String(d.getSeconds()).padStart(2, "0")}`; }

  async function exportGif() {
    const s = exportScene(), fps = +$("expFps").value, scale = Math.min(1, +$("expScale").value);
    const n = Math.ceil(s.tEnd * fps) + 1, frames = [], delays = [];
    for (let i = 0; i < n; i++) {
      const c = renderAt(s, i / fps, scale);
      frames.push(c.getContext("2d").getImageData(0, 0, c.width, c.height));
      delays.push(Math.round(((i + 1) * 100) / fps) - Math.round((i * 100) / fps));
      if (i % 4 === 0) { status(`GIF 프레임 그리는 중 ${i + 1}/${n}`); await tick(); }
    }
    delays[delays.length - 1] += 150;  // hold the final scene
    status("GIF 압축 중…"); await tick();
    const bytes = encodeGIF(frames, delays);
    const name = `${s.caseId}_${stamp()}.gif`;
    const res = await save([{ name, data: b64(bytes) }]);
    status(`GIF 저장: <a href="/exports/${s.caseId}/${name}" target="_blank">${res.dir}/${name}</a> (${(bytes.length / 1048576).toFixed(1)} MB, ${frames.length}프레임, ${frames[0].width}×${frames[0].height})`);
    return `${res.dir}/${name}`;
  }
  async function exportFrames() {
    const s = exportScene(), fps = +$("expFps").value, scale = +$("expScale").value;
    const n = Math.ceil(s.tEnd * fps) + 1, tag = stamp();
    let files = [], saved = 0, dir = "";
    for (let i = 0; i < n; i++) {
      files.push({ name: `${s.caseId}_${tag}_${String(i).padStart(3, "0")}.png`, data: renderAt(s, i / fps, scale).toDataURL("image/png") });
      if (files.length === 10 || i === n - 1) { const r = await save(files); dir = r.dir; saved += r.files.length; files = []; status(`PNG 프레임 저장 중 ${saved}/${n}`); }
    }
    status(`PNG 프레임 ${saved}장 저장: ${dir}/ (${fps} fps, 파일 이름 ${s.caseId}_${tag}_000.png …)`);
  }
  async function exportLast() {
    const s = exportScene(true), name = `${s.caseId}_${stamp()}_final.png`;
    const res = await save([{ name, data: renderAt(s, s.tEnd, +$("expScale").value).toDataURL("image/png") }]);
    status(`마지막 장면 저장: <a href="/exports/${s.caseId}/${name}" target="_blank">${res.dir}/${name}</a>`);
  }
  function guard(fn) {
    return async () => {
      const btns = ["expGif", "expFrames", "expLast"].map($);
      btns.forEach((b) => (b.disabled = true));
      try { await fn(); } catch (e) { status("내보내기 오류: " + e.message); } finally { btns.forEach((b) => (b.disabled = false)); }
    };
  }

  // ---------- wiring ----------
  function setTab(isLive) {
    $("tabReplay").classList.toggle("on", !isLive); $("tabLive").classList.toggle("on", isLive);
    $("replayCtl").classList.toggle("hidden", isLive); $("liveCtl").classList.toggle("hidden", !isLive);
    if (isLive) { showLiveOrder(); setScene(liveScene("", null)); } else replay($("casePick").value);
  }
  $("tabReplay").onclick = () => setTab(false);
  $("tabLive").onclick = () => setTab(true);
  $("casePick").onchange = () => replay($("casePick").value);
  $("play").onclick = () => { fixedT = null; setScene(scene); };
  $("liveSend").onclick = liveSend;
  $("liveText").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.isComposing) liveSend(); });
  $("liveReset").onclick = () => { live.history = []; live.state = []; live.stateKo = []; showLiveOrder(); setScene(liveScene("", null)); status("새 주문을 시작했어요."); };
  $("expGif").onclick = guard(exportGif);
  $("expFrames").onclick = guard(exportFrames);
  $("expLast").onclick = guard(exportLast);

  (async function init() {
    try { await document.fonts.load("20px Jua"); } catch (e) { /* offline: fallback font */ }
    await loadCases();
    const id = params.get("case");
    if (id) $("casePick").value = id;
    const t = params.get("t");
    if (t != null) fixedT = t === "end" ? 1e9 : +t;
    await replay($("casePick").value);
    if (fixedT != null && scene) { fixedT = Math.min(fixedT, scene.tEnd); frame(); }
    const ex = { gif: exportGif, frames: exportFrames, last: exportLast }[params.get("export")];
    if (ex) { await guard(ex)(); document.title = "exported"; }  // e.g. ?case=L09&export=gif
    if (params.get("tab") === "live") {  // e.g. ?tab=live&robot=e4b&say=아아 두 잔이요
      fixedT = null; setTab(true);
      if (params.get("robot")) $("liveRobot").value = params.get("robot");
      if (params.get("say")) { $("liveText").value = params.get("say"); await liveSend(); }
    }
    document.body.dataset.ready = "1";
  })();
})();
