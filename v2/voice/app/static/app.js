// Voice ordering demo page. Mic is opened only while recording and released right after
// (all tracks stopped, AudioContext closed) so Bluetooth earbuds can leave call mode.
const $ = (id) => document.getElementById(id);
const TAIL_MS = 300;          // keep recording this long after release
const MIN_MS = 350;           // ignore shorter presses
let session = null, busy = false, lastAudio = [], player = null;
let queue = [], turnSeq = 0;               // D3: sentences of the current reply still to play (first one plays at once)
let rec = null;               // {stream, recorder, chunks, ctx, timer, start, cancelled}
let pressing = false, opening = false;

async function api(path, body, raw) {
  const opt = { method: body === undefined ? "GET" : "POST" };
  if (raw) { opt.body = body; opt.headers = { "Content-Type": raw }; }
  else if (body !== undefined) { opt.body = JSON.stringify(body); opt.headers = { "Content-Type": "application/json" }; }
  const r = await fetch(path, opt);
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || d.error || ("HTTP " + r.status));
  return d;
}
const fmt = (v) => (v === null || v === undefined) ? "-" : v.toFixed(2) + "s";
function say(m) { $("msg").textContent = m || ""; }
function notice(m) {
  const n = $("notice"); n.textContent = m; n.classList.remove("hidden");
  clearTimeout(notice.t); notice.t = setTimeout(() => n.classList.add("hidden"), 4000);
}

async function newSession(showNotice) {
  turnSeq++; stopAudio();
  const prev = session;
  session = (await api("/api/new_session", { previous: prev })).session;
  $("heard").textContent = "버튼을 누르고 주문해 보세요"; $("heard").className = "bubble empty";
  $("reply").textContent = "…"; $("reply").className = "bubble robot empty";
  $("calls").textContent = "-"; $("staff").classList.add("hidden"); $("confirmed").classList.add("hidden");
  $("kindTag").classList.add("hidden");
  $("history").innerHTML = '<span class="hint">아직 대화가 없어요</span>';
  renderOrder([]); ["tStt", "tLlm", "tTts", "tFirst", "tTotal", "tClient"].forEach(i => $(i).textContent = "-");
  $("tokinfo").textContent = ""; $("inputKind").textContent = ""; $("replay").classList.add("hidden");
  lastAudio = []; say("");
  if (showNotice) notice("🧑 새 손님이에요. 주문 내역과 대화를 모두 비웠어요.");
}

function renderOrder(lines) {
  const ul = $("order"); ul.innerHTML = "";
  if (!lines.length) { ul.innerHTML = '<li class="none">비어 있어요</li>'; return; }
  lines.forEach((t, i) => { const li = document.createElement("li");
    li.innerHTML = `<span class="n">${i + 1}.</span>`; li.append(document.createTextNode(t)); ul.append(li); });
}

function stopAudio() { queue = []; if (player) { player.pause(); player.currentTime = 0; } }
function playNext() {
  if (!queue.length) { player = null; return; }
  player = new Audio(queue.shift());
  player.onended = playNext;
  player.play().catch(e => say("자동 재생이 막혔어요: " + e.message));
}
function play(srcs) {            // one data URL or a list (sentences played in order)
  stopAudio();
  queue = [].concat(srcs || []).filter(Boolean);
  playNext();
}
function enqueue(src) {          // a later sentence of the reply that is playing (or has just ended)
  if (!src) return;
  queue.push(src);
  if (!player || player.ended) playNext();
}

// D3: a turn answers as NDJSON: the turn with the first sentence's audio, then one line per further
// sentence, then {"done": true, "timings": ...}. The first line is shown and played at once, later ones queued.
// turnSeq changes on barge-in, new customer and every new turn, so late sentences of an old reply are dropped.
async function turnApi(path, body, raw, t0) {
  const opt = { method: "POST" };
  if (raw) { opt.body = body; opt.headers = { "Content-Type": raw }; }
  else { opt.body = JSON.stringify(body); opt.headers = { "Content-Type": "application/json" }; }
  const r = await fetch(path + (path.includes("?") ? "&" : "?") + "stream=1", opt);
  if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.detail || ("HTTP " + r.status)); }
  if (!(r.headers.get("content-type") || "").includes("ndjson")) { show(await r.json(), t0); return; }
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = "", first = true; const seq = ++turnSeq;
  const handle = (line) => {
    if (!line.trim()) return;
    const d = JSON.parse(line);
    if (first) { first = false; show(d, t0); setBusy(false); return; }   // the customer may talk again now
    if (seq !== turnSeq) return;   // a newer turn started: drop this reply's later sentences
    if (d.audio) { lastAudio.push(d.audio); enqueue(d.audio); }
    if (d.done) { $("tTotal").textContent = fmt(d.timings.total_s); $("tClient").textContent = fmt((performance.now() - t0) / 1000);
                  if (d.error) say(d.error); }
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) { handle(buf.slice(0, i)); buf = buf.slice(i + 1); }
  }
  handle(buf);
}

const KIND_KO = { readback: "주문 확인 질문", ask_dine: "매장/포장 질문", confirm: "주문 확정", locked: "확정 후 변경 → 직원",
                  read_order: "주문 내역 (주문 상태에서 읽음)", inform: "메뉴 안내 (menu.json)", free_text: "도구 없이 말로 답함",
                  free_text_blocked: "메뉴에 없는 이름이 있어 읽지 않음", fallback: "다시 말해 달라고 함", no_speech: "말소리 없음" };

function show(d, t0) {
  $("heard").textContent = d.transcript || "(아무 말도 인식되지 않았어요)";
  $("heard").className = "bubble" + (d.transcript ? "" : " empty");
  $("inputKind").textContent = d.input === "audio" ? `🎙️ 음성 ${d.timings.audio_in_s ? d.timings.audio_in_s.toFixed(1) + "초" : ""}` : "⌨️ 글자";
  $("reply").textContent = d.reply; $("reply").className = "bubble robot";
  const tag = d.kinds.map(k => KIND_KO[k]).find(Boolean);
  $("kindTag").textContent = tag || ""; $("kindTag").classList.toggle("hidden", !tag);
  $("calls").textContent = JSON.stringify({ tool_calls: d.tool_calls, content: d.content || undefined }, null, 2);
  renderOrder(d.order);
  // order flow: 주문 중 -> 확인 대기 (read back, waiting for yes / dine option) -> 확정 (locked)
  const dine = d.dine_said === "takeout" ? "포장" : d.dine_said === "dine_in" ? "매장" : "매장/포장 미정";
  if (d.confirmed) {
    $("confirmed").textContent = d.dine_option === "takeout" ? "✅ 주문 확정 · 포장 (변경은 직원)" : "✅ 주문 확정 · 매장 (변경은 직원)";
    $("confirmed").className = "badge done";
  } else if (d.phase === "awaiting") {
    $("confirmed").textContent = `⏳ 확인 대기 · ${dine}`; $("confirmed").className = "badge wait";
  } else { $("confirmed").className = "badge hidden"; }
  if (d.staff_called) { $("staffReason").textContent = d.staff_reason; $("staff").classList.remove("hidden"); }
  const t = d.timings;
  $("tStt").textContent = fmt(t.stt_s); $("tLlm").textContent = fmt(t.llm_s); $("tTts").textContent = fmt(t.tts_s);
  $("tFirst").textContent = fmt(t.first_audio_s);
  $("tTotal").textContent = fmt(t.total_s); $("tClient").textContent = fmt((performance.now() - t0) / 1000);
  $("tokinfo").textContent = t.prompt_tokens ? `프롬프트 ${t.prompt_tokens}토큰 (처리 ${fmt(t.prompt_s)}) · 생성 ${t.gen_tokens}토큰` : "";
  const h = $("history"); if (h.querySelector(".hint")) h.innerHTML = "";
  if (d.transcript) { const u = document.createElement("div"); u.className = "u"; u.textContent = "🧑 " + d.transcript; h.append(u); }
  const a = document.createElement("div"); a.className = "a"; a.textContent = "🤖 " + d.reply; h.append(a);
  h.scrollTop = h.scrollHeight;
  say(d.error || "");
  lastAudio = d.audio ? [d.audio] : []; $("replay").classList.toggle("hidden", !d.audio);
  play(d.audio);
}

function setTalk(state) {   // idle | prep | rec | busy
  const t = $("talk");
  t.classList.toggle("prep", state === "prep"); t.classList.toggle("rec", state === "rec");
  t.classList.toggle("busy", state === "busy");
  const toggle = $("toggleMode").checked;
  $("talkLabel").textContent = { idle: toggle ? "눌러서 말하기 시작" : "누르고 있는 동안 말하기",
    prep: "마이크 켜는 중…", rec: toggle ? "다시 누르면 끝" : "떼면 전송", busy: "생각하는 중…" }[state];
}
function setBusy(b) { busy = b; $("send").disabled = b; setTalk(b ? "busy" : "idle"); }

async function sendText() {
  const text = $("textIn").value.trim(); if (!text || busy) return;
  setBusy(true); const t0 = performance.now();
  try { await turnApi("/api/turn/text", { session, text }, null, t0); $("textIn").value = ""; }
  catch (e) { say("오류: " + e.message); } finally { setBusy(false); }
}

// ---------- recording (mic open only while recording) ----------
function release(r) {
  clearInterval(r.timer);
  r.stream.getTracks().forEach(tr => tr.stop());
  if (r.ctx) r.ctx.close().catch(() => {});
  $("lvl").style.width = "0";
}

async function startRec() {
  if (busy || rec || opening) return;
  stopAudio();
  opening = true; setTalk("prep"); say("");
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch (e) { opening = false; setTalk("idle"); say("마이크를 쓸 수 없어요: " + e.message); return; }
  opening = false;
  if (!pressing && !$("toggleMode").checked) {   // released while the mic (or permission prompt) was opening
    stream.getTracks().forEach(tr => tr.stop()); setTalk("idle"); say("마이크가 준비됐어요. 다시 누르고 말해 주세요."); return;
  }
  const type = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "";
  const recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
  const r = { stream, recorder, chunks: [], ctx: null, timer: null, start: 0 };
  recorder.ondataavailable = (e) => e.data.size && r.chunks.push(e.data);
  try {
    r.ctx = new AudioContext(); const an = r.ctx.createAnalyser(); an.fftSize = 1024;
    r.ctx.createMediaStreamSource(stream).connect(an);
    const buf = new Float32Array(an.fftSize);
    r.timer = setInterval(() => { an.getFloatTimeDomainData(buf); let p = 0; for (const v of buf) p = Math.max(p, Math.abs(v));
      $("lvl").style.width = Math.min(100, Math.round(p * 140)) + "%"; }, 60);
  } catch (e) { /* level meter is optional */ }
  recorder.start(); r.start = performance.now(); rec = r;
  setTalk("rec");
}

async function stopRec() {
  if (!rec) return;
  const r = rec; rec = null;
  await new Promise(res => setTimeout(res, TAIL_MS));          // keep the end of the last word
  await new Promise(res => { r.recorder.onstop = res; r.recorder.stop(); });
  release(r);
  const dur = performance.now() - r.start;
  if (dur < MIN_MS + TAIL_MS) { setTalk("idle"); say("너무 짧아요. 말하는 동안 버튼을 누르고 있어 주세요."); return; }
  const blob = new Blob(r.chunks, { type: r.recorder.mimeType || "audio/webm" });
  setBusy(true); const t0 = performance.now();
  try { await turnApi("/api/turn/audio?session=" + session, blob, blob.type || "audio/webm", t0); }
  catch (e) { say("오류: " + e.message); } finally { setBusy(false); }
}

function bargeIn() {   // customer starts talking: stop the robot's voice at once
  turnSeq++; stopAudio();
}

function press() {
  bargeIn();
  if ($("toggleMode").checked) { if (rec) stopRec(); else startRec(); return; }
  pressing = true; startRec();
}
function unpress() {
  if ($("toggleMode").checked) return;
  pressing = false; stopRec();
}

const talk = $("talk");
talk.addEventListener("pointerdown", (e) => { e.preventDefault(); talk.setPointerCapture(e.pointerId); press(); });
["pointerup", "pointercancel"].forEach(ev => talk.addEventListener(ev, unpress));
const typing = () => document.activeElement && ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)
  && document.activeElement.type !== "checkbox";
window.addEventListener("keydown", (e) => { if (e.code === "Space" && !typing()) { e.preventDefault(); if (!e.repeat) press(); } });
window.addEventListener("keyup", (e) => { if (e.code === "Space" && !typing()) { e.preventDefault(); unpress(); } });
window.addEventListener("blur", () => { if (!$("toggleMode").checked) unpress(); });
$("toggleMode").addEventListener("change", () => {
  try { localStorage.setItem("toggleMode", $("toggleMode").checked ? "1" : "0"); } catch (e) {}
  $("talkHint").textContent = $("toggleMode").checked ? "한 번 누르면 녹음 시작, 다시 누르면 끝 (Space도 같음)"
                                                      : "마우스 버튼 또는 Space 키를 누르고 있는 동안 녹음";
  if (!rec) setTalk(busy ? "busy" : "idle");
});

// ---------- voice settings ----------
function fillSid(n) {
  const s = $("sid"); s.innerHTML = "";
  for (let i = 0; i < n; i++) { const o = document.createElement("option"); o.value = i;
    o.textContent = `화자 ${i} (${i < 5 ? "여성" : "남성"} 추정)`; s.append(o); }
}
function syncVoiceUi() { const sup = $("engine").value === "supertonic"; $("sid").disabled = !sup; $("steps").disabled = !sup; }
async function saveVoice() {
  syncVoiceUi();
  try { await api("/api/config", { engine: $("engine").value, sid: +$("sid").value, steps: +$("steps").value }); }
  catch (e) { say("설정 저장 실패: " + e.message); }
}
["engine", "sid", "steps"].forEach(id => $(id).addEventListener("change", saveVoice));
$("preview").addEventListener("click", async () => {
  $("preview").disabled = true;
  try { const d = await api("/api/tts/preview", { engine: $("engine").value, sid: +$("sid").value, steps: +$("steps").value });
        play(d.audio); say(`미리 듣기 합성 ${d.tts_s.toFixed(2)}s`); }
  catch (e) { say("미리 듣기 실패: " + e.message); } finally { $("preview").disabled = false; }
});

$("send").addEventListener("click", sendText);
$("textIn").addEventListener("keydown", (e) => { if (e.key === "Enter") sendText(); });
$("reset").addEventListener("click", () => newSession(true));
$("replay").addEventListener("click", () => play(lastAudio));

async function poll() {
  try {
    const h = await api("/api/health");
    const chip = (id, ok, txt) => { $(id).className = "chip " + (ok ? "ok" : "wait"); $(id).textContent = txt; };
    chip("stStt", h.stt, h.stt ? "STT 준비됨" : "STT 준비 중");
    chip("stLlm", h.llm, h.llm ? "64e 준비됨" + (h.llm_warmup && h.llm_warmup.full_igpu ? " · iGPU" : "") : (h.ollama_up ? "64e 불러오는 중" : "Ollama 꺼짐"));
    if (!h.ollama_up) $("stLlm").className = "chip bad";
    chip("stTts", h.tts, h.tts ? "TTS 준비됨" : "TTS 준비 중");
    if (!h.ready) setTimeout(poll, 1500);
  } catch (e) { setTimeout(poll, 2000); }
}

(async function init() {
  fillSid(10);
  try { $("toggleMode").checked = localStorage.getItem("toggleMode") === "1"; } catch (e) {}
  $("toggleMode").dispatchEvent(new Event("change"));
  try { const c = await api("/api/config"); $("engine").value = c.engine; $("sid").value = c.sid; $("steps").value = c.steps; } catch (e) {}
  syncVoiceUi(); await newSession(false); poll();
})();
