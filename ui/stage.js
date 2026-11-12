// Stage renderer: everything is a pure function of (scene, t) so the same code draws the live page,
// screenshots and exported frames. Canvas is 1100 px wide; height depends on the case.
(function (global) {
  "use strict";

  const W = 1100, M = 12, COL_W = 204, GAP = 14;
  const FONT = "'Jua', 'Malgun Gothic', sans-serif";
  const INK = "#2b2118", PAPER = "#fffaf0", BG = "#ffe9b8";
  const C = {
    red: "#e63946", green: "#2a9d5c", blue: "#3a86ff", orange: "#f77f00", purple: "#8e5bd8",
    teal: "#119c9c", gray: "#8a8178", mint: "#7cffb2", soft: "#fff3d1",
  };
  const ROBOT_COLORS = { vending: "#79b6e0", e4b: "#ffd166", "12b": "#7bd389", "26b": "#ff9f5a", "64e": "#c3a3ff" };
  const KIND = {
    add: ["담기", C.green], modify: ["변경", C.teal], remove: ["빼기", C.red], ask: ["질문", C.blue],
    staff: ["직원", C.orange], confirm: ["확정", C.purple], talk: ["말만", C.gray], silent: ["침묵", C.gray],
  };

  // ---------- timing ----------
  const T_SAY = 0.0, T_THINK = 0.75;
  // real seconds -> animation seconds (log scale keeps the order: 0.002 s ~ 0.25 s, 2 s ~ 1.1 s, 70 s ~ 2.4 s)
  const scaled = (w) => 0.25 + 0.9 * Math.log10(1 + 4 * Math.max(0, w));
  const unscaled = (a) => (a <= 0.25 ? 0 : (Math.pow(10, (a - 0.25) / 0.9) - 1) / 4);

  function timeline(scene) {
    let last = T_THINK;
    for (const r of scene.robots) {
      if (r.off || r.missing) continue;
      if (scene.live) { r.tDone = r.done ? Math.max(r.wall_s, 0.6) : Infinity; }
      else r.tDone = T_THINK + scaled(r.wall_s);
      if (isFinite(r.tDone)) last = Math.max(last, r.tDone);
    }
    scene.tExpected = last + 0.75;
    scene.tEnd = last + (scene.live ? 2.2 : 3.0);
    return scene;
  }

  // ---------- helpers ----------
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const easeBack = (x) => { x = clamp(x, 0, 1); const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2; };
  const font = (px) => `${px}px ${FONT}`;

  function rr(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function panel(ctx, x, y, w, h, fill, r = 14, shadow = 5) {
    ctx.save();
    rr(ctx, x + shadow, y + shadow, w, h, r); ctx.fillStyle = INK; ctx.fill();
    rr(ctx, x, y, w, h, r); ctx.fillStyle = fill; ctx.fill();
    ctx.lineWidth = 3; ctx.strokeStyle = INK; ctx.stroke();
    ctx.restore();
  }
  function shape(ctx, fill, lw = 3) { ctx.fillStyle = fill; ctx.fill(); ctx.lineWidth = lw; ctx.strokeStyle = INK; ctx.stroke(); }
  function circle(ctx, x, y, r) { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); }

  function wrap(ctx, text, maxW, indent = 0) {
    // word wrap with character fallback (Korean words can be long)
    const out = [];
    let line = "", width = maxW - indent;
    const push = () => { out.push(line); line = ""; width = maxW; };
    for (const word of String(text).split(/(\s+)/)) {
      if (!word) continue;
      const cand = line + word;
      if (ctx.measureText(cand).width <= width) { line = cand; continue; }
      if (/^\s+$/.test(word)) { push(); continue; }
      if (line.trim()) push();
      for (const ch of word) {
        if (ctx.measureText(line + ch).width > width && line) push();
        line += ch;
      }
    }
    if (line.trim() || !out.length) out.push(line);
    return out.map((s) => s.trim());
  }

  let pattern = null;
  function dots(ctx) {
    if (pattern) return pattern;
    const c = document.createElement("canvas"); c.width = c.height = 14;
    const g = c.getContext("2d"); g.fillStyle = "#fff3d6"; g.fillRect(0, 0, 14, 14);
    g.fillStyle = "#ffe2a6"; g.beginPath(); g.arc(3.5, 3.5, 2, 0, 7); g.arc(10.5, 10.5, 2, 0, 7); g.fill();
    pattern = ctx.createPattern(c, "repeat");
    return pattern;
  }

  // ---------- faces ----------
  function eyes(ctx, cx, cy, gap, r, st, t, color = INK) {
    ctx.save(); ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineCap = "round"; ctx.lineWidth = 3;
    for (const s of [-1, 1]) {
      const x = cx + s * gap / 2;
      if (st === "right" || st === "answered") {
        ctx.beginPath(); ctx.arc(x, cy + r * 0.4, r, Math.PI * 1.1, Math.PI * 1.9); ctx.stroke();
      } else if (st === "wrong") {
        // ">" for the left eye, "<" for the right eye
        ctx.beginPath(); ctx.moveTo(x - s * r, cy - r); ctx.lineTo(x + s * r * 0.6, cy); ctx.lineTo(x - s * r, cy + r); ctx.stroke();
      } else if (st === "off" || (st === "idle" && (t % 3.3) < 0.13)) {
        ctx.beginPath(); ctx.moveTo(x - r, cy); ctx.lineTo(x + r, cy); ctx.stroke();
      } else {
        circle(ctx, x, cy, r); ctx.fill();
        if (st === "thinking") {  // look up, glint
          ctx.fillStyle = "#fff"; circle(ctx, x + r * 0.35, cy - r * 0.45, r * 0.38); ctx.fill(); ctx.fillStyle = color;
        } else { ctx.fillStyle = "#fff"; circle(ctx, x - r * 0.3, cy - r * 0.35, r * 0.3); ctx.fill(); ctx.fillStyle = color; }
      }
    }
    ctx.restore();
  }
  function mouth(ctx, cx, cy, w, st, t, color = INK) {
    ctx.save(); ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3; ctx.lineCap = "round";
    if (st === "right") {
      ctx.beginPath(); ctx.moveTo(cx - w / 2, cy - 2); ctx.arc(cx, cy - 2, w / 2, 0, Math.PI); ctx.closePath(); ctx.fill();
      ctx.fillStyle = "#ff8fa3"; ctx.beginPath(); ctx.arc(cx, cy + w * 0.28, w * 0.22, Math.PI, 0); ctx.fill();
    } else if (st === "wrong") {
      ctx.beginPath(); const n = 4;
      for (let i = 0; i <= n; i++) ctx.lineTo(cx - w / 2 + (w * i) / n, cy + (i % 2 ? -3 : 3));
      ctx.stroke();
    } else if (st === "thinking") {
      ctx.beginPath(); ctx.ellipse(cx + 3, cy, 4, 3 + Math.abs(Math.sin(t * 6)) * 2, 0, 0, 7); ctx.stroke();
    } else if (st === "off") {
      ctx.beginPath(); ctx.moveTo(cx - w / 4, cy); ctx.lineTo(cx + w / 4, cy); ctx.stroke();
    } else {
      ctx.beginPath(); ctx.arc(cx, cy - w * 0.25, w * 0.4, Math.PI * 0.2, Math.PI * 0.8); ctx.stroke();
    }
    ctx.restore();
  }
  function blush(ctx, cx, cy, gap) {
    ctx.save(); ctx.fillStyle = "rgba(255,120,140,.45)";
    for (const s of [-1, 1]) { ctx.beginPath(); ctx.ellipse(cx + s * gap, cy, 8, 5, 0, 0, 7); ctx.fill(); }
    ctx.restore();
  }
  function sweat(ctx, x, y, t) {
    ctx.save(); const dy = (t * 18) % 10;
    ctx.beginPath(); ctx.moveTo(x, y + dy); ctx.quadraticCurveTo(x + 8, y + 14 + dy, x, y + 18 + dy);
    ctx.quadraticCurveTo(x - 8, y + 14 + dy, x, y + dy); shape(ctx, "#9ad8ff", 2); ctx.restore();
  }
  function antenna(ctx, x, y, len, t, st, color) {
    ctx.save(); ctx.lineWidth = 3; ctx.strokeStyle = INK;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y - len); ctx.stroke();
    const on = st === "thinking" ? (Math.floor(t * 5) % 2 === 0) : st === "right";
    circle(ctx, x, y - len - 6, 7); shape(ctx, on ? "#fff06a" : color, 3);
    if (on) { ctx.strokeStyle = "rgba(255,200,0,.7)"; ctx.lineWidth = 2; circle(ctx, x, y - len - 6, 12); ctx.stroke(); }
    ctx.restore();
  }
  function zz(ctx, x, y, t) {
    ctx.save(); ctx.font = font(18); ctx.fillStyle = C.gray;
    ctx.fillText("z", x, y - (t * 6) % 8); ctx.font = font(14); ctx.fillText("z", x + 12, y - 14 - (t * 6) % 8); ctx.restore();
  }

  // ---------- robots (original drawings) ----------
  function face(ctx, cx, cy, gap, er, mw, st, t, color = INK, cheeks = true) {
    eyes(ctx, cx, cy, gap, er, st, t, color);
    mouth(ctx, cx, cy + er + 13, mw, st, t, color);
    if (cheeks && (st === "right" || st === "answered")) blush(ctx, cx, cy + er + 6, gap * 0.85);
  }

  const DRAW = {
    vending(ctx, cx, by, st, t, col) {
      const w = 104, h = 136, x = cx - w / 2, y = by - h;
      rr(ctx, x, y, w, h, 12); shape(ctx, col);
      rr(ctx, x + 12, y + 12, w - 24, 52, 8); shape(ctx, "#1d2b36");
      face(ctx, cx, y + 30, 34, 7, 18, st, t, C.mint, false);
      const bc = ["#ff6b6b", "#ffd166", "#06d6a0", "#118ab2", "#ef476f", "#ffffff"];
      for (let i = 0; i < 6; i++) { circle(ctx, x + 22 + (i % 3) * 22, y + 80 + Math.floor(i / 3) * 18, 6); shape(ctx, bc[i], 2); }
      rr(ctx, x + w - 24, y + 74, 10, 26, 4); shape(ctx, "#1d2b36", 2);
      rr(ctx, x + 18, y + h - 24, w - 36, 14, 5); shape(ctx, st === "right" || st === "answered" ? "#fff06a" : "#1d2b36", 2);
      if (st === "thinking") {  // little working light
        circle(ctx, x + w - 12, y + 8, 5); shape(ctx, Math.floor(t * 6) % 2 ? "#ff6b6b" : "#ffd166", 2);
      }
      if (st === "wrong") sweat(ctx, x + w + 4, y + 8, t);
      if (st === "off") zz(ctx, x + w - 6, y - 4, t);
    },
    e4b(ctx, cx, by, st, t, col) {
      const r = 40, cy = by - 14 - r;
      ctx.lineWidth = 3;
      for (const s of [-1, 1]) { rr(ctx, cx + s * 16 - 7, by - 18, 14, 18, 5); shape(ctx, "#7a6a55"); }
      antenna(ctx, cx, cy - r + 2, 16, t, st, col);
      circle(ctx, cx, cy, r); shape(ctx, col);
      for (const s of [-1, 1]) { circle(ctx, cx + s * (r + 4), cy + 8, 9); shape(ctx, col); }
      ctx.beginPath(); ctx.ellipse(cx, cy - 4, 27, 20, 0, 0, 7); shape(ctx, "#fff", 2.5);
      face(ctx, cx, cy - 9, 22, 5, 12, st, t);
      if (st === "wrong") sweat(ctx, cx + r - 2, cy - r, t);
      if (st === "off") zz(ctx, cx + r, cy - r, t);
    },
    "12b"(ctx, cx, by, st, t, col) {
      const bw = 74, bh = 44, hw = 96, hh = 70;
      for (const s of [-1, 1]) { rr(ctx, cx + s * 20 - 9, by - 16, 18, 16, 4); shape(ctx, "#5c6b73"); }
      rr(ctx, cx - bw / 2, by - 14 - bh, bw, bh, 10); shape(ctx, col);
      rr(ctx, cx - 18, by - 14 - bh + 10, 36, 20, 5); shape(ctx, "#eafff0", 2);
      for (let i = 0; i < 3; i++) { circle(ctx, cx - 10 + i * 10, by - 14 - bh + 20, 3); shape(ctx, ["#ff6b6b", "#ffd166", "#3a86ff"][i], 1.5); }
      const hy = by - 14 - bh - hh + 6;
      antenna(ctx, cx - 26, hy + 2, 14, t + 0.3, st, col); antenna(ctx, cx + 26, hy + 2, 14, t, st, col);
      rr(ctx, cx - hw / 2, hy, hw, hh, 16); shape(ctx, col);
      rr(ctx, cx - hw / 2 + 10, hy + 10, hw - 20, hh - 20, 10); shape(ctx, "#fff", 2.5);
      face(ctx, cx, hy + 28, 30, 6, 14, st, t);
      if (st === "wrong") sweat(ctx, cx + hw / 2 + 2, hy, t);
      if (st === "off") zz(ctx, cx + hw / 2, hy - 2, t);
    },
    "26b"(ctx, cx, by, st, t, col, pruned) {
      const bw = pruned ? 78 : 104, bh = 50, hw = 116, hh = 76;
      for (const s of [-1, 1]) { rr(ctx, cx + s * 26 - 11, by - 16, 22, 16, 5); shape(ctx, "#5b4b3a"); }
      // arms
      for (const s of [-1, 1]) { rr(ctx, cx + s * (bw / 2 + 6) - 9, by - 14 - bh + 6, 18, 36, 8); shape(ctx, col); }
      rr(ctx, cx - bw / 2, by - 14 - bh, bw, bh, 12); shape(ctx, col);
      if (pruned) {  // scissors badge = "pruned"
        const sx = cx, sy = by - 14 - bh / 2;
        circle(ctx, sx, sy, 15); shape(ctx, "#fff", 2.5);
        ctx.save(); ctx.lineWidth = 2.5; ctx.strokeStyle = INK;
        ctx.beginPath(); ctx.moveTo(sx - 6, sy - 8); ctx.lineTo(sx + 5, sy + 4); ctx.moveTo(sx + 6, sy - 8); ctx.lineTo(sx - 5, sy + 4); ctx.stroke();
        circle(ctx, sx - 6, sy + 7, 3.5); ctx.stroke(); circle(ctx, sx + 6, sy + 7, 3.5); ctx.stroke(); ctx.restore();
      } else {
        for (let i = 0; i < 4; i++) {
          const on = st === "thinking" ? Math.floor(t * 6 + i) % 4 === 0 : i % 2 === 0;
          rr(ctx, cx - 38 + i * 20, by - 14 - bh + 16, 14, 14, 3); shape(ctx, on ? "#fff06a" : "#8a5a2b", 2);
        }
      }
      const hy = by - 14 - bh - hh + 4;
      // ear dials
      for (const s of [-1, 1]) { circle(ctx, cx + s * (hw / 2 + 4), hy + hh / 2, 12); shape(ctx, "#ffd8a8"); }
      if (pruned) {  // a little sprout on top
        ctx.save(); ctx.lineWidth = 3; ctx.strokeStyle = INK;
        ctx.beginPath(); ctx.moveTo(cx, hy); ctx.lineTo(cx, hy - 16); ctx.stroke();
        const wob = st === "thinking" ? Math.sin(t * 8) * 0.25 : 0;
        for (const s of [-1, 1]) {
          ctx.save(); ctx.translate(cx, hy - 14); ctx.rotate(s * (0.6 + wob));
          ctx.beginPath(); ctx.ellipse(0, -9, 6, 11, 0, 0, 7); shape(ctx, "#6fd36f", 2.5); ctx.restore();
        }
        ctx.restore();
      } else antenna(ctx, cx, hy + 2, 14, t, st, col);
      rr(ctx, cx - hw / 2, hy, hw, hh, 18); shape(ctx, col);
      rr(ctx, cx - hw / 2 + 12, hy + 11, hw - 24, hh - 22, 12); shape(ctx, "#fff", 2.5);
      face(ctx, cx, hy + 31, 36, 6.5, 16, st, t);
      if (st === "wrong") sweat(ctx, cx + hw / 2 + 8, hy - 4, t);
      if (st === "off") zz(ctx, cx + hw / 2 + 6, hy - 4, t);
    },
    "64e"(ctx, cx, by, st, t, col) { DRAW["26b"](ctx, cx, by, st, t, col, true); },
  };

  function customer(ctx, cx, by, t, talking) {
    // body
    ctx.save();
    ctx.beginPath(); ctx.moveTo(cx - 52, by); ctx.quadraticCurveTo(cx - 50, by - 62, cx, by - 64);
    ctx.quadraticCurveTo(cx + 50, by - 62, cx + 52, by); ctx.closePath(); shape(ctx, "#5aa9e6");
    ctx.beginPath(); ctx.moveTo(cx - 30, by - 58); ctx.quadraticCurveTo(cx, by - 40, cx + 30, by - 58);
    ctx.lineTo(cx + 26, by - 66); ctx.quadraticCurveTo(cx, by - 54, cx - 26, by - 66); ctx.closePath(); shape(ctx, "#ef476f", 2.5);
    // head
    const hy = by - 108;
    circle(ctx, cx, hy, 42); shape(ctx, "#ffd8b1");
    ctx.beginPath(); ctx.moveTo(cx - 43, hy - 2); ctx.quadraticCurveTo(cx - 44, hy - 46, cx, hy - 44);
    ctx.quadraticCurveTo(cx + 44, hy - 46, cx + 43, hy - 2); ctx.quadraticCurveTo(cx + 30, hy - 26, cx + 6, hy - 22);
    ctx.quadraticCurveTo(cx - 20, hy - 30, cx - 43, hy - 2); ctx.closePath(); shape(ctx, "#6b4226");
    eyes(ctx, cx, hy + 4, 30, 5, (t % 3.7) < 0.12 ? "off" : "idle", 0.5);
    blush(ctx, cx, hy + 16, 24);
    const open = talking && Math.floor(t * 8) % 2 === 0;
    ctx.beginPath();
    if (open) { ctx.ellipse(cx, hy + 22, 7, 6, 0, 0, 7); shape(ctx, "#a33a3a", 2.5); }
    else { ctx.arc(cx, hy + 15, 9, Math.PI * 0.2, Math.PI * 0.8); ctx.lineWidth = 3; ctx.strokeStyle = INK; ctx.stroke(); }
    ctx.restore();
  }

  // ---------- bubbles ----------
  const BUB_W = COL_W - 20, TXT_W = BUB_W - 22, LH = 22, NOTE_LH = 17;

  function layoutBubble(ctx, lines) {
    const rows = [];
    let h = 12;
    for (const ln of lines) {
      ctx.font = font(13);
      const chipW = ctx.measureText(KIND[ln.kind][0]).width + 14;
      ctx.font = font(16);
      const txt = ln.kind === "ask" || ln.kind === "talk" ? `“${ln.text}”` : ln.text;
      const ws = wrap(ctx, txt, TXT_W, chipW + 6);
      ctx.font = font(12.5);
      const ns = ln.note ? wrap(ctx, ln.note, TXT_W) : [];
      rows.push({ ln, chipW, ws, ns });
      h += ws.length * LH + ns.length * NOTE_LH + 8;
    }
    return { rows, h: h + 4 };
  }

  function drawBubble(ctx, x, y, b, tailX, dashed, pop) {
    const h = b.h;
    ctx.save();
    const s = 0.55 + 0.45 * easeBack(pop);
    ctx.translate(tailX, y); ctx.scale(s, s); ctx.translate(-tailX, -y);
    ctx.globalAlpha = clamp(pop * 3, 0, 1);
    rr(ctx, x, y + 12, BUB_W, h, 14);
    ctx.fillStyle = dashed ? "#f4f1ec" : "#fff"; ctx.fill();
    ctx.lineWidth = 3; ctx.strokeStyle = INK; if (dashed) ctx.setLineDash([7, 5]); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(tailX - 10, y + 13.5); ctx.lineTo(tailX, y); ctx.lineTo(tailX + 10, y + 13.5);
    ctx.fillStyle = dashed ? "#f4f1ec" : "#fff"; ctx.fill(); ctx.stroke();
    let cy = y + 12 + 12;
    for (const r of b.rows) {
      const [label, color] = KIND[r.ln.kind];
      rr(ctx, x + 11, cy + 1, r.chipW, 19, 9); ctx.fillStyle = color; ctx.fill();
      ctx.fillStyle = "#fff"; ctx.font = font(13); ctx.textBaseline = "middle"; ctx.fillText(label, x + 18, cy + 11);
      ctx.fillStyle = r.ln.kind === "talk" || r.ln.kind === "silent" ? "#5d554d" : INK; ctx.font = font(16);
      r.ws.forEach((w, i) => ctx.fillText(w, x + 11 + (i === 0 ? r.chipW + 6 : 0), cy + 11 + i * LH));
      cy += r.ws.length * LH;
      ctx.fillStyle = "#857a6e"; ctx.font = font(12.5);
      r.ns.forEach((w, i) => ctx.fillText(w, x + 11, cy + 8 + i * NOTE_LH));
      cy += r.ns.length * NOTE_LH + 8;
    }
    ctx.restore();
  }

  function thinkingCloud(ctx, cx, y, t, t0) {
    const a = clamp((t - t0) / 0.2, 0, 1);
    ctx.save(); ctx.globalAlpha = a;
    const puffs = [[-34, 22, 18], [-12, 14, 21], [14, 15, 20], [36, 23, 17], [-18, 34, 17], [16, 35, 17]];
    ctx.lineWidth = 6; ctx.strokeStyle = INK;
    for (const [dx, dy, r] of puffs) { circle(ctx, cx + dx, y + dy + 14, r); ctx.stroke(); }
    ctx.fillStyle = "#fff";
    for (const [dx, dy, r] of puffs) { circle(ctx, cx + dx, y + dy + 14, r); ctx.fill(); }
    circle(ctx, cx - 4, y + 4, 5); shape(ctx, "#fff", 2.5);
    for (let i = 0; i < 3; i++) {
      const bounce = Math.max(0, Math.sin((t - t0) * 9 - i * 0.9)) * 6;
      circle(ctx, cx - 18 + i * 18, y + 41 - bounce, 5.5); ctx.fillStyle = INK; ctx.fill();
    }
    ctx.restore();
  }

  function stamp(ctx, x, y, ok, p) {
    if (p <= 0) return;
    const s = 1 + 0.9 * (1 - easeBack(clamp(p, 0, 1)));
    ctx.save(); ctx.translate(x, y); ctx.rotate(-0.18); ctx.scale(s, s); ctx.globalAlpha = clamp(p * 4, 0, 1);
    ctx.strokeStyle = C.red; ctx.fillStyle = C.red; ctx.lineCap = "round";
    // symbol on the left, word on the right (Korean marking: ○ = right, ✗ = wrong)
    if (ok) { ctx.lineWidth = 6; circle(ctx, -30, 0, 20); ctx.stroke(); }
    else { ctx.lineWidth = 7; ctx.beginPath(); ctx.moveTo(-46, -17); ctx.lineTo(-14, 17); ctx.moveTo(-14, -17); ctx.lineTo(-46, 17); ctx.stroke(); }
    ctx.font = font(26); ctx.textBaseline = "middle";
    ctx.fillText(ok ? "정답!" : "오답", 2, 2);
    ctx.restore();
  }

  function badge(ctx, x, y, w, icon, text, dim) {
    ctx.save();
    rr(ctx, x, y, w, 28, 14); ctx.fillStyle = dim ? "#f1ece4" : C.soft; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = INK; ctx.stroke();
    ctx.strokeStyle = INK; ctx.lineWidth = 2;
    if (icon === "clock") {
      circle(ctx, x + 15, y + 14, 7); ctx.fillStyle = "#fff"; ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x + 15, y + 14); ctx.lineTo(x + 15, y + 9.5); ctx.moveTo(x + 15, y + 14); ctx.lineTo(x + 18.5, y + 15.5); ctx.stroke();
    } else {
      rr(ctx, x + 8, y + 7, 14, 14, 3); ctx.fillStyle = "#b9e2ff"; ctx.fill(); ctx.stroke();
      ctx.beginPath(); for (let i = 0; i < 3; i++) { ctx.moveTo(x + 11 + i * 4, y + 4); ctx.lineTo(x + 11 + i * 4, y + 7); ctx.moveTo(x + 11 + i * 4, y + 21); ctx.lineTo(x + 11 + i * 4, y + 24); } ctx.stroke();
    }
    let px = 14; ctx.font = font(px);
    while (px > 10 && ctx.measureText(text).width > w - 34) { px -= 0.5; ctx.font = font(px); }
    ctx.fillStyle = INK; ctx.textBaseline = "middle"; ctx.fillText(text, x + 28, y + 15);
    ctx.restore();
  }

  const fmtSec = (s) => (s < 0.01 ? "0.00초" : s < 10 ? s.toFixed(2) + "초" : s.toFixed(1) + "초");

  // ---------- layout ----------
  function layout(ctx, scene) {
    const L = {};
    // customer panel
    const hasSide = scene.orderState.length > 0;
    L.bubX = 196; L.bubW = hasSide ? 610 : 868;
    ctx.font = font(26);
    L.sayLines = wrap(ctx, `“${scene.say}”`, L.bubW - 40);
    ctx.font = font(14);
    L.hist = [];
    for (const m of scene.history) {
      const who = m.role === "user" ? "손님" : "로봇";
      for (const s of wrap(ctx, `${who}: ${m.content}`, L.bubW - 40)) L.hist.push({ who, s });
    }
    L.bubH = 26 + L.hist.length * 19 + (L.hist.length ? 8 : 0) + L.sayLines.length * 36 + 4;
    ctx.font = font(15);
    L.expLines = scene.expected ? wrap(ctx, scene.expected.map((x) => x.text).join(" / "), L.bubW - 140) : [];
    L.custTop = 74;
    L.expY = L.custTop + 16 + L.bubH + 22;
    L.custH = Math.max(172, L.bubH + 34 + (L.expLines.length ? 14 + L.expLines.length * 22 : 0));
    // robot columns
    L.bubbles = scene.robots.map((r) => (r.lines ? layoutBubble(ctx, r.lines) : null));
    L.bubMax = Math.max(64, ...L.bubbles.map((b) => (b ? b.h : 0)));
    L.colTop = L.custTop + L.custH + 22;
    L.colH = 52 + 160 + 16 + L.bubMax + 10 + 58 + 16 + 44;
    L.foot = [].concat(scene.footer);
    L.H = L.colTop + L.colH + 30 + L.foot.length * 21;
    return L;
  }

  // ---------- frame ----------
  function draw(ctx, scene, t, L) {
    ctx.save();
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, L.H);
    // title
    ctx.font = font(30); ctx.fillStyle = INK; ctx.textBaseline = "middle";
    ctx.fillText("카페 알바 로봇 오디션", M + 6, 38);
    ctx.font = font(15); ctx.fillStyle = "#7b6650";
    ctx.fillText(scene.live ? "직접 주문해 보기" : "같은 손님, 다섯 로봇", M + 300, 41);
    ctx.font = font(17);
    const tw = ctx.measureText(scene.tag).width + 30;
    rr(ctx, W - M - tw, 22, tw, 34, 17); shape(ctx, "#fff", 3);
    ctx.fillStyle = INK; ctx.fillText(scene.tag, W - M - tw + 15, 40);

    // customer panel
    panel(ctx, M, L.custTop, W - 2 * M - 5, L.custH, PAPER);
    customer(ctx, M + 92, L.custTop + 176 - 10, t, t < T_THINK + 0.2 && t > 0.15);
    ctx.font = font(16); ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("손님", M + 92, L.custTop + 176 - 10 - 22); ctx.textAlign = "left";
    const pop = clamp((t - T_SAY) / 0.35, 0, 1);
    if (pop > 0) {
      const bx = L.bubX, by = L.custTop + 16;
      ctx.save();
      const s = 0.6 + 0.4 * easeBack(pop);
      ctx.translate(bx, by + 60); ctx.scale(s, s); ctx.translate(-bx, -(by + 60)); ctx.globalAlpha = clamp(pop * 3, 0, 1);
      rr(ctx, bx, by, L.bubW, L.bubH, 22); ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = 3; ctx.strokeStyle = INK; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(bx + 1.5, by + 46); ctx.lineTo(bx - 26, by + 68); ctx.lineTo(bx + 1.5, by + 70);
      ctx.fillStyle = "#fff"; ctx.fill(); ctx.stroke();
      let y = by + 24;
      ctx.font = font(14);
      for (const h of L.hist) { ctx.fillStyle = h.who === "손님" ? "#8c7a66" : "#4f7cac"; ctx.fillText(h.s, bx + 20, y); y += 19; }
      if (L.hist.length) y += 8;
      ctx.font = font(26); ctx.fillStyle = INK;
      for (const s2 of L.sayLines) { ctx.fillText(s2, bx + 20, y + 6); y += 36; }
      ctx.restore();
    }
    if (scene.orderState.length) {  // receipt
      const rx = L.bubX + L.bubW + 18, ry = L.custTop + 16, rw = W - M - 5 - rx - 14;
      const rh = 40 + scene.orderState.length * 22;
      ctx.save(); ctx.translate(rx + rw / 2, ry); ctx.rotate(0.02); ctx.translate(-(rx + rw / 2), -ry);
      rr(ctx, rx, ry, rw, rh, 6); shape(ctx, "#fffdf7", 2.5);
      ctx.font = font(15); ctx.fillStyle = "#7b6650"; ctx.fillText("현재 주문", rx + 12, ry + 18);
      ctx.font = font(14.5); ctx.fillStyle = INK;
      scene.orderState.forEach((s2, i) => ctx.fillText(`${i + 1}. ${s2}`, rx + 12, ry + 42 + i * 22));
      ctx.restore();
    }
    if (L.expLines.length) {
      const a = clamp((t - scene.tExpected) / 0.35, 0, 1);
      const ex = L.bubX + 8, ey = L.expY;
      ctx.save(); ctx.textBaseline = "middle";
      rr(ctx, ex, ey - 14, 104, 28, 14); ctx.fillStyle = a > 0 ? C.red : "#cfc6bb"; ctx.fill();
      ctx.fillStyle = "#fff"; ctx.font = font(15); ctx.fillText("모범 답안", ex + 20, ey + 1);
      ctx.font = font(15);
      if (a > 0) { ctx.globalAlpha = a; ctx.fillStyle = INK; L.expLines.forEach((s2, i) => ctx.fillText(s2, ex + 118, ey + 1 + i * 22)); }
      else { ctx.fillStyle = "#a3968a"; ctx.fillText("다섯 로봇이 모두 답하면 공개!", ex + 118, ey + 1); }
      ctx.restore();
    }

    // robots
    scene.robots.forEach((r, i) => {
      const x = M + i * (COL_W + GAP), y = L.colTop, cx = x + COL_W / 2;
      const off = r.off || r.missing;
      const done = !off && t >= r.tDone;
      const thinking = !off && t >= T_THINK && !done;
      const st = off ? "off" : done ? (r.correct === true ? "right" : r.correct === false ? "wrong" : "answered") : thinking ? "thinking" : "idle";
      ctx.save();
      if (off) ctx.globalAlpha = 0.45;
      panel(ctx, x, y, COL_W - 5, L.colH, "#fff");
      // name plate
      rr(ctx, x + 10, y + 10, COL_W - 25, 38, 10); shape(ctx, ROBOT_COLORS[r.key], 2.5);
      ctx.fillStyle = INK; ctx.font = font(19); ctx.textBaseline = "middle"; ctx.textAlign = "center";
      ctx.fillText(r.name, cx - 2, y + 25); ctx.font = font(11.5); ctx.fillStyle = "#3d3329";
      ctx.fillText(r.sub, cx - 2, y + 40); ctx.textAlign = "left";
      // robot stage
      const sx = x + 10, sy = y + 54, sw = COL_W - 25, sh = 160;
      rr(ctx, sx, sy, sw, sh, 10); ctx.fillStyle = dots(ctx); ctx.fill();
      ctx.save(); rr(ctx, sx, sy, sw, sh, 10); ctx.clip();
      const bob = st === "thinking" ? Math.sin(t * 7 + i) * 2.5 : st === "right" ? -Math.abs(Math.sin((t - r.tDone) * 8)) * 6 * clamp(1 - (t - r.tDone) / 1.2, 0, 1) : 0;
      ctx.fillStyle = "rgba(43,33,24,.12)"; ctx.beginPath(); ctx.ellipse(cx - 2, sy + sh - 10, 56, 8, 0, 0, 7); ctx.fill();
      DRAW[r.key](ctx, cx - 2, sy + sh - 10 + bob, st, t + i * 0.37, ROBOT_COLORS[r.key]);
      ctx.restore();
      // bubble or cloud
      const by = sy + sh + 8;
      if (off) {
        ctx.font = font(16); ctx.fillStyle = C.gray; ctx.textAlign = "center";
        ctx.fillText(r.missing ? "기록 없음" : "쉬는 중", cx - 2, by + 40); ctx.textAlign = "left";
      } else if (thinking) thinkingCloud(ctx, cx - 2, by, t, T_THINK);
      else if (done) {
        const dashed = r.lines.every((l) => l.kind === "talk" || l.kind === "silent");
        drawBubble(ctx, x + 8, by, L.bubbles[i], cx - 2, dashed, clamp((t - r.tDone) / 0.28, 0, 1));
      }
      if (done && r.correct != null) stamp(ctx, cx + 4, y + L.colH - 44 - 16 - 36, r.correct, (t - r.tDone - 0.3) / 0.22);
      // badges
      const gy = y + L.colH - 40;
      let secs = 0;
      if (!off && t >= T_THINK) {
        if (done) secs = r.wall_s;
        else secs = scene.live ? Math.max(0, t) : Math.min(r.wall_s, unscaled(t - T_THINK));
      }
      const bw = (COL_W - 31) / 2;
      ctx.font = font(12); ctx.fillStyle = "#7b6650"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(scene.badgeLabel, cx - 2, gy - 10); ctx.textAlign = "left";
      badge(ctx, x + 10, gy, bw + 6, "clock", off ? "—" : fmtSec(secs), !done);
      badge(ctx, x + 16 + bw + 6, gy, bw - 6, "chip", r.key === "vending" ? "모델 없음" : r.memory_gb == null ? "—" : `${r.memory_gb.toFixed(1)}GB`, !done);
      ctx.restore();
    });

    // footer
    ctx.font = font(13); ctx.fillStyle = "#7b6650"; ctx.textBaseline = "middle";
    L.foot.forEach((line, k) => {
      ctx.font = font(k === 0 ? 15 : 13); ctx.fillStyle = k === 0 ? INK : "#7b6650";
      ctx.fillText(line, M + 6, L.colTop + L.colH + 24 + k * 21);
    });
    ctx.restore();
  }

  function sceneText(scene) {
    const parts = [scene.tag, [].concat(scene.footer).join(""), scene.badgeLabel, scene.say, "카페 알바 로봇 오디션 같은 손님, 다섯 로봇 직접 주문해 보기 손님 로봇 현재 주문 모범 답안 다섯 로봇이 모두 답하면 공개! 정답! 오답 모델 없음 쉬는 중 기록 없음 0123456789.초GB—“”…",
      ...Object.values(KIND).map((k) => k[0])];
    for (const m of scene.history) parts.push(m.content);
    parts.push(...scene.orderState);
    for (const x of scene.expected || []) parts.push(x.text, x.note || "");
    for (const r of scene.robots) { parts.push(r.name, r.sub); for (const x of r.lines || []) parts.push(x.text, x.note || ""); }
    return [...new Set(parts.join(""))].join("");
  }
  async function loadFonts(scene) {
    try { await document.fonts.load(font(16), sceneText(scene)); } catch (e) { /* offline: fallback font */ }
  }

  global.Stage = { W, layout, draw, timeline, scaled, unscaled, loadFonts };
})(window);
