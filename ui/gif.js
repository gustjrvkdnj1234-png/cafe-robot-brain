// Small GIF89a encoder (no dependencies). Global 256-colour palette from all frames,
// each frame written as the changed rectangle only (disposal = keep), infinite loop.
// Usage: const bytes = encodeGIF(frames /* ImageData[] same size */, delaysCs /* number[] */)
(function (global) {
  "use strict";

  function buildPalette(frames) {
    // popularity on a 4-bit-per-channel histogram, colour = mean of the real pixels in the bucket
    const n = 4096, cnt = new Float64Array(n), sr = new Float64Array(n), sg = new Float64Array(n), sb = new Float64Array(n);
    for (const f of frames) {
      const d = f.data;
      for (let i = 0; i < d.length; i += 8) {  // every 2nd pixel is plenty
        const k = ((d[i] >> 4) << 8) | ((d[i + 1] >> 4) << 4) | (d[i + 2] >> 4);
        cnt[k]++; sr[k] += d[i]; sg[k] += d[i + 1]; sb[k] += d[i + 2];
      }
    }
    const order = [];
    for (let k = 0; k < n; k++) if (cnt[k] > 0) order.push(k);
    order.sort((a, b) => cnt[b] - cnt[a]);
    const pal = [];
    for (const k of order.slice(0, 256)) pal.push([sr[k] / cnt[k] | 0, sg[k] / cnt[k] | 0, sb[k] / cnt[k] | 0]);
    while (pal.length < 256) pal.push([0, 0, 0]);
    return pal;
  }

  function makeMapper(pal) {
    const lut = new Int16Array(32768).fill(-1);
    return function (r, g, b) {
      const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
      let v = lut[key];
      if (v >= 0) return v;
      const rr = (r >> 3) * 8 + 4, gg = (g >> 3) * 8 + 4, bb = (b >> 3) * 8 + 4;
      let best = 0, bd = Infinity;
      for (let i = 0; i < 256; i++) {
        const p = pal[i], dr = p[0] - rr, dg = p[1] - gg, db = p[2] - bb;
        const dist = 2 * dr * dr + 4 * dg * dg + 3 * db * db;
        if (dist < bd) { bd = dist; best = i; }
      }
      lut[key] = best;
      return best;
    };
  }

  // LZW tables shared across calls; a generation stamp replaces clearing
  const TABLE = new Int16Array(1 << 20), STAMP = new Int32Array(1 << 20);
  let GEN = 1;

  function lzw(idx, out) {
    const minSize = 8, clear = 256, eoi = 257;
    let size = minSize + 1, next = eoi + 1, cur = 0, bits = 0;
    const block = [];
    const flushByte = (b) => { block.push(b); if (block.length === 255) { out.push(255, ...block); block.length = 0; } };
    const emit = (code) => {
      cur |= code << bits; bits += size;
      while (bits >= 8) { flushByte(cur & 255); cur >>>= 8; bits -= 8; }
    };
    out.push(minSize);
    emit(clear); GEN++;
    let prefix = idx[0];
    for (let i = 1; i < idx.length; i++) {
      const k = idx[i], key = (prefix << 8) | k;
      if (STAMP[key] === GEN) { prefix = TABLE[key]; continue; }
      emit(prefix);
      if (next === 4096) {
        emit(clear); GEN++; size = minSize + 1; next = eoi + 1;
      } else {
        if (next >= (1 << size)) size++;
        TABLE[key] = next++; STAMP[key] = GEN;
      }
      prefix = k;
    }
    emit(prefix);
    emit(eoi);
    if (bits > 0) flushByte(cur & 255);
    if (block.length) out.push(block.length, ...block);
    out.push(0);
  }

  function encodeGIF(frames, delaysCs) {
    const w = frames[0].width, h = frames[0].height;
    const pal = buildPalette(frames), map = makeMapper(pal);
    const out = [];
    const u16 = (v) => out.push(v & 255, (v >> 8) & 255);
    out.push(...[71, 73, 70, 56, 57, 97]);            // GIF89a
    u16(w); u16(h); out.push(0xF7, 0, 0);             // global colour table, 256 entries
    for (const p of pal) out.push(p[0], p[1], p[2]);
    out.push(0x21, 0xFF, 11, ...[78, 69, 84, 83, 67, 65, 80, 69, 50, 46, 48], 3, 1, 0, 0, 0); // NETSCAPE2.0 loop

    // index all frames, then merge identical ones into longer delays
    let prev = null;
    const items = [];
    frames.forEach((f, fi) => {
      const d = f.data, cur = new Uint8Array(w * h);
      for (let p = 0, i = 0; p < cur.length; p++, i += 4) cur[p] = map(d[i], d[i + 1], d[i + 2]);
      let x0 = w, y0 = h, x1 = -1, y1 = -1;
      if (!prev) { x0 = 0; y0 = 0; x1 = w - 1; y1 = h - 1; } else {
        for (let y = 0; y < h; y++) {
          const row = y * w;
          for (let x = 0; x < w; x++) {
            if (cur[row + x] !== prev[row + x]) {
              if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
            }
          }
        }
      }
      const delay = delaysCs[fi];
      if (x1 < 0) { items[items.length - 1].delay += delay; return; }  // nothing changed
      items.push({ cur, x0, y0, x1, y1, delay });
      prev = cur;
    });
    for (const it of items) {
      const rw = it.x1 - it.x0 + 1, rh = it.y1 - it.y0 + 1, sub = new Uint8Array(rw * rh);
      for (let y = 0; y < rh; y++) sub.set(it.cur.subarray((it.y0 + y) * w + it.x0, (it.y0 + y) * w + it.x0 + rw), y * rw);
      out.push(0x21, 0xF9, 4, 0x04); u16(Math.min(65535, Math.round(it.delay))); out.push(0, 0); // keep previous frame
      out.push(0x2C); u16(it.x0); u16(it.y0); u16(rw); u16(rh); out.push(0);
      lzw(sub, out);
    }
    out.push(0x3B);
    return new Uint8Array(out);
  }

  global.encodeGIF = encodeGIF;
  if (typeof module !== "undefined") module.exports = { encodeGIF };
})(typeof window !== "undefined" ? window : globalThis);
