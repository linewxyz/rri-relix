import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { FFmpeg } from '@ffmpeg/ffmpeg'
import { fetchFile, toBlobURL } from '@ffmpeg/util'
import { pipeline, env } from '@huggingface/transformers'
import { UploadCloud, Sparkles, Scissors, Download, Trash2, CheckCircle2, LoaderCircle, FileVideo2, WandSparkles, X, Clock3, Cpu } from 'lucide-react'
import './style.css'

env.allowLocalModels = false
env.useBrowserCache = true
const FFMPEG_CORE = 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm'
const ASR_MODEL = 'onnx-community/whisper-small'
const MAX_SECONDS = 3600          // 60 menit
const MAX_LONG_SIDE = 1920        // 1080p (landscape maupun portrait)
const MAX_SHORT_SIDE = 1080
const MAX_FILE_MB = 1024          // batas aman memori WASM FFmpeg

// Output selalu portrait 9:16. 720x1280 dipilih agar render di browser tidak terlalu lambat;
// ubah ke 1080 x 1920 jika perangkat kuat.
const OUT_W = 720, OUT_H = 1280
const SAMPLE_STEP = 1.5                     // deteksi subjek setiap 1,5 detik

// Model AI tambahan (semuanya gratis & berjalan di browser)
const EMBED_MODEL = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2'  // pemahaman makna (± 118 MB, sekali unduh)
const PERSON_MODEL = 'Xenova/yolos-tiny'                            // cadangan jika wajah tidak terlihat
const MP_VERSION = '0.10.21'
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'

// Panjang klip (detik) saat memilih highlight
const MIN_CLIP = 25, MAX_CLIP = 60, MAX_HIGHLIGHTS = 12

const fmt = n => {
  n = Math.max(0, Number(n) || 0)
  return `${String(Math.floor(n / 60)).padStart(2, '0')}:${String(Math.floor(n % 60)).padStart(2, '0')}`
}
const even = n => Math.max(2, Math.floor(n / 2) * 2)
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

// Hitung ukuran jendela crop 9:16 pada video sumber.
function cropGeometry(iw, ih) {
  const target = 9 / 16, ar = iw / ih
  if (Math.abs(ar - target) < 0.005) return { cw: even(iw), ch: even(ih), axis: null }
  if (ar > target) return { cw: even(Math.min(iw, ih * target)), ch: even(ih), axis: 'x' }
  return { cw: even(iw), ch: even(Math.min(ih, iw / target)), axis: 'y' }
}

function seekTo(video, t) {
  return new Promise(resolve => {
    let done = false
    const finish = () => { if (done) return; done = true; video.removeEventListener('seeked', finish); clearTimeout(tm); resolve() }
    const tm = setTimeout(finish, 4000)
    video.addEventListener('seeked', finish)
    video.currentTime = t
  })
}

// Pilih titik fokus kamera dari kotak hasil deteksi (wajah atau badan orang).
// Kalau semua subjek muat dalam jendela crop, pusatkan di antara mereka; kalau tidak,
// ikuti subjek yang besar dan paling dekat dengan posisi kamera sebelumnya.
function pickFocus(boxes, axis, cropLen, kind, prev) {
  if (!boxes.length) return null
  const lo = axis === 'x' ? 'x0' : 'y0', hi = axis === 'x' ? 'x1' : 'y1'
  const mn = Math.min(...boxes.map(b => b[lo])), mx = Math.max(...boxes.map(b => b[hi]))
  // Vertikal: wajah diletakkan sekitar 38% dari atas (ruang kepala), bukan di tengah.
  const anchor = b => axis === 'x' ? (b.x0 + b.x1) / 2
    : kind === 'face' ? (b.y0 + b.y1) / 2 + cropLen * 0.12 : b.y0 + (b.y1 - b.y0) * 0.35
  if (mx - mn <= cropLen * 0.9) {
    if (axis === 'x') return (mn + mx) / 2
    return kind === 'face' ? (mn + mx) / 2 + cropLen * 0.12 : mn + (mx - mn) * 0.35
  }
  const area = b => (b.x1 - b.x0) * (b.y1 - b.y0)
  const maxA = Math.max(...boxes.map(area))
  const pool = boxes.filter(b => area(b) >= maxA * 0.5)
  if (prev != null) return anchor(pool.reduce((a, b) => Math.abs(anchor(b) - prev) < Math.abs(anchor(a) - prev) ? b : a))
  return anchor(pool.reduce((a, b) => area(b) > area(a) ? b : a))
}

// Potongan (tile) persegi di atas frame, dua skala, supaya wajah kecil di shot lebar tetap terdeteksi.
function makeTiles(W, H) {
  const tiles = [], base = Math.min(W, H)
  for (const s0 of [base, base * 0.5]) {
    const s = Math.floor(s0), step = Math.max(1, Math.floor(s * 0.7))
    const pos = L => {
      const a = []
      for (let p = 0; p < L - s; p += step) a.push(p)
      if (!a.length || (L - s) - a[a.length - 1] > step * 0.3) a.push(Math.max(0, L - s))
      return a
    }
    for (const y of pos(H)) for (const x of pos(W)) tiles.push({ x, y, s })
  }
  return tiles
}

// Buang kotak ganda (wajah yang sama terdeteksi di beberapa tile).
function dedupeBoxes(boxes) {
  const inside = (c, b) => c[0] > b.x0 && c[0] < b.x1 && c[1] > b.y0 && c[1] < b.y1
  const ctr = b => [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2]
  const keep = []
  for (const b of boxes.slice().sort((a, c) => c.score - a.score)) {
    if (!keep.some(k => inside(ctr(b), k) || inside(ctr(k), b))) keep.push(b)
  }
  return keep
}

function detectFaces(faceDet, cv, tc, tctx) {
  const out = []
  for (const t of makeTiles(cv.width, cv.height)) {
    tctx.drawImage(cv, t.x, t.y, t.s, t.s, 0, 0, tc.width, tc.height)
    const k = t.s / tc.width
    for (const d of faceDet.detect(tc).detections || []) {
      const b = d.boundingBox
      if (!b) continue
      out.push({
        x0: t.x + b.originX * k, y0: t.y + b.originY * k,
        x1: t.x + (b.originX + b.width) * k, y1: t.y + (b.originY + b.height) * k,
        score: d.categories?.[0]?.score ?? 1
      })
    }
  }
  return dedupeBoxes(out)
}

// Gerakan kamera dibuat halus: median filter, zona mati, dan batas kecepatan pan.
function smoothPath(focus, step, cropLen, srcLen) {
  const f = focus.slice()
  const first = f.find(v => v != null)
  if (first == null) return f.map(() => (srcLen - cropLen) / 2)
  let last = first
  for (let i = 0; i < f.length; i++) { if (f[i] == null) f[i] = last; else last = f[i] }
  const med = f.map((_, i) => {
    const w = [f[Math.max(0, i - 1)], f[i], f[Math.min(f.length - 1, i + 1)]].sort((a, b) => a - b)
    return w[1]
  })
  let cam = med[0]
  const dead = cropLen * 0.12, maxMove = cropLen * 0.5 * step
  return med.map((c, i) => {
    if (i > 0) {
      const d = c - cam
      if (Math.abs(d) > dead) cam += clamp(d * 0.6, -maxMove, maxMove)
    }
    return clamp(cam - cropLen / 2, 0, srcLen - cropLen)   // sisi kiri/atas jendela crop
  })
}

// Gerakan kamera cadangan: pan linear dari satu sisi ke sisi lain sepanjang klip.
function sweepTrack(len, cropLen, srcLen) {
  return { times: [0, len], pos: [0, Math.max(0, srcLen - cropLen)], found: 0 }
}

// Deteksi subjek pada beberapa titik waktu klip, hasilnya posisi crop per keyframe.
// Prioritas: wajah (MediaPipe) -> badan orang (YOLOS, dimuat hanya jika perlu) -> tidak ada.
async function trackSubject({ video, faceDet, getPerson, clip, end, geo, srcW, srcH, onProgress }) {
  const { cw, ch, axis } = geo
  const len = end - clip.start
  const cropLen = axis === 'x' ? cw : ch, srcLen = axis === 'x' ? srcW : srcH
  const times = []
  for (let t = 0; t < len; t += SAMPLE_STEP) times.push(t)
  times.push(len)

  const scale = Math.min(1, (axis === 'x' ? 1280 : 720) / srcW)
  const cv = document.createElement('canvas')
  cv.width = Math.round(srcW * scale); cv.height = Math.round(srcH * scale)
  const ctx = cv.getContext('2d', { willReadFrequently: false })
  const tc = document.createElement('canvas'); tc.width = tc.height = 256
  const tctx = tc.getContext('2d')
  const pcv = document.createElement('canvas')
  pcv.width = 640; pcv.height = Math.round(640 * srcH / srcW)
  const pctx = pcv.getContext('2d')

  const focus = []
  let prev = null
  for (let i = 0; i < times.length; i++) {
    let f = null
    try {
      await seekTo(video, Math.min(clip.start + times[i], Math.max(0, video.duration - 0.05)))
      ctx.drawImage(video, 0, 0, cv.width, cv.height)
      let boxes = [], kind = 'face'
      if (faceDet) boxes = detectFaces(faceDet, cv, tc, tctx)
      if (!boxes.length) {
        const person = await getPerson()
        if (person) {
          pctx.drawImage(cv, 0, 0, pcv.width, pcv.height)
          const blob = await new Promise(r => pcv.toBlob(r, 'image/jpeg', 0.85))
          const k = cv.width / pcv.width
          boxes = (await person(blob, { threshold: 0.5 })).filter(d => d.label === 'person').map(d => ({
            x0: d.box.xmin * k, x1: d.box.xmax * k, y0: d.box.ymin * k, y1: d.box.ymax * k
          }))
          kind = 'person'
        }
      }
      const inSrc = boxes.map(b => ({ x0: b.x0 / scale, x1: b.x1 / scale, y0: b.y0 / scale, y1: b.y1 / scale }))
      f = pickFocus(inSrc, axis, cropLen, kind, prev)
      if (f != null) prev = f
    } catch (e) { console.warn('deteksi gagal di', times[i], e) }
    focus.push(f)
    onProgress?.(Math.round((i + 1) / times.length * 100))
  }
  const found = focus.filter(v => v != null).length
  // Tidak ada subjek terdeteksi sama sekali: pan perlahan dari kiri ke kanan (atau atas ke bawah).
  if (!found) return sweepTrack(len, cropLen, srcLen)
  return { times, pos: smoothPath(focus, SAMPLE_STEP, cropLen, srcLen), found }
}

// Ekspresi ffmpeg linear antar keyframe: posisi crop sebagai fungsi waktu t.
function panExpr(times, pos) {
  const r = v => Math.round(v * 10) / 10
  if (pos.every(v => Math.abs(v - pos[0]) < 1)) return String(r(pos[0]))
  const parts = []
  for (let i = 0; i < times.length - 1; i++) {
    const t0 = times[i].toFixed(2), t1 = times[i + 1].toFixed(2)
    const p0 = r(pos[i]), p1 = r(pos[i + 1])
    parts.push(`gte(t,${t0})*lt(t,${t1})*(${p0}+(${r(p1 - p0)})*(t-${t0})/${(times[i + 1] - times[i]).toFixed(2)})`)
  }
  parts.push(`gte(t,${times[times.length - 1].toFixed(2)})*${r(pos[pos.length - 1])}`)
  return parts.join('+')
}

function buildFilter(geo, expr) {
  const { cw, ch, axis } = geo
  const x = axis === 'x' ? `'${expr}'` : '0'
  const y = axis === 'y' ? `'${expr}'` : '0'
  return `crop=w=${cw}:h=${ch}:x=${x}:y=${y},scale=${OUT_W}:${OUT_H}:flags=bicubic,setsar=1,format=yuv420p`
}

// Decode WAV PCM16 mono 16 kHz langsung ke Float32Array.
// (AudioContext.decodeAudioData akan me-resample ke 44.1/48 kHz sehingga Whisper salah baca.)
function wavToFloat32(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  let pos = 12, start = -1, len = 0
  while (pos + 8 <= bytes.byteLength) {
    const id = String.fromCharCode(bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3])
    const size = dv.getUint32(pos + 4, true)
    if (id === 'data') { start = pos + 8; len = Math.min(size, bytes.byteLength - start); break }
    pos += 8 + size + (size % 2)
  }
  if (start < 0) throw new Error('Data audio WAV tidak ditemukan. Video mungkin tidak memiliki audio.')
  const n = len >> 1
  if (n < 16000) throw new Error('Audio terlalu pendek atau kosong.')
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(start + i * 2, true) / 32768
  return out
}

// ---------- Pemilihan highlight ----------
const endsSentence = t => /[.!?…]["'”’)\]]*$/.test((t || '').trim())
const dotv = (a, b) => { let x = 0; for (let i = 0; i < a.length; i++) x += a[i] * b[i]; return x }

function meanNorm(vecs) {
  const m = new Float32Array(vecs[0].length)
  for (const v of vecs) for (let i = 0; i < m.length; i++) m[i] += v[i]
  let n = 0; for (let i = 0; i < m.length; i++) n += m[i] * m[i]
  n = Math.sqrt(n) || 1
  for (let i = 0; i < m.length; i++) m[i] /= n
  return m
}

// Segmen Whisper -> daftar {start,end,text} yang valid.
function normalizeSegments(chunks, totalDuration) {
  return (chunks || []).map(c => {
    const s = c.timestamp?.[0] ?? c.start ?? 0
    const e = c.timestamp?.[1] ?? c.end
    return {
      start: Number(s),
      end: e == null ? Number(totalDuration) || Number(s) : Number(e), // chunk terakhir sering end = null
      text: (c.text || '').trim()
    }
  }).filter(c => c.text && Number.isFinite(c.start) && Number.isFinite(c.end) && c.end > c.start)
}

// Embedding kalimat (model makna), diproses per batch.
async function embedTexts(embedder, texts, onProgress) {
  const out = []
  for (let i = 0; i < texts.length; i += 16) {
    const t = await embedder(texts.slice(i, i + 16), { pooling: 'mean', normalize: true })
    for (const row of t.tolist()) out.push(Float32Array.from(row))
    onProgress?.(Math.round(Math.min(texts.length, i + 16) / texts.length * 100))
  }
  return out
}

const HEUR_TERMS = ['karena', 'tetapi', 'namun', 'penting', 'harus', 'jangan', 'ternyata', 'menurut', 'masyarakat', 'generasi', 'kesimpulan', 'pertama', 'kedua', 'berarti', 'solusi', 'masalah', 'kita', 'saya', 'anda', 'bagaimana', 'mengapa']
function heuristic(text) {
  const lower = text.toLowerCase()
  const hits = HEUR_TERMS.filter(t => lower.includes(t)).length
  const words = text.split(/\s+/).length
  const raw = 45 + Math.min(words, 90) * .35 + hits * 3 + (/[!?]/.test(text) ? 5 : 0)
  return clamp((raw - 45) / 54, 0, 1)
}

// Semua kandidat klip yang diawali di awal kalimat dan DIAKHIRI di akhir kalimat.
function makeCandidates(segs, minLen, maxLen) {
  const ratio = segs.filter(x => endsSentence(x.text)).length / segs.length
  const hasPunct = ratio >= 0.2          // transkrip tanpa tanda baca: pakai batas segmen saja
  const cands = []
  for (let i = 0; i < segs.length; i++) {
    if (hasPunct && i > 0 && !endsSentence(segs[i - 1].text)) continue
    let first = -1, last = -1, fb = -1
    for (let j = i; j < segs.length; j++) {
      const dur = segs[j].end - segs[i].start
      if (dur > maxLen) break
      if (fb < 0 && dur >= minLen) fb = j
      if (dur >= minLen && (!hasPunct || endsSentence(segs[j].text) || j === segs.length - 1)) {
        if (first < 0) first = j
        last = j
      }
    }
    const ends = new Set()
    if (first >= 0) { ends.add(first); ends.add(last) } else if (fb >= 0) ends.add(fb)
    for (const j of ends) cands.push({ i, j })
  }
  return cands
}

// Pilih highlight: batas di akhir kalimat, skor makna (embedding) + heuristik, tanpa klip yang tumpang tindih
// dan tanpa klip yang isinya terlalu mirip (MMR).
function pickHighlights(segs, totalDuration, embeds) {
  if (!segs.length) return []
  let cands = makeCandidates(segs, MIN_CLIP, MAX_CLIP)
  if (!cands.length) cands = makeCandidates(segs, 8, MAX_CLIP)
  if (!cands.length) return []

  const useEmb = Array.isArray(embeds) && embeds.length === segs.length
  const globalEmb = useEmb ? meanNorm(embeds) : null
  const items = cands.map(({ i, j }) => {
    const part = segs.slice(i, j + 1)
    const text = part.map(x => x.text).join(' ')
    const c = {
      start: Math.max(0, segs[i].start - 0.15),
      end: totalDuration ? Math.min(totalDuration, segs[j].end + 0.35) : segs[j].end + 0.35,
      text, h: heuristic(text)
    }
    if (useEmb) {
      const vs = embeds.slice(i, j + 1)
      c.emb = meanNorm(vs)
      c.cen = dotv(c.emb, globalEmb)                                        // seberapa mewakili inti pembahasan
      c.coh = vs.reduce((a, v) => a + dotv(v, c.emb), 0) / vs.length        // seberapa fokus pada satu topik
    }
    return c
  }).filter(c => c.end - c.start >= 5)
  if (!items.length) return []

  const norm = key => {
    const vals = items.map(c => c[key]), lo = Math.min(...vals), hi = Math.max(...vals)
    items.forEach(c => { c[key + 'N'] = hi > lo ? (c[key] - lo) / (hi - lo) : 0.5 })
  }
  if (useEmb) { norm('cen'); norm('coh') }
  items.forEach(c => { c.final = useEmb ? 0.4 * c.cenN + 0.35 * c.cohN + 0.25 * c.h : c.h })

  const picked = []
  let pool = items.slice()
  while (picked.length < MAX_HIGHLIGHTS && pool.length) {
    let best = null, bestVal = -Infinity
    for (const c of pool) {
      let sim = 0
      if (useEmb) for (const p of picked) sim = Math.max(sim, dotv(c.emb, p.emb))
      const val = c.final - 0.6 * Math.max(0, sim - 0.55)
      if (val > bestVal) { bestVal = val; best = c }
    }
    if (!best) break
    picked.push(best)
    pool = pool.filter(c => !(c.start < best.end && c.end > best.start))   // buang yang tumpang tindih
  }
  return picked.sort((a, b) => b.final - a.final).map(c => ({
    id: crypto.randomUUID(), start: c.start, end: c.end, text: c.text,
    score: Math.min(99, Math.round(50 + 49 * c.final)), rendered: false, url: ''
  }))
}

function App() {
  const [file, setFile] = useState(null), [url, setUrl] = useState(''), [duration, setDuration] = useState(0)
  const [clips, setClips] = useState([]), [asr, setAsr] = useState(null)
  const [busy, setBusy] = useState(false), [progress, setProgress] = useState(0), [phase, setPhase] = useState('')
  const [status, setStatus] = useState('Upload rekaman siaran untuk memulai.'), [error, setError] = useState('')
  const [modelReady, setModelReady] = useState(false)
  const videoRef = useRef(null), inputRef = useRef(null)
  const ffmpegRef = useRef(null), faceRef = useRef(null), personRef = useRef(null), personFailRef = useRef(false), embRef = useRef(null), dimsRef = useRef({ w: 0, h: 0 }), urlRef = useRef(''), clipsRef = useRef([])

  urlRef.current = url
  clipsRef.current = clips

  // Bebaskan memori saat tab ditutup / komponen dilepas.
  useEffect(() => () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    clipsRef.current.forEach(c => c.url && URL.revokeObjectURL(c.url))
    ffmpegRef.current?.terminate()
    try { faceRef.current?.close() } catch {}
  }, [])

  // Peringatan sebelum refresh/tutup tab jika ada proses atau klip yang belum diunduh.
  useEffect(() => {
    const h = e => {
      if (busy || clips.some(c => c.url)) { e.preventDefault(); e.returnValue = '' }
    }
    window.addEventListener('beforeunload', h)
    return () => window.removeEventListener('beforeunload', h)
  }, [busy, clips])

  function revokeAll() {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    clipsRef.current.forEach(c => c.url && URL.revokeObjectURL(c.url))
  }

  function resetVideo(message) {
    revokeAll()
    setFile(null); setUrl(''); setClips([]); setDuration(0)
    if (inputRef.current) inputRef.current.value = ''
    setStatus(message || 'Workspace dibersihkan.')
  }

  function selectFile(f) {
    if (!f || busy) return
    if (!f.type.startsWith('video/')) { setError('File harus berupa video.'); return }
    if (f.size > MAX_FILE_MB * 1048576) { setError(`Ukuran file maksimal ${MAX_FILE_MB} MB.`); return }
    revokeAll()
    setFile(f); setUrl(URL.createObjectURL(f)); setClips([]); setDuration(0); setError('')
    setStatus('Memeriksa video...')
  }

  function loaded() {
    const v = videoRef.current
    if (!v) return
    const d = v.duration, w = v.videoWidth, h = v.videoHeight
    if (!Number.isFinite(d) || d <= 0) { resetVideo(''); setError('Durasi video tidak dapat dibaca.'); return }
    if (d > MAX_SECONDS) { resetVideo(''); setError('Durasi video maksimal 60 menit.'); return }
    if (Math.max(w, h) > MAX_LONG_SIDE || Math.min(w, h) > MAX_SHORT_SIDE) {
      resetVideo(''); setError('Resolusi video maksimal 1080p (1920 × 1080).'); return
    }
    dimsRef.current = { w, h }
    setDuration(d); setError('')
    setStatus(`Video siap: ${w}×${h}, durasi ${fmt(d)}. Pemrosesan dilakukan di browser.`)
  }

  async function getFFmpeg() {
    if (ffmpegRef.current) return ffmpegRef.current
    const f = new FFmpeg()
    f.on('progress', ({ progress }) => setProgress(Math.max(0, Math.min(100, Math.round((progress || 0) * 100)))))
    await f.load({
      coreURL: await toBlobURL(`${FFMPEG_CORE}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${FFMPEG_CORE}/ffmpeg-core.wasm`, 'application/wasm')
    })
    ffmpegRef.current = f
    return f
  }

  const dlProgress = p => {
    if (p.status === 'progress' && p.total) setProgress(Math.round(p.loaded / p.total * 100))
  }

  // Model makna (embedding) untuk memilih highlight. Gagal -> kembali ke skor heuristik.
  async function loadEmbedder() {
    if (embRef.current) return embRef.current
    try {
      try { embRef.current = await pipeline('feature-extraction', EMBED_MODEL, { dtype: 'q8', progress_callback: dlProgress }) }
      catch { embRef.current = await pipeline('feature-extraction', EMBED_MODEL, { progress_callback: dlProgress }) }
    } catch (e) { console.warn('Model makna gagal dimuat, memakai skor sederhana.', e) }
    return embRef.current
  }

  // Detektor wajah MediaPipe (dimuat dari CDN). Gagal -> pakai deteksi badan orang / pan.
  async function loadFaceDetector() {
    if (faceRef.current) return faceRef.current
    try {
      const mp = await import(/* @vite-ignore */ `${MP_BASE}/vision_bundle.mjs`)
      const vision = await mp.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`)
      faceRef.current = await mp.FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'CPU' },
        runningMode: 'IMAGE', minDetectionConfidence: 0.55
      })
    } catch (e) { console.warn('Detektor wajah gagal dimuat.', e) }
    return faceRef.current
  }

  // Detektor badan orang: hanya dimuat saat ada frame tanpa wajah terdeteksi.
  async function getPerson() {
    if (personRef.current) return personRef.current
    if (personFailRef.current) return null
    try {
      setStatus('Memuat AI cadangan untuk mendeteksi orang...')
      try { personRef.current = await pipeline('object-detection', PERSON_MODEL, { dtype: 'q8', progress_callback: dlProgress }) }
      catch { personRef.current = await pipeline('object-detection', PERSON_MODEL, { progress_callback: dlProgress }) }
    } catch (e) { console.warn('Detektor orang gagal dimuat.', e); personFailRef.current = true }
    return personRef.current
  }

  async function transcribe() {
    if (!file || busy) return
    setBusy(true); setError(''); setProgress(0)
    clips.forEach(c => c.url && URL.revokeObjectURL(c.url))
    setClips([])
    const src = 'relix-source.mp4', wav = 'relix-audio.wav'
    let engine = null
    try {
      // 1. Whisper
      let transcriber = asr
      if (!transcriber) {
        setPhase('Memuat model AI Whisper')
        setStatus('Mengunduh model Whisper. Pertama kali dapat memerlukan waktu dan kuota data.')
        transcriber = await pipeline('automatic-speech-recognition', ASR_MODEL, {
          dtype: 'q8',
          progress_callback: p => {
            if (p.status === 'progress' && p.total) setProgress(Math.round(p.loaded / p.total * 100))
            if (p.status === 'download' || p.status === 'initiate') setStatus(`Mengunduh model AI: ${p.file || 'model'}...`)
          }
        })
        setAsr(transcriber); setModelReady(true)
      }

      // 2. FFmpeg + ekstrak audio 16 kHz mono
      setPhase('Memuat mesin video'); setProgress(0)
      engine = await getFFmpeg()
      setPhase('Ekstrak audio'); setProgress(0); setStatus('Membaca video dan mengekstrak audio...')
      await engine.writeFile(src, await fetchFile(file))
      const rc = await engine.exec(['-i', src, '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', wav])
      if (rc !== 0) throw new Error('Gagal mengekstrak audio. Format video mungkin tidak didukung atau tidak memiliki audio.')
      const samples = wavToFloat32(await engine.readFile(wav))
      await engine.deleteFile(wav).catch(() => {})

      // 3. Transkripsi
      setPhase('AI transcription'); setProgress(0)
      setStatus('Whisper sedang mengenali percakapan. Durasi proses bergantung pada perangkat.')
      const result = await transcriber(samples, {
        chunk_length_s: 30, stride_length_s: 5,
        return_timestamps: true, language: 'indonesian', task: 'transcribe'
      })
      const segs = normalizeSegments(result.chunks || [], duration)

      // 3b. Pahami makna tiap kalimat untuk memilih momen terbaik
      let embeds = null
      if (segs.length) {
        setPhase('Memuat AI makna'); setProgress(0)
        setStatus('Memuat model pemahaman makna (sekali unduh, tersimpan di browser)...')
        const embedder = await loadEmbedder()
        if (embedder) {
          try {
            setPhase('Menganalisis makna'); setProgress(0)
            setStatus('Menilai topik dan inti pembahasan tiap bagian...')
            embeds = await embedTexts(embedder, segs.map(x => x.text), setProgress)
          } catch (e) { console.warn('Embedding gagal.', e); embeds = null }
        }
      }
      const candidates = pickHighlights(segs, duration, embeds)
      if (!candidates.length) throw new Error('Tidak ditemukan kandidat highlight. Pastikan video memiliki audio percakapan yang jelas.')
      setClips(candidates)

      // 4. Auto-reframe + render otomatis semua kandidat
      const { w: srcW, h: srcH } = dimsRef.current
      const geo = cropGeometry(srcW, srcH)

      let faceDet = null
      if (geo.axis) {
        setPhase('Memuat AI deteksi wajah'); setProgress(0)
        setStatus('Memuat detektor wajah untuk auto-reframe...')
        faceDet = await loadFaceDetector()
      }

      // elemen video terpisah untuk mengambil frame analisis
      const probe = document.createElement('video')
      probe.muted = true; probe.preload = 'auto'; probe.src = url
      await new Promise(r => { probe.onloadeddata = r; probe.onerror = r; setTimeout(r, 8000) })

      let failed = 0
      try {
        for (let i = 0; i < candidates.length; i++) {
          const c = candidates[i], out = `clip-${i}.mp4`
          try {
            const end = duration ? Math.min(c.end, duration) : c.end
            const len = end - c.start

            // a. reframing: cari posisi subjek sepanjang klip
            let track = { times: [0, len], pos: [0, 0] }
            if (geo.axis) {
              const axisLen = geo.axis === 'x' ? srcW : srcH, cropLen = geo.axis === 'x' ? geo.cw : geo.ch
              track = sweepTrack(len, cropLen, axisLen)   // cadangan jika model deteksi tidak tersedia
              setPhase(`Reframing klip ${i + 1}/${candidates.length}`); setProgress(0)
              setStatus(`Mencari subjek utama di Highlight ${String(i + 1).padStart(2, '0')}...`)
              track = await trackSubject({ video: probe, faceDet, getPerson, clip: c, end, geo, srcW, srcH, onProgress: setProgress })
            }

            // b. potong + crop 9:16 + encode
            setPhase(`Render klip ${i + 1}/${candidates.length}`); setProgress(0)
            setStatus(`Membuat video portrait: Highlight ${String(i + 1).padStart(2, '0')} (${fmt(c.start)} – ${fmt(c.end)})...`)
            const rc2 = await engine.exec([
              '-ss', String(c.start), '-i', src, '-t', String(len),
              '-vf', buildFilter(geo, panExpr(track.times, track.pos)),
              '-map', '0:v:0', '-map', '0:a?',
              '-r', '30',
              '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24',
              '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', out
            ])
            if (rc2 !== 0) throw new Error('ffmpeg gagal')
            const data = await engine.readFile(out)
            const obj = URL.createObjectURL(new Blob([data], { type: 'video/mp4' }))
            setClips(prev => prev.map(x => x.id === c.id ? { ...x, url: obj, rendered: true } : x))
          } catch (e) {
            console.error(e); failed++
          } finally {
            await engine.deleteFile(out).catch(() => {})
          }
        }
      } finally {
        probe.removeAttribute('src'); probe.load()
      }
      setStatus(failed
        ? `Selesai. ${candidates.length - failed} klip berhasil, ${failed} gagal dirender.`
        : `${candidates.length} klip portrait selesai dirender. Silakan unduh hasilnya.`)
    } catch (e) {
      console.error(e)
      setError(`Analisis AI gagal: ${e?.message || e}`)
      setStatus('Periksa format video, koneksi internet, dan kapasitas memori browser.')
    } finally {
      // Hentikan FFmpeg agar memori WASM (video sumber) dilepas.
      if (ffmpegRef.current) { try { ffmpegRef.current.terminate() } catch {} ffmpegRef.current = null }
      setBusy(false); setPhase(''); setProgress(0)
    }
  }

  const mmss = n => `${String(Math.floor(n / 60)).padStart(2, '0')}m${String(Math.floor(n % 60)).padStart(2, '0')}`
  function download(c, i) {
    if (!c.url) return
    const a = document.createElement('a'); a.href = c.url
    a.download = `highlight-${String(i + 1).padStart(2, '0')}_${mmss(c.start)}-${mmss(c.end)}.mp4`
    a.click()
  }
  function clear() {
    if (busy) return
    if (clips.some(c => c.url) && !window.confirm('Klip yang belum diunduh akan hilang. Lanjutkan?')) return
    resetVideo('Workspace dibersihkan.'); setError('')
  }

  return <div className="shell">
    <header className="nav"><div className="brand"><div className="mark">R</div><div><b>RRI <i>RELIX</i></b><small>by line</small></div></div><div className="navright"><span className="dot" /> AI BROWSER WORKSPACE <span className="org">LPP RRI BANJARMASIN</span></div></header>
    <main>
      <section className="hero"><div className="eyebrow"><Sparkles size={14} /> AI-POWERED PRODUCTION WORKSPACE</div><h1>Turn your broadcast<br /><span>into content.</span></h1><p>AI-Powered Broadcast Content &amp; Production Studio</p></section>
      <section className="grid">
        <div className="panel"><div className="panelhead"><div><small>01 / SOURCE</small><h2>Video source</h2></div><FileVideo2 /></div>
          {!file ? <button className="drop" onClick={() => inputRef.current?.click()} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); selectFile(e.dataTransfer.files?.[0]) }}><UploadCloud size={34} /><b>Upload broadcast video</b><span>Drag &amp; drop atau pilih file dari perangkat</span><strong>Select video</strong><small>MP4 · MOV · WEBM · Maks. 60 menit · 1080p · Tidak diunggah ke server</small></button>
            : <><video className="sourcevideo" ref={videoRef} src={url} controls onLoadedMetadata={loaded} /><div className="fileline"><b>{file.name}</b><span>{(file.size / 1048576).toFixed(1)} MB · {fmt(duration)}</span><button className="ghost" onClick={clear} disabled={busy}><X size={16} /></button></div></>}
          <input ref={inputRef} hidden type="file" accept="video/*" onChange={e => selectFile(e.target.files?.[0])} />
        </div>
        <div className="panel"><div className="panelhead"><div><small>02 / AI ENGINE</small><h2>Analyze broadcast</h2></div><Sparkles /></div>
          <div className="ai-card"><div className="ai-icon"><WandSparkles /></div><div><b>Whisper AI · Multilingual</b><p>Transkripsi Bahasa Indonesia, pemilihan momen berdasarkan makna, lalu klip otomatis dipotong portrait 9:16 mengikuti wajah.</p></div></div>
          <div className="modelrow"><span><Cpu size={15} /> AI model</span><b>{modelReady ? 'Loaded in browser' : 'Whisper Small · multilingual'}</b></div>
          <button className="primary" onClick={transcribe} disabled={!file || !duration || busy}><Sparkles size={16} />{busy ? 'AI sedang bekerja...' : 'Generate AI Highlights'}</button>
          <p className="hint">Model AI diunduh saat pertama kali digunakan dan tersimpan di cache browser. Jangan tutup tab selama proses berjalan.</p>
        </div>
      </section>
      <section className="panel results">
        <div className="resulthead"><div><small>03 / AI OUTPUT</small><h2>Highlight results <em>{clips.length}</em></h2><p>Klip dipilih berdasarkan makna, dipotong di akhir kalimat, dan dirender otomatis portrait 9:16 dengan auto-reframe pada wajah.</p></div><button className="outline" onClick={clear} disabled={!file || busy}><Trash2 size={14} /> Clear</button></div>
        {!clips.length ? <div className="empty"><div><Scissors size={25} /></div><b>No highlights generated</b><span>Upload video lalu jalankan AI untuk menemukan kandidat momen menarik.</span></div> :
          <div className="clipgrid">{clips.map((c, i) => <article className="clip chosen" key={c.id}>
            <div className="cliptop"><b>HIGHLIGHT {String(i + 1).padStart(2, '0')}</b><span className="score">AI {c.score}%</span></div>
            <div className="clipmeta"><span><Clock3 size={13} />{fmt(c.start)} – {fmt(c.end)}</span><span>{fmt(c.end - c.start)}</span></div>
            {c.url && <video className="preview" src={c.url} controls style={{ maxHeight: 460, aspectRatio: "9 / 16", margin: "0 auto", display: "block", background: "#000" }} />}
            <div className="clipbuttons">
              {c.rendered
                ? <button className="download" onClick={() => download(c, i)}><Download size={14} /> Download MP4</button>
                : <span className="hint">{busy ? 'Menunggu proses otomatis...' : 'Render gagal'}</span>}
            </div>
          </article>)}</div>}
        {busy && <div className="progress"><div><span><LoaderCircle className="spin" size={14} /> {phase || 'Processing'}</span><b>{progress}%</b></div><div className="track"><i style={{ width: `${progress}%` }} /></div></div>}
      </section>
      <div className="privacy"><CheckCircle2 size={18} /><div><b>Privacy-first · Browser processing</b><p>Video mentah tidak dikirim ke server aplikasi. AI inference dan rendering berlangsung di browser. Hasil klip hanya tersedia sementara di sesi browser; unduh sebelum membersihkan workspace atau menutup tab.</p></div></div>
      {status && <p className="status">{status}</p>}{error && <p className="error">{error}</p>}
    </main>
    <footer><b>RRI RELIX <span>by line</span></b><small>AI-Powered Broadcast Content &amp; Production Studio</small><small>© LPP RRI Banjarmasin</small></footer>
  </div>
}
createRoot(document.getElementById('root')).render(<App />)
