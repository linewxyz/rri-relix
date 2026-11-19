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
const ASR_MODEL = 'onnx-community/whisper-tiny'
const MAX_SECONDS = 3600          // 60 menit
const MAX_LONG_SIDE = 1920        // 1080p (landscape maupun portrait)
const MAX_SHORT_SIDE = 1080
const MAX_FILE_MB = 1024          // batas aman memori WASM FFmpeg

// Output selalu portrait 9:16. 720x1280 dipilih agar render di browser tidak terlalu lambat;
// ubah ke 1080 x 1920 jika perangkat kuat.
const OUT_W = 720, OUT_H = 1280
const DETECT_MODEL = 'Xenova/yolos-tiny'   // deteksi orang untuk auto-reframe
const SAMPLE_STEP = 1.5                     // deteksi subjek setiap 1,5 detik
const DETECT_WIDTH = 512                    // lebar frame yang dianalisis

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

// Pilih titik fokus dari hasil deteksi orang: kalau semua orang muat dalam jendela crop,
// pusatkan di antara mereka; kalau tidak, ikuti orang yang paling besar di frame.
function pickFocus(people, axis, cropLen, srcW, srcH) {
  if (!people.length) return null
  const lo = axis === 'x' ? 'x0' : 'y0', hi = axis === 'x' ? 'x1' : 'y1'
  const mn = Math.min(...people.map(p => p[lo])), mx = Math.max(...people.map(p => p[hi]))
  if (mx - mn <= cropLen * 0.9) {
    // vertikal: condong ke bagian atas (wajah), bukan tengah badan
    return axis === 'x' ? (mn + mx) / 2 : mn + (mx - mn) * 0.35
  }
  const big = people.reduce((a, b) => (b.x1 - b.x0) * (b.y1 - b.y0) > (a.x1 - a.x0) * (a.y1 - a.y0) ? b : a)
  return axis === 'x' ? (big.x0 + big.x1) / 2 : big.y0 + (big.y1 - big.y0) * 0.35
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

// Deteksi subjek pada beberapa titik waktu klip, hasilnya posisi crop per keyframe.
async function trackSubject({ video, detector, clip, end, geo, srcW, srcH, onProgress }) {
  const { cw, ch, axis } = geo
  const len = end - clip.start
  const times = []
  for (let t = 0; t < len; t += SAMPLE_STEP) times.push(t)
  times.push(len)
  const cvs = document.createElement('canvas')
  const scale = DETECT_WIDTH / srcW
  cvs.width = DETECT_WIDTH; cvs.height = Math.round(srcH * scale)
  const ctx = cvs.getContext('2d')
  const focus = []
  for (let i = 0; i < times.length; i++) {
    let f = null
    try {
      await seekTo(video, Math.min(clip.start + times[i], Math.max(0, video.duration - 0.05)))
      ctx.drawImage(video, 0, 0, cvs.width, cvs.height)
      const blob = await new Promise(r => cvs.toBlob(r, 'image/jpeg', 0.85))
      const dets = await detector(blob, { threshold: 0.5 })
      const people = dets.filter(d => d.label === 'person').map(d => ({
        x0: d.box.xmin / scale, x1: d.box.xmax / scale, y0: d.box.ymin / scale, y1: d.box.ymax / scale
      }))
      f = pickFocus(people, axis, axis === 'x' ? cw : ch, srcW, srcH)
    } catch (e) { console.warn('deteksi gagal di', times[i], e) }
    focus.push(f)
    onProgress?.(Math.round((i + 1) / times.length * 100))
  }
  const pos = smoothPath(focus, SAMPLE_STEP, axis === 'x' ? cw : ch, axis === 'x' ? srcW : srcH)
  return { times, pos, found: focus.filter(v => v != null).length }
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

function groupHighlights(chunks, totalDuration) {
  const usable = (chunks || []).map(c => {
    const s = c.timestamp?.[0] ?? c.start ?? 0
    const e = c.timestamp?.[1] ?? c.end
    return {
      start: Number(s),
      end: e == null ? Number(totalDuration) || Number(s) : Number(e), // chunk terakhir sering end = null
      text: (c.text || '').trim()
    }
  }).filter(c => c.text && Number.isFinite(c.start) && Number.isFinite(c.end) && c.end > c.start)
  if (!usable.length) return []

  // Gabungkan segmen berdekatan menjadi kandidat klip ±28–65 detik.
  const groups = []; let current = null
  for (const seg of usable) {
    if (!current) current = { start: seg.start, end: seg.end, text: seg.text, parts: [seg] }
    else if (seg.start - current.start <= 55 && seg.end - current.start <= 65) {
      current.end = seg.end; current.text += ' ' + seg.text; current.parts.push(seg)
    } else { groups.push(current); current = { start: seg.start, end: seg.end, text: seg.text, parts: [seg] } }
    if (current && current.end - current.start >= 28) { groups.push(current); current = null }
  }
  if (current) groups.push(current)

  const terms = ['karena', 'tetapi', 'namun', 'penting', 'harus', 'jangan', 'ternyata', 'menurut', 'masyarakat', 'generasi', 'kesimpulan', 'pertama', 'kedua', 'berarti', 'solusi', 'masalah', 'kita', 'saya', 'anda', 'bagaimana', 'mengapa']
  return groups.filter(g => g.end - g.start >= 8).map(g => {
    const lower = g.text.toLowerCase()
    const hits = terms.filter(t => lower.includes(t)).length
    const words = g.text.split(/\s+/).length
    const score = Math.min(99, Math.round(45 + Math.min(words, 90) * .35 + hits * 3 + (/[!?]/.test(g.text) ? 5 : 0)))
    return {
      ...g, id: crypto.randomUUID(), score, rendered: false, url: ''
    }
  }).sort((a, b) => b.score - a.score).slice(0, 12)
    .map((g, i) => ({ ...g, title: `Highlight ${String(i + 1).padStart(2, '0')}` })) // nomor mengikuti urutan skor
}

function App() {
  const [file, setFile] = useState(null), [url, setUrl] = useState(''), [duration, setDuration] = useState(0)
  const [clips, setClips] = useState([]), [asr, setAsr] = useState(null)
  const [busy, setBusy] = useState(false), [progress, setProgress] = useState(0), [phase, setPhase] = useState('')
  const [status, setStatus] = useState('Upload rekaman siaran untuk memulai.'), [error, setError] = useState('')
  const [modelReady, setModelReady] = useState(false)
  const videoRef = useRef(null), inputRef = useRef(null)
  const ffmpegRef = useRef(null), detectorRef = useRef(null), dimsRef = useRef({ w: 0, h: 0 }), urlRef = useRef(''), clipsRef = useRef([])

  urlRef.current = url
  clipsRef.current = clips

  // Bebaskan memori saat tab ditutup / komponen dilepas.
  useEffect(() => () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    clipsRef.current.forEach(c => c.url && URL.revokeObjectURL(c.url))
    ffmpegRef.current?.terminate()
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
      const candidates = groupHighlights(result.chunks || [], duration)
      if (!candidates.length) throw new Error('Tidak ditemukan kandidat highlight. Pastikan video memiliki audio percakapan yang jelas.')
      setClips(candidates)

      // 4. Auto-reframe + render otomatis semua kandidat
      const { w: srcW, h: srcH } = dimsRef.current
      const geo = cropGeometry(srcW, srcH)

      let detector = detectorRef.current
      if (!detector && geo.axis) {
        setPhase('Memuat AI deteksi subjek'); setProgress(0)
        setStatus('Mengunduh model deteksi subjek untuk auto-reframe...')
        try {
          const cb = p => { if (p.status === 'progress' && p.total) setProgress(Math.round(p.loaded / p.total * 100)) }
          try { detector = await pipeline('object-detection', DETECT_MODEL, { dtype: 'q8', progress_callback: cb }) }
          catch { detector = await pipeline('object-detection', DETECT_MODEL, { progress_callback: cb }) }
          detectorRef.current = detector
        } catch (e) {
          console.warn('Model deteksi gagal dimuat, memakai crop tengah.', e)
          detector = null
        }
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
              const centered = (axisLen - cropLen) / 2
              track = { times: [0, len], pos: [centered, centered] }
              if (detector) {
                setPhase(`Reframing klip ${i + 1}/${candidates.length}`); setProgress(0)
                setStatus(`Mencari subjek utama di ${c.title}...`)
                track = await trackSubject({ video: probe, detector, clip: c, end, geo, srcW, srcH, onProgress: setProgress })
              }
            }

            // b. potong + crop 9:16 + encode
            setPhase(`Render klip ${i + 1}/${candidates.length}`); setProgress(0)
            setStatus(`Membuat video portrait: ${c.title} (${fmt(c.start)} – ${fmt(c.end)})...`)
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

  function update(id, patch) { setClips(prev => prev.map(c => c.id === id ? { ...c, ...patch } : c)) }
  function download(c) {
    if (!c.url) return
    const a = document.createElement('a'); a.href = c.url; a.download = `${safe(c.title)}.mp4`; a.click()
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
          <div className="ai-card"><div className="ai-icon"><WandSparkles /></div><div><b>Whisper AI · Multilingual</b><p>Transkripsi Bahasa Indonesia, deteksi highlight, lalu klip otomatis dipotong portrait 9:16 dengan auto-reframe mengikuti subjek.</p></div></div>
          <div className="modelrow"><span><Cpu size={15} /> AI model</span><b>{modelReady ? 'Loaded in browser' : 'Whisper Tiny · multilingual'}</b></div>
          <button className="primary" onClick={transcribe} disabled={!file || !duration || busy}><Sparkles size={16} />{busy ? 'AI sedang bekerja...' : 'Generate AI Highlights'}</button>
          <p className="hint">Model AI diunduh saat pertama kali digunakan dan tersimpan di cache browser. Jangan tutup tab selama proses berjalan.</p>
        </div>
      </section>
      <section className="panel results">
        <div className="resulthead"><div><small>03 / AI OUTPUT</small><h2>Highlight results <em>{clips.length}</em></h2><p>Klip dirender otomatis dalam format portrait 9:16 dengan auto-reframe pada subjek. Unduh MP4 yang diinginkan.</p></div><button className="outline" onClick={clear} disabled={!file || busy}><Trash2 size={14} /> Clear</button></div>
        {!clips.length ? <div className="empty"><div><Scissors size={25} /></div><b>No highlights generated</b><span>Upload video lalu jalankan AI untuk menemukan kandidat momen menarik.</span></div> :
          <div className="clipgrid">{clips.map((c, i) => <article className="clip chosen" key={c.id}>
            <div className="cliptop"><b>HIGHLIGHT {String(i + 1).padStart(2, '0')}</b><span className="score">AI {c.score}%</span></div>
            <input className="titleinput" value={c.title} onChange={e => update(c.id, { title: e.target.value })} />
            <div className="clipmeta"><span><Clock3 size={13} />{fmt(c.start)} – {fmt(c.end)}</span><span>{fmt(c.end - c.start)}</span></div>
            <div className="transcript">{c.text}</div>
            {c.url && <video className="preview" src={c.url} controls style={{ maxHeight: 460, aspectRatio: "9 / 16", margin: "0 auto", display: "block", background: "#000" }} />}
            <div className="clipbuttons">
              {c.rendered
                ? <button className="download" onClick={() => download(c)}><Download size={14} /> Download MP4</button>
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
