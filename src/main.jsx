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
// 'blur'  = video utuh di tengah, latar blur (tidak ada bagian yang terpotong)
// 'crop'  = video memenuhi layar, sisi kiri/kanan terpotong
const PORTRAIT_MODE = 'blur'
const SUB_H = 260                 // tinggi gambar subtitle
const SUB_BOTTOM = 190            // jarak subtitle dari tepi bawah

const fmt = n => {
  n = Math.max(0, Number(n) || 0)
  return `${String(Math.floor(n / 60)).padStart(2, '0')}:${String(Math.floor(n % 60)).padStart(2, '0')}`
}
// Pecah segmen Whisper menjadi cue pendek (maks. ~7 kata) dengan waktu proporsional.
function makeCues(subtitle, clipLen, maxWords = 7) {
  const cues = []
  for (const s of subtitle || []) {
    const words = (s.text || '').split(/\s+/).filter(Boolean)
    if (!words.length) continue
    const n = Math.ceil(words.length / maxWords), size = Math.ceil(words.length / n)
    const groups = []
    for (let i = 0; i < words.length; i += size) groups.push(words.slice(i, i + size).join(' '))
    const total = groups.reduce((a, g) => a + g.length, 0)
    const segEnd = Math.min(s.end, clipLen)
    let t = Math.max(0, s.start)
    const dur = Math.max(0.3, segEnd - t)
    for (const g of groups) {
      const share = dur * g.length / total
      cues.push({ start: t, end: Math.min(clipLen, t + share), text: g })
      t += share
    }
  }
  return cues.filter(c => c.end - c.start > 0.1)
}

// Gambar teks subtitle (putih, outline hitam) ke PNG transparan memakai canvas browser,
// jadi tidak butuh font/libass di FFmpeg.
async function cueToPng(text) {
  const cv = document.createElement('canvas')
  cv.width = OUT_W; cv.height = SUB_H
  const ctx = cv.getContext('2d')
  const fontSize = 40, lineH = 50, maxW = OUT_W - 80
  ctx.font = `800 ${fontSize}px "Segoe UI", Roboto, Arial, sans-serif`
  ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'
  ctx.lineJoin = 'round'
  const lines = []; let line = ''
  for (const w of text.split(' ')) {
    const test = line ? line + ' ' + w : w
    if (line && ctx.measureText(test).width > maxW) { lines.push(line); line = w } else line = test
  }
  if (line) lines.push(line)
  const shown = lines.slice(-4)
  // rata bawah: baris terakhir selalu di posisi yang sama
  let y = SUB_H - 24 - (shown.length - 1) * lineH
  for (const l of shown) {
    ctx.lineWidth = 9; ctx.strokeStyle = 'rgba(0,0,0,.95)'; ctx.strokeText(l, OUT_W / 2, y)
    ctx.fillStyle = '#fff'; ctx.fillText(l, OUT_W / 2, y)
    y += lineH
  }
  const blob = await new Promise(r => cv.toBlob(r, 'image/png'))
  return new Uint8Array(await blob.arrayBuffer())
}

// Filter FFmpeg: ubah ke portrait 9:16 lalu tempel gambar subtitle sesuai waktu.
function buildFilter(cues) {
  const base = PORTRAIT_MODE === 'crop'
    ? `[0:v:0]scale=${OUT_W}:${OUT_H}:force_original_aspect_ratio=increase,crop=${OUT_W}:${OUT_H},setsar=1[v0]`
    : `[0:v:0]split=2[a][b];` +
      `[a]scale=90:160:force_original_aspect_ratio=increase,crop=90:160,boxblur=2:1,scale=${OUT_W}:${OUT_H},setsar=1[bg];` +
      `[b]scale=${OUT_W}:${OUT_H}:force_original_aspect_ratio=decrease,setsar=1[fg];` +
      `[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]`
  const y = OUT_H - SUB_H - SUB_BOTTOM
  let g = base
  cues.forEach((c, i) => {
    g += `;[v${i}][${i + 1}:v]overlay=0:${y}:enable='between(t,${c.start.toFixed(2)},${c.end.toFixed(2)})'[v${i + 1}]`
  })
  g += `;[v${cues.length}]format=yuv420p[vout]`
  return g
}
const safe = s => (s || 'rri-relix').replace(/\.[^.]+$/, '').replace(/[^a-z0-9-_]+/gi, '-').toLowerCase()

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
      ...g, id: crypto.randomUUID(), score, rendered: false, url: '',
      subtitle: g.parts.map(p => ({ start: p.start - g.start, end: p.end - g.start, text: p.text }))
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
  const ffmpegRef = useRef(null), urlRef = useRef(''), clipsRef = useRef([])

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

      // 4. Render otomatis semua kandidat
      let failed = 0
      for (let i = 0; i < candidates.length; i++) {
        const c = candidates[i], out = `clip-${i}.mp4`
        setPhase(`Render klip ${i + 1}/${candidates.length}`); setProgress(0)
        setStatus(`Membuat video portrait + subtitle: ${c.title} (${fmt(c.start)} – ${fmt(c.end)})...`)
        const pngs = []
        try {
          const end = duration ? Math.min(c.end, duration) : c.end
          const len = end - c.start
          const cues = makeCues(c.subtitle, len)
          const inputs = ['-ss', String(c.start), '-i', src]
          for (let k = 0; k < cues.length; k++) {
            const name = `cue-${i}-${k}.png`
            await engine.writeFile(name, await cueToPng(cues[k].text))
            pngs.push(name)
            inputs.push('-i', name)
          }
          const rc2 = await engine.exec([
            ...inputs, '-t', String(len),
            '-filter_complex', buildFilter(cues),
            '-map', '[vout]', '-map', '0:a?',
            '-r', '30',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '25',
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
          for (const n of pngs) await engine.deleteFile(n).catch(() => {})
        }
      }
      setStatus(failed
        ? `Selesai. ${candidates.length - failed} klip berhasil, ${failed} gagal dirender.`
        : `${candidates.length} klip selesai dirender. Silakan unduh hasilnya.`)
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
          <div className="ai-card"><div className="ai-icon"><WandSparkles /></div><div><b>Whisper AI · Multilingual</b><p>Transkripsi Bahasa Indonesia, deteksi highlight, lalu klip otomatis dibuat portrait 9:16 dengan subtitle tertanam.</p></div></div>
          <div className="modelrow"><span><Cpu size={15} /> AI model</span><b>{modelReady ? 'Loaded in browser' : 'Whisper Tiny · multilingual'}</b></div>
          <button className="primary" onClick={transcribe} disabled={!file || !duration || busy}><Sparkles size={16} />{busy ? 'AI sedang bekerja...' : 'Generate AI Highlights'}</button>
          <p className="hint">Model AI diunduh saat pertama kali digunakan dan tersimpan di cache browser. Jangan tutup tab selama proses berjalan.</p>
        </div>
      </section>
      <section className="panel results">
        <div className="resulthead"><div><small>03 / AI OUTPUT</small><h2>Highlight results <em>{clips.length}</em></h2><p>Klip dirender otomatis dalam format portrait 9:16 dengan subtitle tertanam. Unduh MP4 yang diinginkan.</p></div><button className="outline" onClick={clear} disabled={!file || busy}><Trash2 size={14} /> Clear</button></div>
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
