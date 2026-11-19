import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { FFmpeg } from '@ffmpeg/ffmpeg'
import { fetchFile, toBlobURL } from '@ffmpeg/util'
import { pipeline, env } from '@huggingface/transformers'
import { UploadCloud, Sparkles, Scissors, Captions, Download, Trash2, CheckCircle2, LoaderCircle, FileVideo2, WandSparkles, Subtitles, X, Clock3, Cpu, CheckSquare, Square } from 'lucide-react'
import './style.css'

env.allowLocalModels = false
env.useBrowserCache = true
const FFMPEG_CORE = 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm'
const ASR_MODEL = 'onnx-community/whisper-tiny'

const fmt = n => {
  n = Math.max(0, Number(n)||0)
  return `${String(Math.floor(n/60)).padStart(2,'0')}:${String(Math.floor(n%60)).padStart(2,'0')}`
}
const srtTime = n => {
  const ms=Math.floor((n%1)*1000), t=Math.floor(n)
  return `${String(Math.floor(t/3600)).padStart(2,'0')}:${String(Math.floor(t%3600/60)).padStart(2,'0')}:${String(t%60).padStart(2,'0')},${String(ms).padStart(3,'0')}`
}
const safe = s => (s||'rri-relix').replace(/\.[^.]+$/,'').replace(/[^a-z0-9-_]+/gi,'-').toLowerCase()

function groupHighlights(chunks, totalDuration) {
  const usable = (chunks||[]).map(c=>({
    start:Number(c.timestamp?.[0] ?? c.start ?? 0),
    end:Number(c.timestamp?.[1] ?? c.end ?? 0),
    text:(c.text||'').trim()
  })).filter(c=>c.text && Number.isFinite(c.start) && Number.isFinite(c.end) && c.end>c.start)
  if (!usable.length) return []
  // Group adjacent ASR segments into 25–55 second candidate clips.
  const groups=[]; let current=null
  for (const seg of usable) {
    if (!current) current={start:seg.start,end:seg.end,text:seg.text,parts:[seg]}
    else if (seg.start-current.start <= 55 && seg.end-current.start <= 65) {
      current.end=seg.end; current.text += ' '+seg.text; current.parts.push(seg)
    } else { groups.push(current); current={start:seg.start,end:seg.end,text:seg.text,parts:[seg]} }
    if (current && current.end-current.start >= 28) { groups.push(current); current=null }
  }
  if (current) groups.push(current)
  const terms=['karena','tetapi','namun','penting','harus','jangan','ternyata','menurut','masyarakat','generasi','kesimpulan','pertama','kedua','berarti','solusi','masalah','kita','saya','anda','bagaimana','mengapa']
  return groups.filter(g=>g.end-g.start>=8).map((g,i)=>{
    const lower=g.text.toLowerCase()
    const hits=terms.filter(t=>lower.includes(t)).length
    const words=g.text.split(/\s+/).length
    const score=Math.min(99, Math.round(45 + Math.min(words,90)*.35 + hits*3 + (/[!?]/.test(g.text)?5:0)))
    return {...g,id:crypto.randomUUID(),title:`Highlight ${String(i+1).padStart(2,'0')}`,score,selected:true,subtitle:g.parts.map(p=>({start:p.start-g.start,end:p.end-g.start,text:p.text}))}
  }).sort((a,b)=>b.score-a.score).slice(0,12)
}

function App(){
 const [file,setFile]=useState(null),[url,setUrl]=useState(''),[duration,setDuration]=useState(0)
 const [clips,setClips]=useState([]),[asr,setAsr]=useState(null),[ffmpeg,setFfmpeg]=useState(null)
 const [busy,setBusy]=useState(false),[progress,setProgress]=useState(0),[phase,setPhase]=useState('')
 const [status,setStatus]=useState('Upload rekaman siaran untuk memulai.'),[error,setError]=useState('')
 const [language,setLanguage]=useState('indonesian'),[modelReady,setModelReady]=useState(false)
 const videoRef=useRef(null),inputRef=useRef(null)

 useEffect(()=>()=>{if(url)URL.revokeObjectURL(url);clips.forEach(c=>c.url&&URL.revokeObjectURL(c.url))},[])
 function selectFile(f){
  if(!f)return
  if(!f.type.startsWith('video/')){setError('File harus berupa video.');return}
  if(url)URL.revokeObjectURL(url)
  clips.forEach(c=>c.url&&URL.revokeObjectURL(c.url))
  setFile(f);setUrl(URL.createObjectURL(f));setClips([]);setError('')
  setStatus('Video siap dianalisis. Pemrosesan dilakukan di browser.')
 }
 function loaded(){setDuration(videoRef.current?.duration||0)}
 async function getFFmpeg(){
  if(ffmpeg)return ffmpeg
  const f=new FFmpeg()
  f.on('progress',({progress})=>setProgress(Math.round(progress*100)))
  await f.load({coreURL:await toBlobURL(`${FFMPEG_CORE}/ffmpeg-core.js`,'text/javascript'),wasmURL:await toBlobURL(`${FFMPEG_CORE}/ffmpeg-core.wasm`,'application/wasm')})
  setFfmpeg(f);return f
 }
 async function transcribe(){
  if(!file)return
  setBusy(true);setError('');setProgress(0)
  try{
   setPhase('Memuat model AI Whisper');setStatus('Mengunduh model Whisper multilingual. Pertama kali dapat memerlukan waktu dan kuota data cukup besar.')
   let transcriber=asr
   if(!transcriber){
    transcriber=await pipeline('automatic-speech-recognition',ASR_MODEL,{dtype:'q8',progress_callback:p=>{
      if(p.status==='progress'&&p.total)setProgress(Math.round(p.loaded/p.total*100))
      if(p.status==='downloading')setStatus(`Mengunduh model AI: ${p.file||'model'}...`)
    }})
    setAsr(transcriber);setModelReady(true)
   }
   setPhase('Ekstrak audio');setProgress(0);setStatus('Mengekstrak audio dari video...')
   const engine=await getFFmpeg()
   await engine.writeFile('relix-source',await fetchFile(file))
   await engine.exec(['-i','relix-source','-vn','-ac','1','-ar','16000','-c:a','pcm_s16le','relix-audio.wav'])
   const audio=await engine.readFile('relix-audio.wav')
   setPhase('AI transcription');setStatus('Whisper sedang mengenali percakapan. Durasi proses bergantung pada perangkat.')
   const result=await transcriber(new Blob([audio.buffer],{type:'audio/wav'}),{
    chunk_length_s:30,stride_length_s:5,return_timestamps:'chunk',language:'indonesian',task:'transcribe'
   })
   const candidates=groupHighlights(result.chunks||[],duration)
   if(!candidates.length)throw new Error('Transkrip tidak menghasilkan segmen bertimestamp. Coba video dengan audio percakapan yang jelas.')
   setClips(candidates);setStatus(`Analisis selesai. ${candidates.length} kandidat highlight dibuat. Pilih klip yang ingin diunduh.`)
   await engine.deleteFile('relix-source').catch(()=>{});await engine.deleteFile('relix-audio.wav').catch(()=>{})
  }catch(e){setError(`Analisis AI gagal: ${e?.message||e}`);setStatus('Periksa format video, koneksi internet, dan kapasitas memori browser.')}
  finally{setBusy(false);setPhase('');setProgress(0)}
 }
 async function renderClip(c){
  setBusy(true);setError('');setPhase(`Render ${c.title}`);setProgress(0)
  try{
   const engine=await getFFmpeg()
   await engine.writeFile('relix-source',await fetchFile(file))
   const out=`clip-${c.id}.mp4`
   await engine.exec(['-ss',String(c.start),'-i','relix-source','-t',String(c.end-c.start),'-map','0:v:0','-map','0:a?','-c:v','libx264','-preset','ultrafast','-crf','23','-c:a','aac','-b:a','128k','-movflags','+faststart',out])
   const data=await engine.readFile(out),blob=new Blob([data.buffer],{type:'video/mp4'}),obj=URL.createObjectURL(blob)
   setClips(prev=>prev.map(x=>x.id===c.id?{...x,blob,url:obj,rendered:true}:x))
   await engine.deleteFile(out).catch(()=>{});await engine.deleteFile('relix-source').catch(()=>{})
   setStatus(`${c.title} selesai dirender dan siap diunduh.`)
  }catch(e){setError(`Render gagal: ${e?.message||e}`)}
  finally{setBusy(false);setPhase('');setProgress(0)}
 }
 async function renderSelected(){
  for(const c of clips.filter(x=>x.selected&&!x.rendered)){await renderClip(c)}
 }
 function update(id,patch){setClips(prev=>prev.map(c=>c.id===id?{...c,...patch}:c))}
 function download(c){
  if(!c.url)return
  const a=document.createElement('a');a.href=c.url;a.download=`${safe(c.title)}.mp4`;a.click()
 }
 function downloadSrt(c){
  const body=c.subtitle.map((s,i)=>`${i+1}\n${srtTime(s.start)} --> ${srtTime(s.end)}\n${s.text}\n`).join('\n')
  const a=document.createElement('a');const u=URL.createObjectURL(new Blob([body],{type:'text/plain;charset=utf-8'}));a.href=u;a.download=`${safe(c.title)}.srt`;a.click();URL.revokeObjectURL(u)
 }
 function clear(){
  if(url)URL.revokeObjectURL(url);clips.forEach(c=>c.url&&URL.revokeObjectURL(c.url))
  setFile(null);setUrl('');setClips([]);setDuration(0);setStatus('Workspace dibersihkan.');setError('')
 }
 const selected=clips.filter(c=>c.selected)
 return <div className="shell">
  <header className="nav"><div className="brand"><div className="mark">R</div><div><b>RRI <i>RELIX</i></b><small>by line</small></div></div><div className="navright"><span className="dot"/> AI BROWSER WORKSPACE <span className="org">LPP RRI BANJARMASIN</span></div></header>
  <main>
   <section className="hero"><div className="eyebrow"><Sparkles size={14}/> AI-POWERED PRODUCTION WORKSPACE</div><h1>Turn your broadcast<br/><span>into content.</span></h1><p>AI-Powered Broadcast Content &amp; Production Studio</p></section>
   <section className="grid">
    <div className="panel"><div className="panelhead"><div><small>01 / SOURCE</small><h2>Video source</h2></div><FileVideo2/></div>
     {!file?<button className="drop" onClick={()=>inputRef.current?.click()} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();selectFile(e.dataTransfer.files?.[0])}}><UploadCloud size={34}/><b>Upload broadcast video</b><span>Drag &amp; drop atau pilih file dari perangkat</span><strong>Select video</strong><small>MP4 · MOV · WEBM · Tidak diunggah ke server</small></button>:<><video className="sourcevideo" ref={videoRef} src={url} controls onLoadedMetadata={loaded}/><div className="fileline"><b>{file.name}</b><span>{(file.size/1048576).toFixed(1)} MB · {fmt(duration)}</span><button className="ghost" onClick={clear}><X size={16}/></button></div></>}
     <input ref={inputRef} hidden type="file" accept="video/*" onChange={e=>selectFile(e.target.files?.[0])}/>
    </div>
    <div className="panel"><div className="panelhead"><div><small>02 / AI ENGINE</small><h2>Analyze broadcast</h2></div><Sparkles/></div>
     <div className="ai-card"><div className="ai-icon"><WandSparkles/></div><div><b>Whisper AI · Multilingual</b><p>Transkripsi otomatis Bahasa Indonesia dan deteksi kandidat highlight dari segmen percakapan.</p></div></div>
     <div className="modelrow"><span><Cpu size={15}/> AI model</span><b>{modelReady?'Loaded in browser':'Whisper Tiny · multilingual'}</b></div>
     <button className="primary" onClick={transcribe} disabled={!file||busy}><Sparkles size={16}/>{busy?'AI sedang bekerja...':'Generate AI Highlights'}</button>
     <p className="hint">Model AI diunduh saat pertama kali digunakan dan dapat tersimpan di cache browser. Ukuran unduhan model cukup besar.</p>
    </div>
   </section>
   <section className="panel results">
    <div className="resulthead"><div><small>03 / AI OUTPUT</small><h2>Highlight results <em>{clips.length}</em></h2><p>Pilih klip yang akan dirender dan diunduh.</p></div><button className="outline" onClick={clear} disabled={!file}><Trash2 size={14}/> Clear</button></div>
    {!clips.length?<div className="empty"><div><Scissors size={25}/></div><b>No highlights generated</b><span>Upload video lalu jalankan AI untuk menemukan kandidat momen menarik.</span></div>:
     <>
      <div className="selectionbar"><button className="textbtn" onClick={()=>setClips(prev=>prev.map(c=>({...c,selected:true})))}><CheckSquare size={15}/> Select all</button><button className="textbtn" onClick={()=>setClips(prev=>prev.map(c=>({...c,selected:false})))}><Square size={15}/> Deselect all</button><span>{selected.length} of {clips.length} selected</span></div>
      <div className="clipgrid">{clips.map((c,i)=><article className={`clip ${c.selected?'chosen':''}`} key={c.id}>
       <div className="cliptop"><label><input type="checkbox" checked={c.selected} onChange={e=>update(c.id,{selected:e.target.checked})}/><span className="checkvisual"><CheckCircle2 size={16}/></span></label><b>HIGHLIGHT {String(i+1).padStart(2,'0')}</b><span className="score">AI {c.score}%</span></div>
       <input className="titleinput" value={c.title} onChange={e=>update(c.id,{title:e.target.value})}/>
       <div className="clipmeta"><span><Clock3 size={13}/>{fmt(c.start)} – {fmt(c.end)}</span><span>{fmt(c.end-c.start)}</span></div>
       <div className="transcript">{c.text}</div>
       <details><summary>Edit subtitle transcript</summary><textarea value={c.subtitle.map(s=>s.text).join('\n')} onChange={e=>{const lines=e.target.value.split('\n');update(c.id,{subtitle:c.subtitle.map((s,j)=>({...s,text:lines[j]??''}))})}}/></details>
       {c.url&&<video className="preview" src={c.url} controls/>}
       <div className="clipbuttons">{c.rendered?<button className="download" onClick={()=>download(c)}><Download size={14}/> Download MP4</button>:<button className="render" disabled={busy||!c.selected} onClick={()=>renderClip(c)}><Scissors size={14}/> Render MP4</button>}<button className="srt" onClick={()=>downloadSrt(c)}><Captions size={14}/> SRT</button></div>
      </article>)}</div>
      <div className="bulk"><button className="primary" onClick={renderSelected} disabled={busy||!selected.some(c=>!c.rendered)}><Download size={16}/> Render selected ({selected.length})</button><span>Hanya klip terpilih yang akan diproses. Download MP4 per klip.</span></div>
     </>}
    {busy&&<div className="progress"><div><span><LoaderCircle className="spin" size={14}/> {phase||'Processing'}</span><b>{progress}%</b></div><div className="track"><i style={{width:`${progress}%`}}/></div></div>}
   </section>
   <div className="privacy"><CheckCircle2 size={18}/><div><b>Privacy-first · Browser processing</b><p>Video mentah tidak dikirim ke server aplikasi. AI inference dan rendering berlangsung di browser. Hasil klip hanya tersedia sementara di sesi browser; unduh sebelum membersihkan workspace atau menutup tab.</p></div></div>
   {status&&<p className="status">{status}</p>}{error&&<p className="error">{error}</p>}
  </main>
  <footer><b>RRI RELIX <span>by line</span></b><small>AI-Powered Broadcast Content &amp; Production Studio</small><small>© LPP RRI Banjarmasin</small></footer>
 </div>
}
createRoot(document.getElementById('root')).render(<App/>)
