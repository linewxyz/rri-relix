# RRI RELIX — AI MVP

**by line**  
**AI-Powered Broadcast Content & Production Studio**

## Fitur
- Upload video manual.
- Transkripsi otomatis menggunakan Whisper multilingual (`onnx-community/whisper-tiny`) di browser.
- Membuat kandidat highlight dari segmen transkrip.
- Skor kandidat berbasis sinyal teks sederhana (bukan penilaian semantik LLM).
- Pilih / batalkan pilihan klip satu per satu atau Select All.
- Render MP4 hanya untuk klip yang dipilih.
- Download MP4 dan subtitle SRT per klip.
- Edit judul dan teks subtitle sebelum unduh.
- Tidak ada upload video mentah ke server aplikasi.

## Batasan penting
Ini adalah AI-assisted MVP. Whisper menjalankan AI speech recognition. Pemilihan highlight saat ini mengelompokkan segmen transkrip dan memberi skor heuristik berbasis teks; belum menggunakan LLM untuk memahami konteks secara mendalam. Subtitle dapat diunduh sebagai SRT terpisah dan belum dibakar permanen ke dalam MP4. Video klip MP4 hasil render berisi potongan audio/video asli.

Model Whisper diunduh dari Hugging Face pada penggunaan pertama dan dapat berukuran besar. Waktu proses bergantung pada RAM, CPU/GPU browser, durasi, dan format video. Disarankan Chrome/Edge desktop, RAM minimal 8 GB, serta koneksi internet stabil.

## Deploy gratis
1. Buat repository GitHub bernama `rri-relix`.
2. Upload isi folder proyek ini ke root repository.
3. Import repository di Vercel.
4. Framework preset: **Vite**.
5. Build command: `npm run build`.
6. Output directory: `dist`.
7. Deploy.

Header COOP/COEP di `vercel.json` dan Vite config dibutuhkan agar FFmpeg WebAssembly dapat berjalan dengan SharedArrayBuffer.

## Biaya
Kode aplikasi dan library tidak memerlukan lisensi berbayar. Vercel Free dan Hugging Face CDN tunduk pada batas dan kebijakan masing-masing. Tidak ada backend encoding berbayar karena inferensi dan render berjalan di browser.

## Privasi
File video diproses lokal di browser, tidak dikirim ke server aplikasi. Hasil render berada sementara dalam sesi browser. Unduh semua hasil yang dibutuhkan sebelum menutup tab atau membersihkan workspace.
