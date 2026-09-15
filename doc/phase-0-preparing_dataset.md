# Fase 0 — Persiapan Dataset

> Turunan dari `doc/main-documentation.md` Bagian 10. Fokus dokumen ini: detail setup yang perlu selesai sebelum (dan selama) notebook `01_data_preparation.ipynb` ditulis — environment, requirements.txt, dan konfirmasi struktur dataset aktual.

## 0. Ringkasan Fase

|                    |                                                                                |
| ------------------ | ------------------------------------------------------------------------------ |
| Tujuan             | Dataset final 850 citra, 5 kelas Bahasa Indonesia, siap pakai                  |
| Ref. Proposal      | 3.3.3, Tabel 3.2                                                               |
| Ref. Dokumen Utama | `doc/main-documentation.md` Bagian 4.1                                         |
| Prasyarat          | Tidak ada — pemetaan folder → label sudah dipastikan benar, tinggal diterapkan |
| Output             | `dataset/processed/`, `notebooks/01_data_preparation.ipynb`                    |
| Status             | Perlu Diulang — koreksi kualitas data ditemukan 14 September 2026 (lihat Bagian 8)                        |

## 1. Strategi Pengembangan: Hybrid VS Code + Kaggle

Pendekatan hybrid sudah tepat untuk skala proyek ini:

- **Fase 0–1 (persiapan data, pra-pemrosesan) → VS Code, lokal.** Tidak butuh GPU — murni image I/O, resize, dan operasi Pandas/NumPy. Iterasi dan debugging jauh lebih cepat lokal dibanding di notebook cloud.
- **Fase 2 dst. (training) → Kaggle Notebook.** Baru di titik ini GPU (T4×2) dibutuhkan.

**Titik serah-terima yang perlu direncanakan dari sekarang (bukan pekerjaan Fase 0, tapi dampak dari keputusan Fase 0):** hasil `dataset/processed/` tidak otomatis bisa diakses notebook di Kaggle. Sebelum Fase 2 mulai, folder ini perlu di-upload sebagai **Kaggle Dataset privat** (via `kaggle datasets create` atau web UI), baru direferensikan sebagai input di notebook training. Dicatat di sini supaya tidak terlupa saat Fase 1 selesai.

## 2. Environment Aktual

### 2.1 Lokasi Proyek

```
TA_Mila/
├── ProposalPenelitian_MilaLestari.pdf
├── Repository/
│   └── TUGAS-AKHIR/        # git repo (working directory utama)
├── Thesis/
└── tugas_akhir/            # virtual environment (sibling dari Repository/)
```

### 2.2 Virtual Environment

Sudah dibuat (`tugas_akhir`). Aktivasi (dari `TA_Mila/`):

```bash
source tugas_akhir/bin/activate
cd Repository/TUGAS-AKHIR
```

### 2.3 Setup VS Code

1. Buka folder `Repository/TUGAS-AKHIR` di VS Code
2. Install extension **Python** dan **Jupyter** (jika belum)
3. `Ctrl+Shift+P` → **Python: Select Interpreter** → pilih interpreter di dalam `tugas_akhir/`
4. Daftarkan kernel Jupyter untuk venv ini:
   ```bash
   python -m ipykernel install --user --name=tugas-akhir --display-name "TA Mila (tugas_akhir)"
   ```
5. Saat membuka `notebooks/01_data_preparation.ipynb`, pilih kernel **"TA Mila (tugas_akhir)"** di pojok kanan atas

## 3. `requirements.txt` — Fase 0-1 (Minimal)

Diisi bertahap, bukan lengkap sekaligus. Untuk Fase 0-1, cukup:

```
pillow
numpy
pandas
matplotlib
seaborn
jupyter
ipykernel
```

Install:

```bash
pip install -r requirements.txt
```

`scikit-learn` ditambah saat Fase 1 (butuh `train_test_split`); `tensorflow`, `streamlit`, `kaggle` ditambah saat Fase 2 dan 4 — tidak relevan untuk notebook Fase 0 ini.

## 4. Dataset Sumber — Struktur Aktual di Disk

Struktur folder `dataset/raw/` yang sudah dikonfirmasi (perhatikan: penomoran Source 1/2 di sini **kebalikan** dari asumsi awal — sudah dikoreksi juga di `doc/main-documentation.md`):

| Folder Disk                                  | Isi Dataset Asli                                                     | Kelas Dipakai → Kelas Final                   | Jumlah Dipakai                          |
| -------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------- |
| `dataset/raw/Source 1 (vijayvkb98)/`         | DS Dataset (7 kelas: Rainy, Sunrise, Sand, Cloudy, Fog, Shine, Snow) | Cloudy → Berawan, Fog → Berkabut              | 170 + 170 (sampling, `random_state=42`) |
| `dataset/raw/Source 2 (jonathanvitotaufik)/` | Weather Image Dataset (Cerah, Hujan, Mendung)                        | Cerah, Hujan, Mendung → Cerah, Hujan, Mendung | 170 + 170 + 170 (semua)                 |

Kelas Rainy, Sunrise, Sand, Shine, Snow pada Source 1 tidak dipakai (tidak relevan / sudah terwakili Source 2), sesuai Proposal 3.3.3.

**Pemetaan folder → label sudah dipastikan benar** — notebook Fase 0 tidak perlu langkah eksplorasi/verifikasi visual tambahan, langsung diterapkan.

## 5. Rencana Isi Notebook `notebooks/01_data_preparation.ipynb`

1. **Konfigurasi** — path sumber & tujuan, daftar 5 kelas final, `random_state=42`
2. **Salin Source 2 (jonathanvitotaufik)** — Cerah, Hujan, Mendung, seluruh 170 citra per kelas disalin apa adanya
3. **Sampling Source 1 (vijayvkb98)** — hanya folder Cloudy dan Fog; sampling acak 170 dari masing-masing (`random_state=42`); kelas lain diabaikan
4. **Simpan ke `dataset/processed/{Kelas}/`** — 5 folder kelas Bahasa Indonesia (Cerah, Mendung, Hujan, Berawan, Berkabut); rename file jadi `{kelas}_{urutan:04d}.{ext}` agar tidak ada bentrok nama antar sumber
5. **Manifest** — tulis `dataset/processed/manifest.csv` (kolom: `filename_baru`, `kelas`, `sumber_asal`, `filename_asli`) untuk keterlacakan
6. **Sanity check** — hitung ulang jumlah tiap kelas (harus 170) dan total (harus 850); coba buka tiap file dengan `PIL.Image.open()` untuk pastikan tidak ada file korup
7. **(Opsional) Visualisasi cepat** — grid sampel citra per kelas + bar chart distribusi kelas, simpan ke `results/figures/` (bahan BAB IV nanti)

Notebook ini **tidak** melakukan resize/normalisasi/augmentasi — itu bagian Fase 1 (Ref. 3.3.4).

## 6. Output & Definition of Done

- [ ] `dataset/processed/Cerah/` — 170 file valid
- [ ] `dataset/processed/Mendung/` — 170 file valid
- [ ] `dataset/processed/Hujan/` — 170 file valid
- [ ] `dataset/processed/Berawan/` — 170 file valid
- [ ] `dataset/processed/Berkabut/` — 170 file valid
- [ ] Total 850 citra, tidak ada file korup
- [ ] `dataset/processed/manifest.csv` tersedia
- [ ] `notebooks/01_data_preparation.ipynb` jalan top-to-bottom tanpa error di kernel venv `tugas_akhir`
- [ ] Update status Fase 0 di `doc/main-documentation.md` Bagian 10 → Selesai

## 8. Update — Koreksi Kualitas Data (Temuan Manual, 14 September 2026)

### 8.1 Temuan

Hasil pengecekan visual manual terhadap dataset sumber (bukan pengecekan otomatis — ini temuan mata langsung dari Anda) menemukan dua masalah di `dataset/raw/Source 1 (vijayvkb98)/Cloudy/`:

1. **Cross-class mislabeling** — sejumlah citra di folder `Cloudy` secara visual jelas menunjukkan kondisi berkabut (visibilitas sangat terbatas karena kabut tebal), bukan berawan biasa. Dua contoh terkonfirmasi:
   - `cloudy159.jpg` — bandara nyaris tak terlihat karena kabut tebal
   - `cloudy54.jpg` — jalan dengan visibilitas sangat terbatas, siluet pohon karena kabut
2. **File duplikat dengan penamaan bergaya OS** — pola `cloudy48.jpg`, `cloudy48(1).jpg`, `cloudy48(2).jpg` mengindikasikan file yang sama diunduh/disalin berulang tanpa overwrite. Berisiko **data leakage** kalau salinan yang mirip lolos sampling ke 170 yang dipakai lalu terpisah ke subset train dan test yang berbeda.

Sebagai observasi tambahan (bukan masalah): kelas `Foggy` di sumber yang sama memakai 3 pola nama berbeda (`foggy-*`, `haze-*`, `mist-*`) — kemungkinan digabung dari beberapa sub-sumber oleh kurator dataset asli. Ini secara semantik wajar (haze dan mist berdekatan dengan fog) dan tidak dianggap sebagai kesalahan, berbeda dari temuan #1 di atas yang jelas salah folder.

**Kemungkinan keterkaitan dengan hasil Fase 2:** confusion matrix EfficientNetB0 (seed=42) menunjukkan Berawan salah diklasifikasikan sebagai Berkabut (4%) dan Mendung (4%). Ada kemungkinan citra bermasalah di atas termasuk yang tersampel ke 170 dan berakhir di test split — jika benar, model sebenarnya "melihat" ciri kabut dengan benar, hanya label groundtruth-nya yang keliru. Perlu dicek terhadap `manifest.csv` versi lama sebagai bahan diskusi BAB IV nanti (terlepas dari apakah dataset diperbaiki atau tidak).

### 8.2 Keputusan

Diperbaiki sekarang (bukan didokumentasikan sebagai keterbatasan saja), karena ini titik termurah untuk redo: sebelum ResNet50 mulai dan sebelum seluruh 5 seed EfficientNetB0 selesai. Konsekuensi: `dataset/processed/` berubah → Fase 1 (`dataset/split/`) harus digenerate ulang → 4 seed EfficientNetB0 yang sudah dilatih (Fase 2) tidak valid lagi dengan dataset baru dan perlu diulang.

### 8.3 Mekanisme Koreksi (Reproducible, Bukan Geser File Manual)

Ditambahkan ke `notebooks/01_data_preparation.ipynb`, diterapkan **sebelum** langkah sampling `random_state=42`:

**Langkah 1 — Deduplikasi otomatis.** Hash (MD5) setiap file di seluruh `Source 1 (vijayvkb98)` (bukan cuma Cloudy — sekalian cek semua kelas untuk jaga-jaga). File dengan hash identik → simpan hanya satu (yang nama filenya tanpa suffix `(n)`), sisanya masuk daftar exclude otomatis.

**Langkah 2 — Contact sheet untuk audit manual.** Generate grid visual berlabel nama file untuk 170 citra `Berawan` (dan `Berkabut`) yang **sudah tersampel di run sebelumnya** — bukan re-audit 300+ citra mentah, cukup yang benar-benar terpakai. Anda review, tambahkan temuan lain (kalau ada) ke `dataset_corrections.json`.

**Langkah 3 — File koreksi manual** `dataset/dataset_corrections.json` (root repo), sudah diisi 2 temuan awal sebagai starting point:

```json
{
  "reclassify": {
    "cloudy159.jpg": "Berkabut",
    "cloudy54.jpg": "Berkabut"
  },
  "exclude": []
}
```
`exclude` diisi otomatis dari hasil deduplikasi Langkah 1, digabung manual kalau Anda temukan file lain yang perlu dibuang (bukan direklasifikasi) saat review contact sheet.

**Langkah 4 — Terapkan sebelum sampling.** Saat membangun pool kandidat tiap kelas dari Source 1:
- File di `exclude` → dibuang dari pool asalnya, tidak masuk kandidat manapun
- File di `reclassify` → dipindah dari pool kelas asal ke pool kelas tujuan
- Sampling `random_state=42` tetap 170/kelas seperti sebelumnya, tinggal jalan dari pool yang sudah bersih

Sisa pool tetap cukup besar untuk 170/kelas (Cloudy: 323 dikurangi beberapa exclude/reclassify masih ratusan; Foggy: 259 ditambah reklasifikasi baru).

**Catatan reproduktibilitas:** hasil sampel 170/kelas yang baru **tidak akan identik** dengan yang lama (pool sumbernya berubah) — ini memang tujuannya (membangun ulang dari data yang sudah bersih), bukan mereproduksi sampel lama yang sudah diketahui bermasalah. Reproduktibilitas berlaku ke depan: siapa pun yang menjalankan ulang kode + `dataset_corrections.json` yang sama akan dapat 850 citra yang identik.

### 8.4 Dampak ke Fase Berikutnya

- **Fase 1** harus dijalankan ulang sepenuhnya begitu `dataset/processed/` baru selesai (murah, ~1 menit untuk 4.250 file per estimasi sebelumnya)
- **Fase 2**: arsipkan (jangan hapus) 4 checkpoint + metrik EfficientNetB0 yang sudah ada (mis. pindah ke `models/_pre_correction/`, `results/_pre_correction/`) sebagai catatan historis — bisa berguna untuk membandingkan dampak koreksi data di BAB IV kalau diperlukan. Lalu retrain kelima seed dari awal dengan dataset baru.
- **BAB III/IV**: proses koreksi data ini justru memperkuat narasi metodologi — bisa ditulis sebagai langkah quality control tambahan di luar spesifikasi awal proposal, menunjukkan kehati-hatian terhadap kualitas dataset crowd-sourced.

---

_Versi dokumen: 0.2 — 14 September 2026 (Bagian 8: koreksi kualitas data — mislabeling & duplikasi Source 1)_

### 8.5 Update — Audit Visual & Revisi Keputusan (15 September 2026)

Setelah dataset_corrections.json diterapkan, dilakukan audit visual contact sheet:

**File contact sheet yang dihasil:**
- `results/figures/audit_berawan.png` — 170 citra Berawan tersampel (dari Source 1 Cloudy)
- `results/figures/audit_berkabut.png` — 170 citra Berkabut tersampel (dari Source 1 Fog)

**Temuan tambahan dari audit:**

| Kelas 1 | Kelas 2 | Jenis | Contoh |
|----------|---------|-------|--------|
| Berawan | Mendung | Tone gelap | cloudy195, cloudy181, cloudy55, cloudy50, cloudy82, cloudy282 — awan terlihat gelap seperti mendung |
| Berawan | Berkabut | Tekstur kabut | cloudy159, cloudy54 — sudah dipindahkan ke Berkabut |
| Berkabut | Snow | Klasifikasi batas | mist-013.jpg — terlihat seperti bersalju, namun dalam toleransi |

**Revisi Keputusan (vs Section 8.3 awal):**

Berdasarkan review lebih lanjut, CLAHE untuk Berawan-Mendung di-pending:

| Solusi | Scope | Status |
|--------|-------|--------|
| **Label correction** (`dataset_corrections.json`) | Berawan → Mendung | Prioritas 1 — perluas json dengan 6 file baru |
| **CLAHE ablation** | Berkabut saja | Prioritas 2 — eksperimen terkontrol, bukan preprocessing wajib |
| **Dokumentasi keterbatasan** | Confusion matrix | BAB IV — ambiguitas semantik yang genuinely tidak bisa diputuskan |

Detail keputusan dan argumen ada di `doc/phase-1-preprocessing-split.md` Bagian 8.4-8.6.

---

_Versi dokumen: 0.4 — 15 September 2026 (Bagian 8.5: revisi keputusan CLAHE vs label correction)_
