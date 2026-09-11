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
| Status             | Belum Mulai                                                                    |

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

## 7. Catatan untuk Fase Berikutnya

- **Fase 1** (masih VS Code, venv sama) memakai `dataset/processed/` sebagai input pipeline resize/normalize/augment/split — tambah `scikit-learn` ke `requirements.txt` saat itu
- **Sebelum Fase 2** (training di Kaggle): upload `dataset/processed/` sebagai Kaggle Dataset privat (`kaggle datasets create` atau web UI) — lihat Bagian 1

---

_Versi dokumen: 0.1 — 9 September 2026_
