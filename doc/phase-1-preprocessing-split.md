# Fase 1 — Pipeline Pra-pemrosesan & Pembagian Data

> Turunan dari `doc/main-documentation.md` Bagian 10. Prasyarat: Fase 0 selesai dan sudah diverifikasi (850 citra, 5 kelas, `dataset/processed/`).

## 0. Ringkasan Fase

| | |
|---|---|
| Tujuan | Fungsi loader (RGB, resize, augmentasi, preprocess per-arsitektur) + duplikat fisik split 5-seed, siap dipanggil training |
| Ref. Proposal | 3.3.4, 3.3.5, Tabel 3.3 |
| Ref. Dokumen Utama | `doc/main-documentation.md` Bagian 4.2–4.3 |
| Prasyarat | Fase 0 selesai |
| Output | `dataset/split/seed{42,123,2024,7,99}/{train,val,test}/{kelas}/`, `src/data_pipeline.py`, `notebooks/02_preprocessing_and_split.ipynb` |
| Status | Selesai (11 September 2026 — bug flat-folder, floating-point precision round()/int(), dan syntax parentheses sudah diperbaiki & didokumentasikan) |

## 1. Keputusan Desain

**Sudah Anda konfirmasi eksplisit:**

| Keputusan | Pilihan | Alasan |
|---|---|---|
| Penyimpanan split | Duplikat fisik: `dataset/split/seed{n}/{train,val,test}/{kelas}/` | Struktur eksplisit, gampang diverifikasi manual/di-zip untuk upload Kaggle |
| Augmentasi | tf.data + Keras preprocessing layers (bukan `ImageDataGenerator`) | Konsisten dengan pemakaian `tf.keras.applications.*.preprocess_input` di proposal 3.3.4 |

**Default saya, belum dibantah — anggap oke kecuali dikoreksi:**

| Keputusan | Pilihan | Alasan |
|---|---|---|
| Lokasi kode augmentasi | `src/data_pipeline.py` (`.map()` pada tf.data pipeline) | Fase 2 punya 2 notebook training terpisah — satu sumber kebenaran mencegah konfigurasi diam-diam beda antar arsitektur |
| RGB conversion | Sekali, saat menyalin `dataset/processed/` → `dataset/split/seed{n}/...` | Sudah menyalin fisik untuk split — sekalian bereskan RGB di titik itu, hindari overhead konversi berulang tiap epoch saat training |
| `preprocess_input` per-arsitektur | Diparameterisasi sebagai argumen fungsi di `data_pipeline.py` | Satu pipeline dipakai 2 arsitektur, bukan 2 pipeline terpisah |

**Belum pernah ditanya di sesi ini — usulan saya, mohon dikonfirmasi:**

| Keputusan | Usulan | Alasan |
|---|---|---|
| 5 nilai seed | `[42, 123, 2024, 7, 99]` | Angka konkret sembarang dengan 42 sebagai anchor (dipakai juga di Fase 0) — folder dinamai langsung dari nilai ini (`seed42/`, dst.) agar transparan. Bebas diganti, tidak ada makna khusus di balik angka-angka ini |

**Efek samping menguntungkan dari "duplikat fisik":** karena `dataset/split/seed{n}/` adalah satu folder yang sama dibaca oleh *kedua* skrip training (EfficientNetB0 dan ResNet50), syarat "pasangan run harus pakai data split identik" untuk validitas paired t-test (Proposal 3.3.8) otomatis terpenuhi by construction — tidak perlu koordinasi manual antar 2 notebook training di Fase 2.

## 2. Update `requirements.txt`

Tambah satu baris ke `requirements.txt` yang sudah ada (Fase 0):

```
scikit-learn
```

(dibutuhkan untuk `train_test_split` dengan `stratify=`). `tensorflow` **belum** ditambah di fase ini — augmentasi dan `preprocess_input` di `data_pipeline.py` ditulis dan diuji sebagai spesifikasi/fungsi, tapi eksekusi penuh (yang benar-benar butuh TensorFlow jalan) baru terjadi di Fase 2 saat training. Jika ingin menjalankan sel validasi tf.data secara lokal di notebook Fase 1 (disarankan, lihat Bagian 4 langkah 6), `tensorflow` perlu diinstall juga di venv lokal — cukup untuk uji fungsi, bukan untuk training sungguhan.

## 3. Estimasi Disk

5 seed × 850 citra = 4.250 citra terduplikasi di `dataset/split/`, ditambah 850 di `dataset/processed/` (master, tidak diubah). Untuk dataset seukuran ini (~170 citra/kelas, kemungkinan besar total di bawah 1 GB), tambahan ini seharusnya tidak masalah — hanya dicatat sebagai FYI kalau ruang disk terbatas.

## 4. Rencana Isi Notebook `notebooks/02_preprocessing_and_split.ipynb`

1. **Konfigurasi** — path `dataset/processed/` → `dataset/split/`, daftar kelas, daftar seed `[42, 123, 2024, 7, 99]`, target rasio 70:15:15
2. **Untuk tiap seed** — `train_test_split(..., stratify=label, random_state=seed)` dua kali berurutan (pisah train vs sisa 70:30, lalu sisa dipisah val:test 50:50) menghasilkan 3 daftar file per kelas
3. **Salin + convert RGB** — untuk tiap seed, tiap subset (train/val/test), tiap kelas: baca file dari `dataset/processed/{kelas}/`, `Image.convert("RGB")`, simpan ke `dataset/split/seed{n}/{subset}/{kelas}/` (nama file tetap sama seperti di `processed/`, tidak perlu rename ulang — sudah unik dari Fase 0)
4. **Sanity check per seed** — hitung ulang jumlah per subset
   - **Per kelas:** train ≈119, val ≈25, test ≈26
   - **Total per seed:** train=595, val=125, test=130
   - **Catatan pembulatan:** `int(170 × 0.15) = int(25.5) = 25` — Python membulatkan sisa menjadi 25+26, bukan 25+25. Ini **normal dan valid** selama val+test=255=850−595 dan proporsi kelas per subset konsisten
   - Pastikan distribusi kelas proporsional di tiap subset
   - Pastikan train/val/test **disjoint** (tidak ada file dobel dalam seed sama)
5. **`src/data_pipeline.py`** — tulis fungsi `build_tf_dataset(seed, subset, preprocess_fn, augment=False, batch_size=32)`:
   - baca citra dari `dataset/split/seed{n}/{subset}/`
   - resize ke 224×224
   - kalau `augment=True` (hanya untuk `subset="train"`): terapkan `RandomRotation`, `RandomFlip("horizontal")`, `RandomTranslation`, `RandomZoom` (parameter sesuai proposal: rotasi ±20°, shift ±15%, zoom ±15%) via `.map()`
   - terapkan `preprocess_fn` (mis. `tf.keras.applications.efficientnet.preprocess_input` atau `.resnet50.preprocess_input`) via `.map()`
   - `batch()` dan `prefetch()`
6. **Validasi cepat di notebook** — panggil `build_tf_dataset` untuk satu seed contoh (`train`, `augment=True`) dan (`val`, `augment=False`), tampilkan grid sampel: batch train (augmentasi terlihat — tiap panggilan beda) vs batch val (tidak berubah), simpan ke `results/figures/`

## 5. Output & Definition of Done

- [x] `dataset/split/seed{42,123,2024,7,99}/{train,val,test}/{kelas}/` — 5 seed lengkap, tiap seed total 850 citra (train=595/val=125/test=130), semua RGB valid
- [x] Distribusi kelas proporsional (stratified) di tiap subset, tiap seed (train≈119/kelas, val≈25/kelas, test≈26/kelas)
- [x] Train/val/test disjoint dalam seed yang sama (tidak ada file dobel)
- [x] `src/data_pipeline.py` berisi `build_tf_dataset(seed, subset, preprocess_fn, augment)` yang bisa dipanggil beda `preprocess_fn` untuk 2 arsitektur
- [x] Notebook validasi menunjukkan augmentasi aktif di train, tidak aktif di val/test
- [x] `requirements.txt` sudah menambahkan `scikit-learn`
- [x] Update status Fase 1 di `doc/main-documentation.md` Bagian 10 → Selesai

## 6. Catatan Teknis Penting (Bug Fixes)

### Bug: Flat-folder vs Struktur Keras-Compatible
Pada implementasi awal, salin file ke `dataset/split/seed{n}/{subset}/` (flat, tanpa subfolder kelas) — menyebabkan `image_dataset_from_directory` tidak bisa membaca label. **Sudah diperbaiki:** file disalin ke `dataset/split/seed{n}/{subset}/{kelas}/` — setiap subset punya subfolder per kelas, kompatibel dengan `tf.keras.preprocessing.image_dataset_from_directory`.

### Bug: Floating-Point Precision — round() vs int()
**Masalah:** `170 × 0.70 = 118.9999...` (bukan 119.0) karena representasi biner floating-point.
- `int(118.9999)` → 118 (salah, harus 119)
- `round(118.9999)` → 119 (benar)
- `train_test_split` sklearn menggunakan `round()` untuk pembagian train, `int()` untuk val

**Formula yang benar (Opsi 2 — konsisten dengan train_test_split):**
```python
expected_train_per_class = round(TARGET_PER_CLASS * TRAIN_RATIO)   # 119
expected_val_per_class   = int(TARGET_PER_CLASS * VAL_RATIO)       # 25
expected_test_per_class  = TARGET_PER_CLASS - train - val          # 26
expected_train = 5 * 119   # 595
expected_val   = 5 * 25    # 125
expected_test  = 850 - 595 - 125  # 130
```
Menggunakan `round()` untuk train dan `int()` untuk val — TIDAK symmetric rounding.

### Bug: Syntax Error — Missing Closing Parentheses
Formula `n = (len(list(glob(...))))` membutuhkan 4 penutup `)` di akhir (1 untuk `n=`, 3 untuk `len/list/glob`). **Sudah diperbaiki** dengan helper function `count_files()` di cell 8 untuk menghindari parenthesis hell.

## 7. Catatan untuk Fase Berikutnya

- **Sebelum Fase 2:** seluruh `dataset/split/` (semua 5 seed sekaligus) diupload **satu kali** sebagai satu Kaggle Dataset privat — supaya kedua notebook training (EfficientNetB0, ResNet50) mengakses seed yang sama dari satu sumber, tanpa upload ulang per seed
- **Fase 2** memanggil `build_tf_dataset` dari `src/data_pipeline.py` dengan `preprocess_fn` berbeda per arsitektur — modul ini perlu ikut disalin/diimpor ke lingkungan Kaggle Notebook (via upload sebagai dataset tambahan, atau di-paste langsung ke sel notebook)

---
*Versi dokumen: 0.3 — 11 September 2026 (bug floating-point round/int + syntax parentheses)*
