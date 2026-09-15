# Fase 2 — Arsitektur Model & Skrip Training

> Turunan dari `doc/main-documentation.md` Bagian 10. Prasyarat: Fase 1 selesai dan sudah diverifikasi (5 seed split × 850 citra, `src/data_pipeline.py` siap).

## 0. Ringkasan Fase

| | |
|---|---|
| Tujuan | Notebook training 2 arsitektur × 5 seed × 2 tahap (freeze + fine-tune), siap dijalankan di Kaggle GPU T4×2 |
| Ref. Proposal | 3.3.6, 3.3.7, Tabel 3.3 |
| Ref. Dokumen Utama | `doc/main-documentation.md` Bagian 4.4 |
| Prasyarat | Fase 1 selesai; `dataset/split/` di-upload sebagai Kaggle Dataset privat `dataset-cuaca-split` di `/kaggle/input/` |
| Output | `src/model_builder.py`, `src/train.py`, `notebooks/03a_training_efficientnetb0.ipynb`, `notebooks/03b_training_resnet50.ipynb`, `models/` |
| Status | Training EfficientNetB0 selesai; ResNet50 belum mulai. Bug timing (0.0) ditemukan & diperbaiki di `src/train.py` |

## 1. Keputusan Desain

**Sudah terkunci dari proposal 3.3.6:**

| Keputusan | Nilai | Ref. Proposal |
|---|---|---|
| Epoch Stage 1 (feature extraction) | 10 | 3.3.6 |
| Epoch Stage 2 (fine-tuning) | 10 | 3.3.6 |
| Optimizer | Adam | 3.3.6 |
| Loss | Categorical Crossentropy | 3.3.6 |
| Batch size | 32 | 3.3.6 |
| LR Stage 1 (freeze) | 1×10⁻³ | 3.3.6 |
| LR Stage 2 (fine-tune) | 1×10⁻⁴ | 3.3.6 |
| EfficientNetB0: layer di-unfreeze | 20 layer terakhir (dari 237) | 3.3.6 |
| ResNet50: layer di-unfreeze | 15 layer terakhir (dari 175) | 3.3.6 |
| EarlyStopping | patience=5, monitor=val_loss | 3.3.6 |
| ReduceLROnPlateau | factor=0.5, patience=5, monitor=val_loss | 3.3.6 |
| ModelCheckpoint | monitor=val_accuracy, save_best_only=True | 3.3.6 |
| Metrik evaluasi | accuracy, precision, recall, F1-score (macro avg) | 3.3.7 |
| Confusion matrix | 5×5, dinormalisasi | 3.3.7 |
| Random seed per run | [42, 123, 2024, 7, 99] | 3.3.6, 3.3.5 |
| Platform training | Kaggle Notebook GPU T4×2 | main-documentation.md Bagian 6 |

## 2. Arsitektur Classification Head (Identik Kedua Model)

Sesuai proposal 3.3.6 — kedua model pakai head yang **identik**:

```
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dropout(0.3)
        ↓
Dense(512, activation='relu')
        ↓
BatchNormalization
        ↓
Dropout(0.3)
        ↓
Dense(5, activation='softmax')
```

## 3. Arsitektur Base Model

### EfficientNetB0
- Input: 224×224×3
- Base: `tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False)`
- Layer di-unfreeze Stage 2: **20 layer terakhir** (dari 237 total)
- Dihitung dari output: ambil semua layer dari index `-20` ke akhir

### ResNet50
- Input: 224×224×3
- Base: `tf.keras.applications.ResNet50(weights='imagenet', include_top=False)`
- Layer di-unfreeze Stage 2: **15 layer terakhir** (dari 175 total)
- Dihitung dari output: ambil semua layer dari index `-15` ke akhir

## 4. Dua Tahap Training

### Stage 1 — Feature Extraction (Freeze)
```
for epoch in range(1, 11):   # 10 epoch
    - Semua base model layer:.trainable = False
    - Classifier head: .trainable = True
    - Compile dengan LR = 1×10⁻³
    - Train pada train split
    - Validate pada val split
    - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    - EarlyStopping: monitor val_loss, patience=5
```

### Stage 2 — Fine-Tuning (Unfreeze)
```
for epoch in range(1, 11):   # 10 epoch
    - Top N layer base model: .trainable = True
    - Classifier head: .trainable = True
    - Compile dengan LR = 1×10⁻⁴ (10× lebih kecil dari Stage 1)
    - Train pada train split
    - Validate pada val split
    - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    - ModelCheckpoint menyimpan model terbaik Stage 2
```

**Checkpointing:**
- Stage 1 checkpoint: `models/{arsitektur}_seed{n}_stage1.keras`
- Stage 2 checkpoint: `models/{arsitektur}_seed{n}.keras` (final, overwrite Stage 1)
- `save_best_only=True` → hanya menyimpan yang terbaik

## 5. Setup Kaggle Notebook

### 5a. Struktur Direktori di Kaggle

```
/kaggle/
├── input/
│   └── dataset-cuaca-split/     ← Kaggle Dataset privat (data split 5-seed)
│       └── dataset/split/
│           ├── seed42/{train,val,test}/{Cerah,Mendung,Hujan,Berawan,Berkabut}/
│           ├── seed123/...
│           ├── seed2024/...
│           ├── seed7/...
│           └── seed99/...
│
└── working/
    └── TUGAS-AKHIR/              ← Git clone repo GitHub
        ├── notebooks/
        │   ├── 03a_training_efficientnetb0.ipynb
        │   └── 03b_training_resnet50.ipynb
        ├── src/
        │   ├── data_pipeline.py
        │   ├── model_builder.py
        │   └── train.py
        ├── models/                  ← output training (.keras)
        └── results/                 ← output evaluasi (JSON, PNG)
```

### 5b. Clone Repo ke Kaggle Working

```bash
# Di Kaggle Terminal (New → Terminal)
%cd /kaggle/working
!git clone -b preprocessing_img https://github.com/milalestari/TUGAS-AKHIR.git

# Verifikasi
!ls TUGAS-AKHIR/src/       # harus ada model_builder.py, train.py, data_pipeline.py
```

### 5c. Upload Dataset ke Kaggle

1. Buka https://www.kaggle.com → Create New Dataset
2. Upload folder `dataset/split/` (ZIP atau langsung folder)
3. Nama: `dataset-cuaca-split` (privat)
4. Di notebook: notebook akan otomatis melihat `/kaggle/input/dataset-cuaca-split/`

### 5d. Bootstrap Cell (Auto-detect Path)

Setiap notebook sudah memiliki bootstrap cell di **cell pertama yang bisa dieksekusi** (cell auto-detect). Cell ini:

```python
def find_project_root():
    current = Path.cwd()
    for path in [current, *current.parents]:
        if (path / "requirements.txt").exists() and (path / "src").exists():
            return path
    raise RuntimeError("PROJECT_ROOT TUGAS-AKHIR tidak ditemukan.")

PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_DIR  = Path("/kaggle/input/dataset-cuaca-split")
if not INPUT_DIR.exists():
    INPUT_DIR = PROJECT_ROOT / "dataset" / "split"   # fallback lokal

DATA_DIR    = INPUT_DIR
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
```

Notebook auto-detect `PROJECT_ROOT` — tidak perlu hard-code `/kaggle/working/TUGAS-AKHIR`.

### 5e. Accelerator

```
Notebook Settings → Accelerator → GPU T4×2
```

## 6. Alur Kode di Notebook

```
Sel  1: Judul & informasi (markdown)
Sel  2: Bootstrap auto-detect (auto-detect PROJECT_ROOT + INPUT_DIR)
Sel  3: Import library (TensorFlow, scikit-learn, dll.)
Sel  4: Verifikasi data_pipeline & paths
Sel  5: Load datasets dengan build_tf_dataset (seed=42 demo)
Sel  6: Build model EfficientNetB0/ResNet50 (Stage 1: freeze)
Sel  7: STAGE 1 — Feature Extraction (freeze, 10 epoch)
Sel  8: Load best checkpoint Stage 1
Sel  9: STAGE 2 — Fine-Tuning (unfreeze, 10 epoch)
Sel 10: Visualisasi training history (plot TERPISAH Stage 1 & Stage 2)
Sel 11: Evaluasi pada test set
Sel 12: Classification report + Confusion matrix
Sel 13: Simpan history & metrik ke JSON
Sel 14: Full pipeline — Semua 5 seed (dengan plot per seed)
Sel 15: Summary semua metrik (5 seed)
```

## 6.1 Spesifikasi Plot Training History

Plot training history disimpan di `results/figures/phase_2/`:

| File | Isi |
|------|-----|
| `history_{arch}_seed{n}_stage1.png` | Stage 1: Loss + Accuracy, 10 epoch |
| `history_{arch}_seed{n}_stage2.png` | Stage 2: Loss + Accuracy, 10 epoch |

**Spesifikasi plot:**
- Layout: `subplots(1, 2, figsize=(12, 4))` — Loss di kiri, Accuracy di kanan
- Judul: `"{Arsitektur} Seed {n} — Stage 1 (Feature Extraction)\nLR={LR}"`
- Warna: Train = hijau (`#2ecc71`), Val = merah (`#e74c3c`)
- Style: `sns.set_style('whitegrid')`, marker `o-` (train) dan `s--` (val)
- Simpan: `dpi=150, bbox_inches='tight'`

**Contoh naming:**
```
results/figures/phase_2/
├── history_effnet_seed42_stage1.png
├── history_effnet_seed42_stage2.png
├── history_effnet_seed123_stage1.png
├── history_effnet_seed123_stage2.png
└── ...
```

## 7. Output & Definition of Done

- [x] `src/model_builder.py` berisi fungsi `build_model(arsitektur, seed, preprocess_fn)`
- [x] `src/train.py` berisi fungsi 2-stage training dengan **timing capture** (`t_stage1`/`t_stage2`)
- [x] `notebooks/03a_training_efficientnetb0.ipynb` sudah dijalankan di Kaggle (5 seed selesai: 42, 123, 2024, 7, 99)
- [ ] `notebooks/03b_training_resnet50.ipynb` — **belum dijalankan**
- [x] 5 checkpoint EfficientNetB0 `.keras` tersimpan di `models/`
- [ ] 5 checkpoint ResNet50 `.keras` tersimpan di `models/` — pending
- [x] Classification report dan confusion matrix EfficientNetB0 tersimpan di `results/`
- [ ] Classification report dan confusion matrix ResNet50 — pending
- [ ] 10 checkpoint `.keras` total
- [x] **Selesai:** Data training time (`time_stage1_s`/`time_stage2_s`) tersedia di semua JSON (sudah diperbaiki dan diverifikasi)

## 8. Catatan Teknis

### 8.1 Bug Fix: Timing Data Hilang di JSON (`time_stage1_s` / `time_stage2_s` = 0.0)

**Masalah:**
Semua JSON metrik EfficientNetB0 menunjukkan `time_stage1_s: 0.0` dan `time_stage2_s: 0.0`:

```json
{
  "architecture": "efficientnetb0",
  "seed": 42,
  "time_stage1_s": 0.0,
  "time_stage2_s": 0.0
}
```

**Penyebab:**
- Notebook Cell 11 (FULL PIPELINE) membaca `result.get('t_stage1', 0)` dan `result.get('t_stage2', 0)`
- Fungsi `train_two_stage()` di `src/train.py` **tidak pernah membuat** key `t_stage1`/`t_stage2` di return dict
- Return dict hanya berisi: `stage1`, `stage2`, `model`, `stage1_checkpoint`, `final_checkpoint`

**Perbaikan di `src/train.py`:**

```python
def train_two_stage(...) -> dict:
    # ... (sebelum model.fit Stage 1)
    import time as time_module

    t_s1_start = time_module.time()
    history_stage1 = model.fit(ds_train, validation_data=ds_val,
                                epochs=EPOCHS_STAGE1, callbacks=callbacks_s1,
                                verbose=verbose)
    t_s1 = time_module.time() - t_s1_start

    # ... (setelah Stage 1, sebelum model.fit Stage 2)
    t_s2_start = time_module.time()
    history_stage2 = model.fit(ds_train, validation_data=ds_val,
                                epochs=EPOCHS_STAGE2, callbacks=callbacks_s2,
                                verbose=verbose)
    t_s2 = time_module.time() - t_s2_start

    return {
        "stage1": history_stage1,
        "stage2": history_stage2,
        "model": model,
        "stage1_checkpoint": stage1_checkpoint,
        "final_checkpoint": checkpoint_path,
        "t_stage1": t_s1,   # ← ditambahkan
        "t_stage2": t_s2,   # ← ditambahkan
    }
```

**Catatan:** Jika Cell 11 di notebook sudah dieksekusi dengan nilai 0.0, data training time **sudah hilang dan tidak bisa direkonstruksi**. Perbaikan ini mencegah hal yang sama di notebook ResNet50 dan saat re-run.

### 8.2 Catatan Display Learning Rate: "1.0000e-04"

Nilai LR Stage 2 di training history plot menampilkan sebagai `1.0000e-04`. Ini adalah **format default TensorFlow** saat mencetak `float` ke string (`f"{lr:.4f}"` atau `np.float64` di dict). **Bukan error.** Nilai aktualnya tetap `0.0001` (1×10⁻⁴) yang benar sesuai Proposal 3.3.6.

Untuk tampilan yang lebih bersih, bisa diubah saat plotting:
```python
# Alih-alih langsung pakai nilai dari history, format ulang:
lr_label = f"{1e-4:.0e}"   # → "1e-04"
```

### 8.3 Unfreeze Layer — Cara Hitung Index

```python
# EfficientNetB0: 237 layer total → unfreeze 20 layer terakhir
base_effnet = EfficientNetB0(weights='imagenet', include_top=False)
n_total_layers = len(base_effnet.layers)
n_unfreeze = 20
# Set trainable dari layer index (n_total_layers - n_unfreeze) ke akhir
for layer in base_effnet.layers[n_total_layers - n_unfreeze:]:
    layer.trainable = True

# ResNet50: 175 layer total → unfreeze 15 layer terakhir
base_resnet = ResNet50(weights='imagenet', include_top=False)
n_total_layers = len(base_resnet.layers)
n_unfreeze = 15
for layer in base_resnet.layers[n_total_layers - n_unfreeze:]:
    layer.trainable = True
```

### Dataset Shape & Class Names

- Input shape: `(224, 224, 3)`
- Output: 5 kelas
- Class names (urutan): `["Berkabut", "Berawan", "Cerah", "Hujan", "Mendung"]`
- Berurutan sesuai alphabetical karena `image_dataset_from_directory` pakai `sorted(os.listdir())`

### Mengapa Stage 2 Tidak Mulai dari Best Stage 1?

Proposal tidak meminta load checkpoint Stage 1 sebelum Stage 2. Stage 2 dimulai dari kondisi akhir Stage 1 (bukan dari checkpoint terbaik). Ini berarti bobot Stage 1 langsung dipakai sebagai starting point Stage 2. Alasan: `ModelCheckpoint(save_best_only=True)` di Stage 1 memastikan model terbaik disimpan, dan Stage 2 langsung melanjutkan dari situ.

## 9. Estimasi Waktu Training (Kaggle T4×2)

| Komponen | Estimasi |
|---|---|
| 1 run (1 arsitektur, 1 seed, 2 stage) | ~10-15 menit |
| EfficientNetB0 × 5 seed | ~50-75 menit |
| ResNet50 × 5 seed | ~50-75 menit |
| **Total 10 run** | **~1.5-2.5 jam** |

Dengan T4×2 paralel (EfficientNetB0 di GPU:0, ResNet50 di GPU:1): waktu bisa ditekan ~50%.

## 11. Hasil Training EfficientNetB0 (Fase 2 Selesai)

> Hasil eksekusi di Kaggle GPU T4×2 (13 September 2026). Semua 5 seed selesai.

### 11.1 Ringkasan Hasil per Seed

| Seed | Test Accuracy | Test F1 | Stage 1 Time | Stage 2 Time | Total Time |
|------|---------------|---------|-------------|-------------|------------|
| 7    | **90.00%** | 0.9021 | 150.2s | 145.6s | **4m 56s** |
| 99   | **90.00%** | 0.9004 | 150.0s | 158.7s | **5m 09s** |
| 42   | 88.46% | 0.8875 | 149.2s | 157.0s | 5m 06s |
| 2024 | 88.46% | 0.8843 | 150.4s | 133.9s | 4m 44s |
| 123  | 86.92% | 0.8693 | 151.4s | 156.7s | 5m 08s |
| **Mean** | **88.77%** | **0.8887** | **150.2s** | **150.4s** | **~5m 00s** |
| **Std** | **±1.27%** | **±1.28%** | ±0.7s | ±9.5s | — |

**Total waktu training EfficientNetB0 × 5 seed:** ~25 menit

### 11.2 Output File

```
models/efficientnetb0/
├── efficientnetb0_seed42.keras
├── efficientnetb0_seed123.keras
├── efficientnetb0_seed2024.keras
├── efficientnetb0_seed7.keras
└── efficientnetb0_seed99.keras

results/
├── json/
│   ├── metrics_effnet_seed42.json
│   ├── metrics_effnet_seed123.json
│   ├── metrics_effnet_seed2024.json
│   ├── metrics_effnet_seed7.json
│   └── metrics_effnet_seed99.json
├── csv/
│   └── metrics_effnet_summary.csv
└── figures/phase_2/
    ├── history_effnet_seed42_stage1.png
    ├── history_effnet_seed42_stage2.png
    ├── history_effnet_seed123_stage1.png
    ├── history_effnet_seed123_stage2.png
    ├── history_effnet_seed2024_stage1.png
    ├── history_effnet_seed2024_stage2.png
    ├── history_effnet_seed7_stage1.png
    ├── history_effnet_seed7_stage2.png
    ├── history_effnet_seed99_stage1.png
    └── history_effnet_seed99_stage2.png
```

### 11.3 Catatan Concern

| # | Concern | Prioritas | Status |
|---|---------|-----------|--------|
| 1 | `time_stage1_s`/`time_stage2_s` = 0.0 | 🔴 ~~Critical~~ | ✅ **Selesai** — semua JSON memiliki timing yang valid |
| 2 | EarlyStopping menyebabkan epoch tidak lengkap | 🟡 Medium | ✅ **Selesai** — plot menggunakan dynamic epoch count |
| 3 | Variasi antar seed cukup besar | 🟢 Low | Normal untuk dataset kecil |
| 4 | Hujan/Mendung confusion | 🟢 Low | Bisa diteliti lebih lanjut di Fase 3 |

### 11.4 Checklist untuk ResNet50 (Sebelum Eksekusi)

1. ✅ **Timing bug:** `src/train.py` sudah diperbaiki dengan `t_stage1`/`t_stage2` di return dict
2. ✅ **Plot terpisah:** Cell 23 notebook menggunakan dynamic epoch count untuk plot
3. ✅ **Checkpoint naming:** `resnet50_seed{n}.keras` — konvensi konsisten
4. **Persiapan:** Clone repo branch `training_resnet50` di Kaggle, sync dataset

### 11.5 Analisis Timing

| Metric | Stage 1 | Stage 2 |
|--------|---------|---------|
| Mean time | 150.2s (~2.5 min) | 150.4s (~2.5 min) |
| Std | ±0.7s | ±9.5s |
| Total per seed | ~5 menit | |

**Observasi:**
- Stage 1 dan Stage 2 memiliki waktu yang comparable
- Stage 2 lebih variatif karena EarlyStopping bisa menghentikan lebih awal
- Total per seed ~5 menit konsisten dengan estimasi (10-15 menit/run)

### 11.6 Variasi Antar Seed

| Seed | Accuracy | Deviasi dari Mean |
|------|----------|-------------------|
| 7    | 90.00% | +1.23% |
| 99   | 90.00% | +1.23% |
| 42   | 88.46% | -0.31% |
| 2024 | 88.46% | -0.31% |
| 123  | 86.92% | -1.85% |

Std dev ±1.27% menunjukkan variasi yang **relatif rendah** dibanding estimasi awal (±2.6%). Seed 123 konsisten paling rendah.

---

*Versi dokumen: 0.4 — 13 September 2026 (hasil lengkap EfficientNetB0 × 5 seed, semua concern terselesaikan)*

## 12. Update — Revisi Preprocessing (15 September 2026)

Berdasarkan audit kualitas data dan review lebih lanjut:

### 12.1 Status Koreksi Data

| Item | Status |
|------|--------|
| Deduplikasi MD5 (101 file) | Selesai — `dataset_corrections.json` |
| Reclassify cloudy159, cloudy54 → Berkabut | Selesai |
| Reclassify 6 file Berawan → Mendung | Pending — perlu konfirmasi |
| CLAHE preprocessing | Pending — ablation study untuk Berkabut saja |

### 12.2 Keputusan Final

**Prioritas 1: Label Correction**
- Perluas `dataset_corrections.json` dengan 6 file Berawan → Mendung
- Ini murni mengubah label, tidak menyentuh pipeline training
- Perubahan kecil, risikonya jelas

**Prioritas 2: CLAHE Ablation (Bukan Preprocessing Wajib)**
- CLAHE untuk Berkabut masuk akal secara optik (fog = kontras-lokal-rendah)
- CLAHE untuk Berawan-Mendung di-pending karena ambiguitas semantik
- Jika dicoba: scope HANYA ke kelas Berkabut, bandingkan dengan baseline
- Risiko distribution shift dari pretrained ImageNet perlu dicek

**Prioritas 3: Dokumentasi BAB IV**
- Sisa confusion matrix yang genuinely ambiguous (bukan wrong label) didokumentasikan sebagai keterbatasan metodologi

Detail keputusan dan argumen ada di `doc/phase-1-preprocessing-split.md` Bagian 8.4-8.6.

---

*Versi dokumen: 0.5 — 15 September 2026 (sinkronisasi keputusan CLAHE vs label correction dari phase-1)*
