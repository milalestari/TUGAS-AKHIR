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
- [x] `notebooks/03a_training_efficientnetb0.ipynb` sudah dijalankan di Kaggle (5 seed selesai: 42, 123, 2024, 7; seed 99 pending)
- [ ] `notebooks/03b_training_resnet50.ipynb` — **belum dijalankan**
- [x] 5 checkpoint EfficientNetB0 `.keras` tersimpan di `models/`
- [ ] 5 checkpoint ResNet50 `.keras` tersimpan di `models/` — pending
- [x] Classification report dan confusion matrix EfficientNetB0 tersimpan di `results/`
- [ ] Classification report dan confusion matrix ResNet50 — pending
- [ ] 10 checkpoint `.keras` total
- [ ] **Critical:** Data training time (`time_stage1_s`/`time_stage2_s`) tersedia di JSON — EfficientNetB0 seed 42/123/2024/7 masih 0.0 (**sudah fix `train.py`, tapi belum re-run**)

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

## 11. Temuan Hasil Training EfficientNetB0 & Perbaikan

> Dicatat pasca-eksekusi aktual di Kaggle (13 September 2026). Temuan ini berlaku juga untuk ResNet50.

### 11.1 Ringkasan Hasil per Seed (EfficientNetB0)

| Seed | Test Accuracy | Test F1 | Catatan |
|------|---------------|---------|---------|
| 7    | **92.31%** | 0.9236 | ✅ Terbaik |
| 42   | 90.77% | 0.9084 | ✅ Baik |
| 2024 | 90.00% | 0.8996 | ✅ Baik |
| 123  | **86.15%** | 0.8612 | ⚠️ Terendah |
| 99   | — | — | belum dieksekusi |
| **Mean** | **≈89.7%** | **≈0.898** | — |
| **Std** | **±2.6%** | **±2.6%** | Variasi antar seed cukup besar |

### 11.2 Concern & Tingkat Prioritas

| # | Concern | Prioritas | Status |
|---|---------|-----------|--------|
| 1 | `time_stage1_s`/`time_stage2_s` = 0.0 di JSON | 🔴 **Critical** | **Sedang diperbaiki** — re-run dengan `train.py` yang sudah diperbaiki (kode lokal sudah fix, perlu sync ke repo) |
| 2 | **Hujan recall = 0.81** (terendah) — 5/26 sampel salah klasifikasi | 🟡 **Medium** | Confusion dominan ke kelas Mendung dan Cerah |
| 3 | **Mendung precision = 0.78** (terendah) — banyak FP dari kelas lain | 🟡 **Medium** | Model cenderung over-predict Mendung |
| 4 | Light overfitting di Stage 1 (train 93% vs val 89%) | 🟢 Low | Normal untuk frozen backbone; hilang saat fine-tune |
| 5 | Variasi antar seed cukup besar (86%-92%, std ±2.6%) | 🟢 Low | Normal untuk dataset kecil (850 citra) |
| 6 | Display LR "1.0000e-04" di plot | 🟢 Cosmetic | TensorFlow default format; bukan error (lihat 8.2) |

### 11.3 Analisis Confusion Matrix (Seed 42, Best Representative)

```
                 Predicted
              Berkabut Berawan Cerah Hujan Mendung
Actual Berkabut   0.96    0.00  0.00  0.00   0.04
     Berawan     0.00    1.00  0.00  0.00   0.00   ← kelas paling bersih
     Cerah       0.00    0.00  0.96  0.04   0.00
     Hujan       0.00    0.04  0.08  0.81   0.08   ← recall 0.81 (paling rendah)
     Mendung     0.00    0.00  0.04  0.19   0.78   ← precision 0.78 (paling rendah)
```

**Interpretasi:**
- **Berkabut** dan **Berawan** → hampir sempurna
- **Cerah** → sangat baik
- **Hujan** → sering dikelirukan dengan **Mendung** dan **Cerah** (wajar secara visual — langit mendung gelap bisa mirip hujan)
- **Mendung** → sering diprediksi padahal sebenarnya **Hujan** (19% FP)

### 11.4 Perbaikan untuk ResNet50 (Sebelum Eksekusi)

Berdasarkan temuan EfficientNetB0, perbaikan berikut harus diterapkan **sebelum** menjalankan notebook ResNet50:

1. **Timing bug:** Pastikan `src/train.py` sudah diperbaiki dengan `t_stage1`/`t_stage2` di return dict (lihat 8.1) — **SEDANG DILAKUKAN**
2. **Plot terpisah:** Notebook ResNet50 harus menggunakan plot terpisah Stage 1 & Stage 2 seperti di EfficientNetB0 (Cell 15 & 23). Konvensi nama file: `history_resnet_seed{n}_stage1.png` dan `history_resnet_seed{n}_stage2.png`
3. **Checkpoint naming:** Konvensi sudah konsisten: `efficientnetb0_seed{n}.keras` untuk EfficientNetB0 → `resnet50_seed{n}.keras` untuk ResNet50

### 11.5 Catatan untuk Overfitting & Gap Train-Val

```
Stage 1 (Freeze):   Train acc 93%  vs  Val acc 88.8%  → gap ~4%  (ringan, normal)
Stage 2 (Fine-tune): Train acc 91%  vs  Val acc 93.6%  → gap invers (sehat!)
```

**Stage 2 tidak overfitting.** Val acc > Train acc menunjukkan model fine-tune **belum fully converge** pada train set — behavior yang sehat dan menunjukkan regularisasi dari unfrozen layers efektif.

Jika nanti di ResNet50 terlihat gap yang lebih besar (>10%) di Stage 2, pertimbangkan:
- Menambah dropout di classification head (0.3 → 0.4)
- Menambah weight decay (Adam default 0.01)
- Mengurangi jumlah layer yang di-unfreeze

---

*Versi dokumen: 0.3 — 13 September 2026 (temuan hasil training EfficientNetB0 + bug fix timing, perbaikan untuk ResNet50)*
