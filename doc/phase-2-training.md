# Fase 2 — Arsitektur Model & Skrip Training

> Turunan dari `doc/main-documentation.md` Bagian 10. Prasyarat: Fase 1 selesai dan sudah diverifikasi (5 seed split × 850 citra, `src/data_pipeline.py` siap).

## 0. Ringkasan Fase

| | |
|---|---|
| Tujuan | Notebook training 2 arsitektur × 5 seed × 2 tahap (freeze + fine-tune), siap dijalankan di Kaggle GPU T4×2 |
| Ref. Proposal | 3.3.6, 3.3.7, Tabel 3.3 |
| Ref. Dokumen Utama | `doc/main-documentation.md` Bagian 4.4 |
| Prasyarat | Fase 1 selesai; `dataset/split/` tersedia |
| Output | `src/model_builder.py`, `src/train.py`, `notebooks/03a_training_efficientnetb0.ipynb`, `notebooks/03b_training_resnet50.ipynb`, `models/` |
| Status | Sedang Dikerjakan (12 September 2026) |

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

### 5a. Clone dari GitHub LFS

```bash
# Di Kaggle Notebook → Terminal (New → Terminal)
!apt-get install -y git-lfs > /dev/null 2>&1
!git lfs install

# Clone repo
!git clone https://github.com/milalestari/TUGAS-AKHIR.git
%cd TUGAS-AKHIR

# Verifikasi
!ls dataset/split/    # harus ada 5 seed
!ls src/              # harus ada data_pipeline.py
```

**Catatan:** Folder `dataset/split/` ~500MB — GitHub LFS perlu dikonfigurasi untuk file besar. Jika LFS tidak aktif, gunakan Kaggle Dataset upload sebagai fallback.

### 5b. Fallback: Kaggle Dataset

```bash
# Upload folder dataset/split/ sebagai Kaggle Dataset privat
# Lalu di notebook:
!kaggle datasets download -d USERNAME/ta-cuaca-split -p dataset_split --unzip
```

### 5c. Accelerator
```
Notebook Settings → Accelerator → GPU T4×2
```

## 6. Alur Kode di Notebook

```
Sel  1: Judul & informasi
Sel  2: Setup (clone repo / import library)
Sel  3: Verifikasi data_pipeline
Sel  4: Load data dengan build_tf_dataset
Sel  5: Build model (model_builder)
Sel  6: STAGE 1 — Freeze + Train 10 epoch
Sel  7: Load best checkpoint Stage 1
Sel  8: STAGE 2 — Unfreeze + Fine-tune 10 epoch
Sel  9: Visualisasi training history
Sel 10: Evaluasi pada test split
Sel 11: Classification report + Confusion matrix
Sel 12: Simpan hasil ke results/
Sel 13: Download checkpoint & hasil
```

## 7. Output & Definition of Done

- [ ] `src/model_builder.py` berisi fungsi `build_model(arsitektur, seed, preprocess_fn)`
- [ ] `src/train.py` berisi kelas/training fungsi untuk 2-stage training
- [ ] `notebooks/03a_training_efficientnetb0.ipynb` bisa dijalankan di Kaggle
- [ ] `notebooks/03b_training_resnet50.ipynb` bisa dijalankan di Kaggle
- [ ] Training 10 run (2 arsitektur × 5 seed) selesai di Kaggle
- [ ] 10 checkpoint `.keras` tersimpan di `models/`
- [ ] Classification report dan confusion matrix tersimpan di `results/`
- [ ] Setup Kaggle (clone GitHub / Kaggle Dataset) sudah didokumentasikan

## 8. Catatan Teknis

### Unfreeze Layer — Cara Hitung Index

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

## 10. Catatan untuk Fase Berikutnya

- **Fase 3** membutuhkan 10 file `.keras` di `models/` dan hasil metrik di `results/` — pastikan semua ter-download dari Kaggle ke repo lokal sebelum mulai Fase 3.
- Streamlit app (Fase 4) membutuhkan model terbaik dari Fase 3 — bukan semua 10 checkpoint.

---

*Versi dokumen: 0.1 — 12 September 2026*
