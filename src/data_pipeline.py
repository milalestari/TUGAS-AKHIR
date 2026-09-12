"""
data_pipeline.py — Fase 1
Fungsi loader tf.data untuk EfficientNetB0 dan ResNet50.

Dipanggil oleh kedua notebook training (Fase 2).
Satukan preprocess_input per-arsitektur di satu tempat
agar tidak ada konfigurasi beda antar model.

Usage:
    from src.data_pipeline import build_tf_dataset
    ds_train = build_tf_dataset(
        seed=42,
        subset="train",
        preprocess_fn=tf.keras.applications.efficientnet.preprocess_input,
        augment=True,
        batch_size=32,
    )
    ds_val = build_tf_dataset(
        seed=42,
        subset="val",
        preprocess_fn=tf.keras.applications.efficientnet.preprocess_input,
        augment=False,
        batch_size=32,
    )
"""

import os
from pathlib import Path

import tensorflow as tf

# ──────────────────────────────────────────────
# Konfigurasi (hardcoded — konsisten di semua tempat)
# ──────────────────────────────────────────────

IMG_SIZE = (224, 224)          # native input size EfficientNetB0 & ResNet50
BATCH_SIZE_DEFAULT = 32

# 5 seed cross-validation — sesuai proposal 3.3.5
SEEDS = [42, 123, 2024, 7, 99]

# Rasio split — sesuai proposal Tabel 3.3
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
# Pastikan proporsi: 70 + 15 + 15 = 100

# Augmentasi — sesuai proposal Tabel 3.3
ROTATION_DEG   = 20    # ±20°
SHIFT_FRACTION = 0.15  # ±15%
ZOOM_FRACTION  = 0.15  # ±15%

# Nama 5 kelas Bahasa Indonesia
CLASS_NAMES = ["Berkabut", "Berawan", "Cerah", "Hujan", "Mendung"]

# ──────────────────────────────────────────────
# Augmentasi layers
# ──────────────────────────────────────────────

def _get_augment_layers() -> tf.keras.Sequential:
    """
    Augmentasi untuk subset train.
    Dipanggil inside .map() — harus mengembalikan tensor.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(
            factor=ROTATION_DEG / 360.0,   # Keras pakai fraksi dari 1 penuh
            fill_mode="reflect",
            seed=None,                      # stochastic tiap pemanggilan
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=(-SHIFT_FRACTION, SHIFT_FRACTION),
            width_factor =(-SHIFT_FRACTION, SHIFT_FRACTION),
            fill_mode="reflect",
            seed=None,
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-ZOOM_FRACTION, ZOOM_FRACTION),
            width_factor =(-ZOOM_FRACTION, ZOOM_FRACTION),
            seed=None,
        ),
        tf.keras.layers.RandomFlip(
            mode="horizontal",
            seed=None,
        ),
    ])


# ──────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────

def build_tf_dataset(
    seed: int,
    subset: str,
    preprocess_fn,
    augment: bool = False,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> tf.data.Dataset:
    """
    Bangun tf.data.Dataset dari folder split.

    Parameters
    ----------
    seed         : int  — seed ke-(42, 123, 2024, 7, 99)
    subset       : str  — "train" | "val" | "test"
    preprocess_fn: callable — mis.
                   tf.keras.applications.efficientnet.preprocess_input
                   tf.keras.applications.resnet50.preprocess_input
    augment      : bool — terapkan augmentasi (hanya untuk train)
    batch_size   : int

    Returns
    -------
    tf.data.Dataset  — (batch_images, batch_labels)
    """
    if subset not in ("train", "val", "test"):
        raise ValueError(f"subset harus 'train'|'val'|'test', dapat: {subset!r}")

    split_root = Path(__file__).parent.parent / "dataset" / "split" / f"seed{seed}" / subset

    # Keras.preprocessing.image_dataset_from_directory
    # secara otomatis membaca subfolder sebagai kelas
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=str(split_root),
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=(subset == "train"),
        seed=seed,          # shuffle deterministic per seed
    )

    # Augmentasi — terapkan hanya untuk train
    if augment and subset == "train":
        aug_layers = _get_augment_layers()
        ds = ds.map(
            lambda x, y: (aug_layers(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Preprocessing per-arsitektur (EfficientNetB0 / ResNet50)
    ds = ds.map(
        lambda x, y: (preprocess_fn(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Performance optimization
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ──────────────────────────────────────────────
# Helper: info split (untuk logging / sanity check)
# ──────────────────────────────────────────────

def get_split_info(seed: int) -> dict:
    """
    Hitung jumlah citra per subset tanpa membangun tf.data.Dataset.
    Berguna untuk logging atau verifikasi sebelum training.
    """
    split_root = Path(__file__).parent.parent / "dataset" / "split" / f"seed{seed}"
    info = {}
    for subset in ("train", "val", "test"):
        subset_dir = split_root / subset
        count = 0
        for cls in CLASS_NAMES:
            cls_dir = subset_dir / cls
            if cls_dir.exists():
                count += len(list(cls_dir.glob("*.jpg"))) \
                       + len(list(cls_dir.glob("*.jpeg"))) \
                       + len(list(cls_dir.glob("*.png")))
        info[subset] = count
    return info
