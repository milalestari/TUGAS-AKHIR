"""
model_builder.py — Fase 2
Membangun arsitektur EfficientNetB0 dan ResNet50 dengan
classification head identik sesuai proposal 3.3.6.

Classification head (identik kedua model):
    GlobalAveragePooling2D
            BatchNormalization
            Dropout(0.3)
            Dense(512, activation='relu')
            BatchNormalization
            Dropout(0.3)
            Dense(5, activation='softmax')

Usage:
    from src.model_builder import build_model
    model = build_model("efficientnetb0", seed=42)
    model.summary()
"""

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks

# ──────────────────────────────────────────────
# Konfigurasi (hardcoded — konsisten proposal 3.3.6)
# ──────────────────────────────────────────────

IMG_SIZE = (224, 224)
NUM_CLASSES = 5

# Class names — berurutan sesuai alphabetical
# (sesuai urutan folder di dataset/split/{train,val,test}/)
CLASS_NAMES = ["Berkabut", "Berawan", "Cerah", "Hujan", "Mendung"]

# Classification head
HEAD_DROPOUT_RATE  = 0.3
HEAD_DENSE_UNITS     = 512
HEAD_ACTIVATION     = "relu"

# Training hyperparameters
BATCH_SIZE       = 32
STAGE1_LR         = 1e-3    # Feature extraction (freeze)
STAGE2_LR         = 1e-4    # Fine-tuning (unfreeze)
EPOCHS_STAGE1      = 10       # Feature extraction
EPOCHS_STAGE2      = 10       # Fine-tuning

# Callbacks (identik kedua stage)
ES_PATIENCE        = 5        # EarlyStopping: patience
RLROP_FACTOR       = 0.5      # ReduceLROnPlateau: factor
RLROP_PATIENCE     = 5        # ReduceLROnPlateau: patience

# Unfreeze layer count (proposal 3.3.6)
EFFICIENTNET_UNFREEZE = 20     # 20 layer terakhir dari 237
RESNET50_UNFREEZE    = 15     # 15 layer terakhir dari 175

# Seeds
SEEDS = [42, 123, 2024, 7, 99]


# ──────────────────────────────────────────────
# Seed utilities
# ──────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set semua random seed untuk reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ──────────────────────────────────────────────
# Classification head builder
# ──────────────────────────────────────────────

def build_classification_head(input_tensor):
    """
    Classification head identik untuk EfficientNetB0 dan ResNet50.
    Sesuai proposal 3.3.6:
        GAP → BN → Dropout(0.3) → Dense(512, ReLU) → BN → Dropout(0.3) → Dense(5, softmax)
    """
    x = layers.GlobalAveragePooling2D()(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(HEAD_DROPOUT_RATE)(x)
    x = layers.Dense(HEAD_DENSE_UNITS, activation=HEAD_ACTIVATION)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(HEAD_DROPOUT_RATE)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return outputs


# ──────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────

def build_model(
    architecture: str,
    seed: int,
    preprocess_fn,
) -> tuple:
    """
    Bangun model {EfficientNetB0, ResNet50} dengan classification head.

    Parameters
    ----------
    architecture : str — "efficientnetb0" | "resnet50"
    seed        : int — random seed
    preprocess_fn : callable — tf.keras.applications.efficientnet.preprocess_input
                              atau resnet50.preprocess_input

    Returns
    -------
    model       : tf.keras.Model — model yang sudah dicompile (Stage 1: freeze)
    unfreeze_layer_names : list[str] — nama layer yang akan di-unfreeze di Stage 2
    base_layer_count    : int — total jumlah layer base model
    """
    set_seed(seed)

    if architecture == "efficientnetb0":
        from tensorflow.keras.applications import EfficientNetB0
        base = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(*IMG_SIZE, 3),
        )
        n_unfreeze = EFFICIENTNET_UNFREEZE  # 20

    elif architecture == "resnet50":
        from tensorflow.keras.applications import ResNet50
        base = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(*IMG_SIZE, 3),
        )
        n_unfreeze = RESNET50_UNFREEZE  # 15

    else:
        raise ValueError(
            f"architecture harus 'efficientnetb0' atau 'resnet50', "
            f"dapat: '{architecture}'"
        )

    # Freeze semua layer base model (Stage 1)
    for layer in base.layers:
        layer.trainable = False

    # Input
    inputs = layers.Input(shape=(*IMG_SIZE, 3), name="input_image")

    # Preprocess
    x = preprocess_fn(inputs)

    # Base
    x = base(x, training=False)   # training=False → BN di mode inference

    # Classification head
    outputs = build_classification_head(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=architecture)

    # Compile Stage 1 (freeze)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=STAGE1_LR),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # Catat info untuk Stage 2
    total_layers = len(base.layers)
    layer_names_to_unfreeze = [
        layer.name for layer in base.layers[total_layers - n_unfreeze:]
    ]

    return model, layer_names_to_unfreeze, total_layers


# ──────────────────────────────────────────────
# Stage 2: Unfreeze + Recompile
# ──────────────────────────────────────────────

def unfreeze_for_fine_tune(model: tf.keras.Model) -> tf.keras.Model:
    """
    Unfreeze layer terakhir base model untuk fine-tuning (Stage 2).
    Mengikuti n_unfreeze yang sudah dicatat saat build_model.

    Parameters
    ----------
    model : tf.keras.Model — model Stage 1 (freeze)

    Returns
    -------
    tf.keras.Model — model Stage 2 (unfreeze sebagian)
    """
    # EfficientNetB0 atau ResNet50 base ada di model.layers[1]
    base = model.layers[1]

    # Cari layer yang trainable=False → jadikan True
    for layer in base.layers:
        layer.trainable = False   # reset semua dulu

    # Unfreeze layer terakhir sesuai arsitektur
    if "efficientnetb0" in model.name.lower():
        n_unfreeze = EFFICIENTNET_UNFREEZE
    elif "resnet50" in model.name.lower():
        n_unfreeze = RESNET50_UNFREEZE
    else:
        n_unfreeze = 20  # fallback

    total_layers = len(base.layers)
    for layer in base.layers[total_layers - n_unfreeze:]:
        layer.trainable = True

    # Recompile dengan LR Stage 2
    model.compile(
        optimizer=optimizers.Adam(learning_rate=STAGE2_LR),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    return model


# ──────────────────────────────────────────────
# Callbacks builder
# ──────────────────────────────────────────────

def build_callbacks(
    checkpoint_path: str,
    log_dir: str = None,
) -> list:
    """
    Bangun 3 callbacks sesuai proposal 3.3.6:
        1. EarlyStopping (patience=5, monitor val_loss)
        2. ReduceLROnPlateau (factor=0.5, patience=5, monitor val_loss)
        3. ModelCheckpoint (monitor val_accuracy, save_best_only=True)

    Parameters
    ----------
    checkpoint_path : str — path untuk .keras checkpoint
    log_dir        : str — path untuk TensorBoard logs (opsional)

    Returns
    -------
    list of tf.keras.callbacks.Callback
    """
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=ES_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=RLROP_FACTOR,
            patience=RLROP_PATIENCE,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    if log_dir:
        cb_list.append(
            callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
        )

    return cb_list


# ──────────────────────────────────────────────
# Info helper
# ──────────────────────────────────────────────

def print_model_info(model: tf.keras.Model) -> None:
    """Cetak ringkasan parameter model."""
    print(f"Model      : {model.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output     : {model.output_shape}")
    trainable = sum(
        1 for layer in model.layers if layer.trainable
    )
    frozen = len(model.layers) - trainable
    print(f"Trainable layers : {trainable}")
    print(f"Frozen layers    : {frozen}")
    print(f"Total parameters: {model.count_params():,}")
