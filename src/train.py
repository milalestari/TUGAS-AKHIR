"""
train.py — Fase 2
Training loop 2-stage (freeze → fine-tune) untuk EfficientNetB0 dan ResNet50.

Stage 1: freeze base model, LR=1e-3, 10 epoch
Stage 2: unfreeze top layers, LR=1e-4, 10 epoch

Callbacks (identik kedua stage):
    EarlyStopping (patience=5, monitor=val_loss)
    ReduceLROnPlateau (factor=0.5, patience=5, monitor=val_loss)
    ModelCheckpoint (monitor=val_accuracy, save_best_only=True)

Usage:
    from src.train import train_two_stage
    history = train_two_stage(
        architecture="efficientnetb0",
        seed=42,
        preprocess_fn=tf.keras.applications.efficientnet.preprocess_input,
        ds_train=ds_train,
        ds_val=ds_val,
        checkpoint_path="models/efficientnetb0_seed42.keras",
    )
"""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from model_builder import (
    build_model,
    unfreeze_for_fine_tune,
    build_callbacks,
    print_model_info,
    set_seed,
    EPOCHS_STAGE1,
    EPOCHS_STAGE2,
)


def train_two_stage(
    architecture: str,
    seed: int,
    preprocess_fn,
    ds_train: tf.data.Dataset,
    ds_val: tf.data.Dataset,
    checkpoint_path: str,
    log_dir_stage1: str = None,
    log_dir_stage2: str = None,
    verbose: int = 1,
) -> dict:
    """
    Training dua tahap untuk satu arsitektur × satu seed.

    Parameters
    ----------
    architecture   : str  — "efficientnetb0" | "resnet50"
    seed          : int  — random seed
    preprocess_fn : callable — preprocess_input per arsitektur
    ds_train     : tf.data.Dataset — augmented train dataset
    ds_val       : tf.data.Dataset — non-augmented val dataset
    checkpoint_path : str — path untuk menyimpan best model
    log_dir_stage1 : str — TensorBoard log untuk Stage 1 (opsional)
    log_dir_stage2 : str — TensorBoard log untuk Stage 2 (opsional)
    verbose      : int  — 0=silent, 1=progress bar

    Returns
    -------
    dict — {"stage1": History, "stage2": History, "model": tf.keras.Model}
    """
    set_seed(seed)

    # ── Bangun model Stage 1 (freeze) ────────────────────────────────
    if verbose:
        print("=" * 60)
        print(f"Building {architecture} (seed={seed})")
        print("=" * 60)

    model, layer_names, total_layers = build_model(
        architecture=architecture,
        seed=seed,
        preprocess_fn=preprocess_fn,
    )

    if verbose:
        print_model_info(model)
        print()
        print(f"STAGE 1 — Freeze base model, LR={model.optimizer.learning_rate.numpy():.0e}")
        print(f"  Epochs    : {EPOCHS_STAGE1}")
        print(f"  Callbacks : EarlyStopping(p={5}), "
              f"ReduceLROnPlateau(f=0.5,p=5), "
              f"ModelCheckpoint(val_accuracy)")

    # ── Stage 1: Feature Extraction ───────────────────────────────
    stage1_checkpoint = checkpoint_path.replace(".keras", "_stage1.keras")

    callbacks_s1 = build_callbacks(
        checkpoint_path=stage1_checkpoint,
        log_dir=log_dir_stage1,
    )

    print()
    history_stage1 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS_STAGE1,
        callbacks=callbacks_s1,
        verbose=verbose,
    )

    # Muat best weights Stage 1
    model.load_weights(stage1_checkpoint)

    if verbose:
        print()
        print(f"Stage 1 done. Best val_loss: "
              f"{min(history_stage1.history['val_loss']):.4f}")
        print(f"Stage 1 checkpoint : {stage1_checkpoint}")

    # ── Stage 2: Fine-Tuning ────────────────────────────────────
    if verbose:
        print()
        print(f"STAGE 2 — Unfreeze top layers, LR={1e-4:.0e}")
        print(f"  Epochs    : {EPOCHS_STAGE2}")

    model = unfreeze_for_fine_tune(model)

    if verbose:
        print_model_info(model)

    callbacks_s2 = build_callbacks(
        checkpoint_path=checkpoint_path,
        log_dir=log_dir_stage2,
    )

    print()
    history_stage2 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS_STAGE2,
        callbacks=callbacks_s2,
        verbose=verbose,
    )

    # Muat best weights Stage 2 (yang final)
    model.load_weights(checkpoint_path)

    if verbose:
        print()
        print(f"Stage 2 done. Best val_loss: "
              f"{min(history_stage2.history['val_loss']):.4f}")
        print(f"Final checkpoint : {checkpoint_path}")

    return {
        "stage1": history_stage1,
        "stage2": history_stage2,
        "model": model,
        "stage1_checkpoint": stage1_checkpoint,
        "final_checkpoint": checkpoint_path,
    }


def train_all_seeds(
    architecture: str,
    preprocess_fn,
    get_datasets_fn,
    models_dir: str,
    results_dir: str = None,
    seeds: list = None,
    verbose: int = 1,
) -> dict:
    """
    Training 5 seed × 1 arsitektur.

    Parameters
    ----------
    architecture   : str  — "efficientnetb0" | "resnet50"
    preprocess_fn : callable — preprocess_input per arsitektur
    get_datasets_fn : callable — fungsi(seed) → (ds_train, ds_val)
                       Contoh:
                           def get_datasets(seed):
                               from data_pipeline import build_tf_dataset
                               ds_train = build_tf_dataset(seed, "train",
                                              preprocess_fn, augment=True)
                               ds_val   = build_tf_dataset(seed, "val",
                                              preprocess_fn, augment=False)
                               return ds_train, ds_val
    models_dir  : str — direktori menyimpan .keras
    results_dir : str — direktori menyimpan history (JSON, opsional)
    seeds       : list — [42, 123, 2024, 7, 99]
    verbose     : int

    Returns
    -------
    dict — {seed: {"stage1": History, "stage2": History, ...}, ...}
    """
    if seeds is None:
        from model_builder import SEEDS
        seeds = SEEDS

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for seed in seeds:
        print()
        print("=" * 60)
        print(f"TRAINING {architecture.upper()} — seed={seed} "
              f"({seeds.index(seed)+1}/{len(seeds)})")
        print("=" * 60)

        # Load datasets
        ds_train, ds_val = get_datasets_fn(seed)

        # Checkpoint path
        arch_name = architecture.lower().replace("-", "_")
        ckpt_path = str(models_dir / f"{arch_name}_seed{seed}.keras")

        # Train
        result = train_two_stage(
            architecture=architecture,
            seed=seed,
            preprocess_fn=preprocess_fn,
            ds_train=ds_train,
            ds_val=ds_val,
            checkpoint_path=ckpt_path,
            verbose=verbose,
        )

        results[seed] = result

        # Simpan history sebagai JSON
        if results_dir:
            import json
            history_combined = {
                f"stage1_{k}": v
                for k, v in result["stage1"].history.items()
            }
            history_combined.update({
                f"stage2_{k}": v
                for k, v in result["stage2"].history.items()
            })
            hist_path = results_dir / f"history_{arch_name}_seed{seed}.json"
            with open(hist_path, "w") as f:
                json.dump(history_combined, f, indent=2)
            print(f"History saved: {hist_path}")

        print()
        print(f"SEED {seed} COMPLETE — checkpoint: {ckpt_path}")

    return results
