from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_history(path: Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def epochs_from_history(history: dict[str, Any]) -> list[int]:
    for key in [
        'train_loss',
        'val_loss',
        'val_f1_macro',
        'val_accuracy',
        'epoch_time_sec',
    ]:
        if key in history and isinstance(history[key], list):
            return list(range(1, len(history[key]) + 1))
    raise ValueError('Could not infer number of epochs from training history.')


def plot_loss_and_f1(history: dict[str, Any], output_path: Path, title: str) -> None:
    epochs = epochs_from_history(history)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    ax2.set_ylim(0.0, 0.85)

    # Left axis: losses
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], label='Train Loss',
                 color='slategray', linestyle='-', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], label='Val Loss',
                 color='purple', linestyle='-', linewidth=2)

    # Right axis: F1 curves
    f1_configs = [
        ('val_car_f1', 'Car F1', 'red', '--', 2),
        ('val_bus_f1', 'Bus F1', 'green', '--', 2),
        ('val_truck_f1', 'Truck F1', 'blue', '--', 2),
        ('val_f1_macro', 'Macro F1', 'black', '--', 4),
    ]

    for key, label, color, ls, lw in f1_configs:
        if key in history:
            ax2.plot(epochs, history[key], label=label,
                     color=color, linestyle=ls, linewidth=lw)

    best_epoch = history.get('best_epoch')
    if best_epoch is not None:
        ax1.axvline(best_epoch, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('F1 score')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
        borderaxespad=0.
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)



def plot_loss_and_macro_metrics(history: dict[str, Any], output_path: Path, title: str) -> None:
    epochs = epochs_from_history(history)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    ax2.set_ylim(0.0, 0.85)

    # Left axis: losses
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], label='Train Loss',
                 color='slategray', linestyle='-', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], label='Val Loss',
                 color='purple', linestyle='-', linewidth=2)

    # Right axis: macro metrics
    metric_configs = [
        ('val_precision_macro', 'Macro Precision', 'darkorange', ':', 2),
        ('val_recall_macro', 'Macro Recall', 'deepskyblue', '-.', 2),
        ('val_f1_macro', 'Macro F1', 'black', '--', 4),
    ]

    for key, label, color, ls, lw in metric_configs:
        if key in history:
            ax2.plot(epochs, history[key], label=label,
                     color=color, linestyle=ls, linewidth=lw)

    best_epoch = history.get('best_epoch')
    if best_epoch is not None:
        ax1.axvline(best_epoch, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Metric value')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
        borderaxespad=0.
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot training curves from training_history.json')
    parser.add_argument('--history', type=str, required=True, help='Path to training_history.json')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory for saving plots')
    parser.add_argument('--name', type=str, default='model', help='Short model name for titles and filenames')
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    history_path = Path(args.history)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(history_path)
    name = args.name.strip()

    plot1 = output_dir / f'{name}_loss_f1.png'
    plot2 = output_dir / f'{name}_loss_macro_metrics.png'

    plot_loss_and_f1(
        history,
        plot1,
        title=f'{name}: Train/Val Loss and Class/Macro F1 by Epoch',
    )
    plot_loss_and_macro_metrics(
        history,
        plot2,
        title=f'{name}: Train/Val Loss and Macro Precision/Recall/F1 by Epoch',
    )

    print(f'Saved: {plot1}')
    print(f'Saved: {plot2}')


if __name__ == '__main__':
    main()
