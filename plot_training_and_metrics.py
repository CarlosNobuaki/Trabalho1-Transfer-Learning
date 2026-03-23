import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_training_history(history_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(history_csv)
    if df.empty:
        raise ValueError("training_history.csv esta vazio.")

    df = df.copy()
    df["global_epoch"] = range(1, len(df) + 1)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    sns.lineplot(
        data=df,
        x="global_epoch",
        y="train_loss",
        marker="o",
        label="train",
        ax=axes[0],
    )
    sns.lineplot(
        data=df,
        x="global_epoch",
        y="val_loss",
        marker="o",
        label="val",
        ax=axes[0],
    )
    axes[0].set_title("Loss por epoca")
    axes[0].set_xlabel("Epoca global")
    axes[0].set_ylabel("Loss")

    sns.lineplot(
        data=df,
        x="global_epoch",
        y="train_accuracy",
        marker="o",
        label="train",
        ax=axes[1],
    )
    sns.lineplot(
        data=df,
        x="global_epoch",
        y="val_accuracy",
        marker="o",
        label="val",
        ax=axes[1],
    )
    axes[1].set_title("Accuracy por epoca")
    axes[1].set_xlabel("Epoca global")
    axes[1].set_ylabel("Accuracy")

    sns.lineplot(
        data=df,
        x="global_epoch",
        y="train_f1_macro",
        marker="o",
        label="train",
        ax=axes[2],
    )
    sns.lineplot(
        data=df,
        x="global_epoch",
        y="val_f1_macro",
        marker="o",
        label="val",
        ax=axes[2],
    )
    axes[2].set_title("F1 macro por epoca")
    axes[2].set_xlabel("Epoca global")
    axes[2].set_ylabel("F1 macro")

    for stage_break in df.index[df["stage"].ne(df["stage"].shift())].tolist()[1:]:
        x = stage_break + 1
        for ax in axes:
            ax.axvline(x=x, color="gray", linestyle="--", alpha=0.6)

    fig.suptitle("Historico de treinamento (Head + Fine-tuning)", fontsize=14)
    fig.tight_layout()
    out_file = output_dir / "training_history_curves.png"
    fig.savefig(out_file, dpi=180)
    plt.close(fig)



def plot_metrics_summary(metrics_json: Path, output_dir: Path) -> None:
    with open(metrics_json, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    classes = data["splits"]["classes"]
    eval_data = data["evaluation"]
    splits = ["train", "val", "test"]

    # 1) Grafico de barras com metricas principais por split
    main_metrics = ["accuracy", "f1_macro", "balanced_accuracy", "roc_auc_ovr_macro"]
    rows = []
    for split in splits:
        for metric in main_metrics:
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "value": eval_data[split]["metrics"].get(metric),
                }
            )
    bars_df = pd.DataFrame(rows)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=bars_df, x="metric", y="value", hue="split", ax=ax1)
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Comparacao de metricas por split")
    ax1.set_xlabel("Metrica")
    ax1.set_ylabel("Valor")
    ax1.legend(title="Split")
    fig1.tight_layout()
    fig1.savefig(output_dir / "metrics_by_split.png", dpi=180)
    plt.close(fig1)

    # 2) Heatmaps de matriz de confusao para train/val/test
    fig2, axes = plt.subplots(1, 3, figsize=(20, 5))
    for idx, split in enumerate(splits):
        cm = eval_data[split]["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        sns.heatmap(cm_df, annot=False, cmap="Blues", cbar=True, ax=axes[idx])
        axes[idx].set_title(f"Confusion Matrix - {split}")
        axes[idx].set_xlabel("Predito")
        axes[idx].set_ylabel("Real")
    fig2.tight_layout()
    fig2.savefig(output_dir / "confusion_matrices.png", dpi=180)
    plt.close(fig2)

    # 3) Heatmap de F1 por classe e split
    class_f1_rows = []
    for split in splits:
        report = eval_data[split]["classification_report"]
        for cls in classes:
            class_f1_rows.append(
                {"split": split, "classe": cls, "f1": report[cls]["f1-score"]}
            )

    f1_df = pd.DataFrame(class_f1_rows)
    pivot = f1_df.pivot(index="classe", columns="split", values="f1")

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax3)
    ax3.set_title("F1-score por classe e split")
    fig3.tight_layout()
    fig3.savefig(output_dir / "f1_by_class_split.png", dpi=180)
    plt.close(fig3)



def main() -> None:
    artifacts_dir = Path("artifacts")
    output_dir = artifacts_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    history_csv = artifacts_dir / "training_history.csv"
    metrics_json = artifacts_dir / "metrics_summary.json"

    if not history_csv.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {history_csv}")
    if not metrics_json.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {metrics_json}")

    plot_training_history(history_csv, output_dir)
    plot_metrics_summary(metrics_json, output_dir)

    print("Graficos gerados com sucesso em:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"- {file}")


if __name__ == "__main__":
    main()
