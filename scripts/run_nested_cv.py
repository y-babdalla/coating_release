"""Performs cross-validation and plotting for multiple regression models."""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401 (Required for plotting styles)
import seaborn as sns
from tqdm import tqdm

from src.cross_validation import nested_cross_validation
from src.process_spectra import process_spectrum_dataframe

warnings.filterwarnings("ignore")
plt.style.use(["science", "no-latex"])


def plot_scores(plot_data: dict[str, pd.DataFrame], score_types: list[str]) -> None:
    """Plot multiple regression metrics using strip and point plots.

    Create a grid of plots (one for each metric in 'score_types'),
    displaying the distribution of scores via stripplot and the
    mean and standard deviation via pointplot.

    Parameters:
        plot_data: A dictionary where each key is a score name (e.g., 'R2 Score'),
            and each value is a DataFrame with two columns: one for the score itself
            and one for the model name.
        score_types: A list of strings indicating which metrics to plot
            (e.g., ['R2 Score', 'MAE', 'MSE']).
    """
    sns.set_theme(style="whitegrid")
    font_size = 26
    plt.rcParams.update({"font.size": font_size})

    fig, axes = plt.subplots(1, 3, figsize=(36, 10))

    for i, score_type in enumerate(score_types):
        ax = axes[i]

        sns.stripplot(
            x="Model",
            y=score_type,
            data=plot_data[score_type],
            dodge=True,
            marker="o",
            alpha=0.7,
            color="black",
            ax=ax,
        )

        sns.pointplot(
            x="Model",
            y=score_type,
            data=plot_data[score_type],
            dodge=0.5,
            join=False,
            capsize=0.2,
            ci="sd",
            scale=2,
            marker="o",
            errwidth=2,
            ax=ax,
            palette="Set2",
        )

        ax.set_title(f"{score_type}", fontsize=font_size)
        ax.set_xlabel("Model", fontsize=font_size)
        ax.set_ylabel(score_type, fontsize=font_size)
        ax.tick_params(axis="x", labelsize=font_size)
        ax.tick_params(axis="y", labelsize=font_size)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig("new/model_summary_grid.png", dpi=300)
    plt.show()


def main() -> None:
    """Perform nested cross-validation for multiple models and plot their scores.

    Steps:
        1. Load and process data from an Excel file, removing unnecessary columns
           and downsampling spectra.
        2. Define a list of model names and perform nested cross-validation for each.
        3. Collect R2, MAE, and MSE scores for each model and format them for plotting.
        4. Call 'plot_scores' to produce a grid of plots for model performance comparison.
    """
    model_names: list[str] = ["LightGBM", "XGBoost", "KNN", "SVM", "RF"]

    df = pd.read_excel("data/coating_release.xlsx", sheet_name="all_data")
    df = df.drop(["polysaccharide name"], axis=1)

    processed_data = process_spectrum_dataframe(df, downsample=20)
    x = processed_data.drop(["release", "index"], axis=1)
    y = processed_data["release"]

    scores_data: dict[str, dict[str, list[float]]] = {
        model: {score: [] for score in ["R2 Score", "MAE", "MSE"]} for model in model_names
    }

    for model_name in tqdm(model_names, desc="Models"):
        results_df: pd.DataFrame = nested_cross_validation(x=x, y=y, model_name=model_name)
        scores_data[model_name]["R2 Score"].extend(results_df["R2 Score"].tolist())
        scores_data[model_name]["MAE"].extend(results_df["MAE"].tolist())
        scores_data[model_name]["MSE"].extend(results_df["MSE"].tolist())

    plot_data: dict[str, pd.DataFrame] = {}
    for score_type in ["R2 Score", "MAE", "MSE"]:
        scores = []
        models = []
        for model, model_scores in scores_data.items():
            scores.extend(model_scores[score_type])
            models.extend([model] * len(model_scores[score_type]))
        plot_data[score_type] = pd.DataFrame({score_type: scores, "Model": models})

    plot_scores(plot_data, ["R2 Score", "MAE", "MSE"])


if __name__ == "__main__":
    main()
