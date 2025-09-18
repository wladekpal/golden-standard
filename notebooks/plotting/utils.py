import os
import pandas as pd
import numpy as np
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import matplotlib.pyplot as plt


# Function to extract metrics from runs
def extract_run_data(run, download_again: bool = False):
    # Extract run config and metrics
    config = run.config
    summary = run.summary._json_dict
    # TODO: potentially just load csv
    if os.path.exists(f"./data/history_{run.id}.csv") and not download_again:
        history = pd.read_csv(f"./data/history_{run.id}.csv")
    else:
        history = run.history(pandas=True)

        history.to_csv(f"./data/history_{run.id}.csv", index=False)
    # Combine run information into a single dictionary

    run_data = {"run_id": run.id, "name": run.name, **config, **summary}
    return run_data, history


def create_rliable_compatible_data(df, metric):
    return df[metric].to_numpy()


def moving_average_smoothing(data, window_size=5):
    """Apply a moving average filter to the last axis of the input data, ensuring no wrap-around."""
    pad_width = window_size // 2
    padded_data = np.pad(data, pad_width, mode="edge")
    smoothed = np.convolve(padded_data, np.ones(window_size) / window_size, mode="valid")
    return smoothed


def aggregate_data_from_wandb(
    runs,
    metrics: list[str],
    possible_names: list,
    grouping_func,
    return_last_step: bool = True,
    download_again: bool = False,
):
    # Download/load history from wandb
    all_histories = []
    for run in runs:
        run_data, history = extract_run_data(run, download_again)
        all_histories.append(history)


    data = {possible_name: [] for possible_name in possible_names}
    epochs = {possible_name: [] for possible_name in possible_names}
    seeds = {possible_name: [] for possible_name in possible_names}


    for run, history in zip(runs, all_histories):
        
        scores = []
        for metric in metrics:
            scores.append(create_rliable_compatible_data(history, metric))

        scores = np.stack(scores, axis=-1)
        epoch = create_rliable_compatible_data(history, 'epoch')

        grouping_param = grouping_func(run.config)
        data[grouping_param].append(scores)
        epochs[grouping_param].append(epoch)



    final_data = {possible_name: [] for possible_name in possible_names}

    for group, group_data in data.items():
        correct_data = []
        min_len = 100000
        for run_data in group_data:
            correct_data.append(run_data[~np.isnan(run_data[:, 0]), :])
            min_len = min(min_len, correct_data[-1].shape[0])

        if return_last_step:
            final_group_data = np.array([d[-1, :] for d in correct_data])
        else:
            final_group_data = np.array([d[None, :min_len] for d in correct_data]).reshape(-1, 1, min_len)

        final_data[group] = final_group_data

    return final_data


def draw_interval_estimates_plot(runs, keys, metrics_names, title, figures_path="./figures"):
    aggregate_func = lambda x: np.array([metrics.aggregate_iqm(x[:, i]) for i in range(x.shape[-1])])

    aggregate_scores, aggregate_scores_cis = rly.get_interval_estimates(
        runs, aggregate_func, reps=500
    )


    plot_utils.plot_interval_estimates(
            aggregate_scores,
            aggregate_scores_cis,
            metric_names=metrics_names,
            algorithms=keys,
            row_height=0.7,
            xlabel=title,
            subfigure_width=5.0
        )
    # plt.title(title, fontsize="xx-large")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'{title}.png'),bbox_inches='tight')