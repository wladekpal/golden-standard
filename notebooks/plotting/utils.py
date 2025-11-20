import os
import pandas as pd
import numpy as np
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mc
import colorsys


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


    for run, history in zip(runs, all_histories):
        
        scores = []
        for metric in metrics.keys():
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
            final_group_data = np.array([d[:min_len, :] for d in correct_data]).reshape(-1, min_len, len(metrics))

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
            metric_names=metrics_names.values(),
            algorithms=keys,
            row_height=0.7,
            xlabel=None,
            subfigure_width=10.0
        )
    # plt.title(title, fontsize="xx-large")

    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'{title}.png'),bbox_inches='tight')



def draw_interval_estimates_plot_per_alg(runs, keys, metrics_names, title, figures_path="./figures"):
    aggregate_func = lambda x: np.array([metrics.aggregate_iqm(x[:, i]) for i in range(x.shape[-1])])


    aggregate_scores, aggregate_scores_cis = rly.get_interval_estimates(
        runs, aggregate_func, reps=500
    )

    def plot_interval_estimates(point_estimates,
                                interval_estimates,
                                metric_names,
                                algorithms=None,
                                colors=None,
                                color_palette='colorblind',
                                max_ticks=4,
                                subfigure_width=3.4,
                                row_height=0.37,
                                xlabel_y_coordinate=-0.1,
                                xlabel='Normalized Score',
                                **kwargs):

        if algorithms is None:
            algorithms = list(point_estimates.keys())
        num_metrics = len(point_estimates[algorithms[0]])
        figsize = (subfigure_width * num_metrics, row_height * len(algorithms))
        fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize)
        if colors is None:
            color_palette = sns.color_palette(color_palette, n_colors=(len(algorithms) + 1) // 2)[::-1]
            colors = {}
            for i, alg in enumerate(algorithms):
                colors[alg] = color_palette[i // 2]
        h = kwargs.pop('interval_height', 0.6)

        for idx, metric_name in enumerate(metric_names):
            for alg_idx, algorithm in enumerate(algorithms):
                ax = axes[idx] if num_metrics > 1 else axes
                # Plot interval estimates.
                lower, upper = interval_estimates[algorithm][:, idx]
                
                ax.barh(
                    y=alg_idx,
                    width=upper - lower,
                    height=h,
                    left=lower,
                    color=colors[algorithm],
                    alpha=0.75,
                    label=algorithm)
                # Plot point estimates.
                ax.vlines(
                    x=point_estimates[algorithm][idx],
                    ymin=alg_idx - (7.5 * h / 16),
                    ymax=alg_idx + (6 * h / 16),
                    label=algorithm,
                    color='k',
                    alpha=0.5)
                if alg_idx % 2 == 0 and alg_idx > 0:
                    plt.hlines(alg_idx - 0.5, 0, 1.0, color='gray')

            ax.set_yticks(list(range(len(algorithms))))
            ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
            if idx != 0:
                ax.set_yticks([])
            else:
                ax.set_yticklabels(algorithms, fontsize='x-large')
            ax.set_title(metric_name, fontsize='xx-large')
            ax.tick_params(axis='both', which='major')
            plot_utils._decorate_axis(ax, ticklabelsize='xx-large', wrect=5)
            ax.spines['left'].set_visible(False)
            ax.set_xlim(left=-0.05, right=1.05)
            ax.grid(True, axis='x', alpha=0.25)
        fig.text(0.4, xlabel_y_coordinate, xlabel, ha='center', fontsize='xx-large')
        plt.subplots_adjust(wspace=kwargs.pop('wspace', 0.11), left=0.0)
        return fig, axes



    plot_interval_estimates(
            aggregate_scores,
            aggregate_scores_cis,
            metric_names=metrics_names.values(),
            algorithms=keys,
            row_height=0.7,
            xlabel=None,
            subfigure_width=10.0
        )

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'{title}.png'),bbox_inches='tight')
    plt.savefig(os.path.join(figures_path, f'{title}.pdf'),bbox_inches='tight')





def draw_curves_plot(runs, keys, metrics_names, title, figures_path="./figures"):

    fig, axes = plt.subplots(1, len(metrics_names))

    if len(metrics_names) == 1:
        axes = [axes]

    fig.set_figheight(10)
    fig.set_figwidth(len(metrics_names) * 10 + 5)

    min_len = 1000000
    for v in runs.values():
        min_len = min(min_len, v.shape[1])

    for metric_idx, metric_name in enumerate(metrics_names.values()):
        metric_data = {k:data[:,:,metric_idx] for k,data in runs.items()}

        frames = np.arange(0, min_len, 1)
        frames[-1] -= 1
        ale_frames_scores_dict = {algorithm: score[:, frames] for algorithm, score in metric_data.items()}
        iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])])
        iqm_scores, iqm_cis = rly.get_interval_estimates(ale_frames_scores_dict, iqm, reps=2000)

        colors = {alg:return_color(alg) for alg in keys} 

        plot_utils.plot_sample_efficiency_curve(
                frames + 1,
                iqm_scores,
                iqm_cis,
                algorithms=keys,
                colors=colors,
                xlabel=r"Epochs",
                ylabel=None,
                legend=False,
                grid_alpha=0.4,
                ax=axes[metric_idx]
            )
        axes[metric_idx].set_title(metric_name, fontsize=30)
    plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    # plt.suptitle(title, fontsize="xx-large", va='bottom')
    plt.tight_layout(w_pad=3.0)
    plt.savefig(os.path.join(figures_path, f"{title}.png"),bbox_inches='tight')
    plt.savefig(os.path.join(figures_path, f"{title}.pdf"),bbox_inches='tight')


def lighten_color(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def return_color(agent, train=True):
    color_palette = sns.color_palette('colorblind')
    idx = 0

    if 'GCDQN' in agent:
        if 'MC' in agent:
            idx = 0
        else:
            idx = 1
    elif 'C-LEARN' in agent:
        if 'MC' in agent:
            idx = 2
        else:
            idx = 3
    elif 'CRL' in agent:
        idx = 4
    elif 'GCIQL' in agent:
        if 'MC' in agent:
            idx = 8
        else:
            idx = 7
    else:
        idx = 5

    c = color_palette[idx]

    if not train:
        c = lighten_color(c, amount=0.4)

    return c
    