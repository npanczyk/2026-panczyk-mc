import matplotlib.pyplot as plt


def plot_power(history, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["time"], history["desired_power"], label="Desired Power")
    ax.plot(history["time"], history["observed_power"], label="Observed Power")
    ax.plot(history["time"], history["actual_power"], label="Actual Power")
    ax.legend()
    ax.set_xlabel("Time, [s]")
    ax.set_ylabel("Fraction of Total Power")
    ax.grid()
    fig.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/power.png", dpi=300)

    return fig
