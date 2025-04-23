import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(df, bands=['B02', 'B03', 'B04'],
                 log_scale=False,
                 title="SSIM Metrics Over Training Epochs",
                 y_label="SSIM",
                 verbose=False,
                 save=False,
                 save_path="ssim_metrics_plot.svg",
                 color_palette="plasma"):
    """
    Plot metrics with customizable bands, title and color palette.

    Args:
        df (DataFrame): DataFrame containing metrics
        bands (list): List of band names to plot
        log_scale (bool): Whether to use logarithmic y-scale
        title (str): Plot title
        y_label (str): Y-axis label
        verbose (bool): Whether to display the plot
        save (bool): Whether to save the plot
        save_path (str): Path to save the figure
        color_palette (str): Name of color palette to use
    """
    # Set up color palette
    colors = sns.color_palette(color_palette, len(bands))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loop through bands to plot
    for i, band in enumerate(bands):
        # Plot training curves (dashed)
        ax.plot(df['epoch'], df[f'train_{band}'], '--',
                label=f'Train {band}', color=colors[i])

        # Plot validation curves (solid)
        ax.plot(df['epoch'], df[f'val_{band}'],
                label=f'Val {band}', color=colors[i])

    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')

    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Save and show
    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=300)

    if verbose:
        plt.show()
    plt.close()


def plot_training_loss(df,
                       title="Training and Validation Loss",
                       y_label="Loss",
                       log_scale=False,
                       verbose=False,
                       save=False,
                       save_path="loss_plot.svg",
                       color_palette="plasma"):
    """
    Plot training and validation loss over epochs.

    Args:
        df (DataFrame): DataFrame containing 'train_loss' and 'val_loss' columns and 'epoch' column
        title (str): Plot title
        y_label (str): Y-axis label
        log_scale (bool): Whether to use logarithmic y-scale
        verbose (bool): Whether to display the plot
        save (bool): Whether to save the plot
        save_path (str): Path to save the figure
        color_palette (str): Name of color palette to use
    """
    # Set up color palette
    colors = sns.color_palette(color_palette, 2)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df["train_loss"], label='Training Loss', color=colors[0])
    ax.plot(df["epoch"], df["val_loss"], label='Validation Loss', color=colors[1])

    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')

    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Save and show
    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=300)

    if verbose:
        plt.show()
    plt.close()