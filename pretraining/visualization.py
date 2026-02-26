import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interpn


def get_density_scatter_plot_visualization(
        process_variables=None,
        ignore_value=-9999,
        min_value=0,
        max_value=60,
):
    """
    Creates a density scatter plot visualization for the regression model
    :param process_variables: A function that takes in inputs, labels, and outputs and returns the processed versions
    :return: A function that takes in inputs, labels, and outputs and returns a boxplot visualization
    """

    def density_scatter_visualization(
            inputs, labels, outputs, bins=30, height_range=range(1, max_value), **kwargs
    ):
        try:
            inputs = inputs.detach().cpu().numpy().squeeze()
            labels = labels.detach().cpu().numpy().squeeze()
            outputs = outputs.detach().cpu().numpy().squeeze()

            if process_variables is not None:
                inputs, labels, outputs = process_variables(inputs, labels, outputs)

            outputs = outputs[labels != ignore_value]
            labels = labels[labels != ignore_value]

            # Check for NaN or infinite values
            if (not np.isfinite(labels).all() or not np.isfinite(outputs).all() or 
                len(labels) == 0 or len(outputs) == 0):
                raise ValueError("Data contains NaN, infinite values, or is empty")

            ax = plt.gca()
            x = np.array(labels).flatten()
            y = np.array(outputs).flatten()

            fig, ax = plt.subplots()

            data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
            z = interpn(
                (x_e[:-1], y_e[:-1]),
                data,
                np.vstack([x, y]).T,
                method="splinef2d",
                bounds_error=False,
            )

            # To be sure to plot all data
            z[np.where(np.isnan(z))] = 0.0

            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

            if np.min(z) < 0:
                z = z + abs(np.min(z) + 0.1)
            norm = Normalize(vmin=np.min(z), vmax=np.max(z), clip=True)

            ax.scatter(x, y, c=z, s=0.1, cmap="viridis", norm=norm, **kwargs)
            plt.xlim([min_value, max_value])
            plt.ylim([min_value, max_value])

            fig = plt.gcf()
            plt.grid(False)
            cbar = fig.colorbar(
                cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, **kwargs
            )
            cbar.ax.set_ylabel("Density")

            ax.plot(
                height_range, height_range, c="r", linewidth=3, label="x=y", linestyle="--"
            )
            ax.set_xlabel("Ground truth height")
            ax.set_ylabel("Predicted height")
            ax.legend()

            return fig
            
        except Exception as e:
            # Return empty figure with error message when visualization fails
            print(f"Density scatter plot visualization failed: {e}")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Visualization failed:\n{str(e)[:100]}...', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.axis('off')
            return fig

    return density_scatter_visualization


def get_input_output_visualization(
        process_variables=None,
        transparent_value=None,
        rgb_channels=[2, 1, 0],
        single_month_scaling=False,
):
    """
    Get a visualization function that plots the input and output of the model.
    :param process_variables: A function that processes the variables before plotting.
    :param transparent_value: The value that should be transparent in the output.
    :param rgb_channels: The channels that should be used for the RGB image.
    :return: A function that can be used for visualization.
    """

    def input_output_visualization(inputs, labels, outputs):
        try:
            inputs = inputs.cpu().detach().numpy()
            if labels is not None:
                labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()

            # Check for NaN or infinite values
            if not np.isfinite(inputs).all() or not np.isfinite(outputs).all():
                raise ValueError("Data contains NaN or infinite values")

            if transparent_value is not None:
                outputs[outputs == transparent_value] = None

            if process_variables is not None:
                inputs, labels, outputs = process_variables(inputs, labels, outputs)

            if inputs.ndim == 6:
                inputs = inputs[:, :, 0, :, :, :]
                labels = labels[:, 0, :, :] if labels is not None else None
                outputs = outputs[:, :, 0, :, :] if outputs.ndim == 5 else outputs

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))

            if not single_month_scaling:
                inputs_normalized = np.clip(inputs / 3000, 0, 1)
            else:
                inputs_normalized = inputs

            # loop over the first two images
            n_images = min(inputs.shape[0], 2)  
            for i in range(n_images):
                if inputs_normalized.ndim == 4:
                    # This is the 2D-case with input (batch_size, Z, H, W), where Z is either 6*12 or 12 (months*channels, or channels where months are collapsed)
                    # plot the RGB image in the first column
                    axs[i, 0].imshow(
                        inputs_normalized[i, rgb_channels, :, :].transpose(1, 2, 0)
                    )  # note that we reverse the order to RGB (matplotlib expects HWC)
                elif inputs_normalized.ndim == 5:
                    # This is the 3D-case with input (batch_size, channels, months, H, W)
                    month_idx = 1
                    axs[i, 0].imshow(
                        inputs_normalized[i, rgb_channels, month_idx , :, :].transpose(1, 2, 0)
                    )  # note that we reverse the order to RGB (matplotlib expects HWC)
                axs[i, 0].set_title(f"Image {i + 1}")

                # plot the model output in the second column
                im = axs[i, 1].imshow(outputs[i, 0, :, :], cmap="viridis", vmin=0, vmax=35)
                axs[i, 1].set_title(f"Output {i + 1}")

            # Create colorbar
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            return fig
            
        except Exception as e:
            # Return empty figure with error message when visualization fails
            print(f"Input-output visualization failed: {e}")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, f'Visualization failed:\n{str(e)[:100]}...', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.axis('off')
            return fig

    return input_output_visualization

def get_visualization_boxplots(
        process_variables=None,
        ignore_value=-9999,
        min_value=0,
        max_value=60,
        step_size=5,
):
    """
    Creates a boxplot visualization for the regression model
    :param process_variables: A function that takes in inputs, labels, and outputs and returns the processed versions
    :return: A function that takes in inputs, labels, and outputs and returns a boxplot visualization
    """

    def visualization_boxplots(inputs, labels, outputs):
        try:
            inputs = inputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()

            if process_variables is not None:
                inputs, labels, outputs = process_variables(inputs, labels, outputs)

            # Check for NaN or infinite values
            if not np.isfinite(labels).all() or not np.isfinite(outputs).all():
                raise ValueError("Data contains NaN or infinite values")

            # Reshape labels and outputs into 1D arrays
            labels_1d = labels.flatten()
            outputs_1d = outputs.flatten()

            # Calculate errors
            errors = outputs_1d - labels_1d

            # Only keep non-zero label locations
            non_zero_label_locs = labels_1d != ignore_value
            labels_1d = labels_1d[non_zero_label_locs]
            errors = errors[non_zero_label_locs]

            # Check if we have any valid data left
            if len(labels_1d) == 0 or len(errors) == 0:
                raise ValueError("No valid data points after filtering")

            # Create bins for 'label' array
            bins = np.arange(min_value, max_value, step_size)
            bin_indices = np.digitize(labels_1d, bins)

            # Initialize lists to hold errors for each bin
            bin_errors = [[] for _ in range(len(bins) + 1)]
            bin_counts = [0] * (len(bins) + 1)
            for bin_idx, error in zip(bin_indices, errors):
                bin_errors[bin_idx].append(error)
                bin_counts[bin_idx] += 1

            # Modify x-axis labels with counts
            x_labels = []
            for i in range(len(bin_counts) - 1):
                x_labels.append(f"{bins[i]}-{bins[i] + step_size} (n={bin_counts[i]:,})")
            x_labels.append(f">{bins[-1]} (n={bin_counts[-1]:,})")

            # Create the boxplot
            plt.figure(figsize=(10, 6))
            box_plot = sns.boxplot(data=bin_errors, fliersize=2)

            box_plot.set_xticklabels(x_labels)

            plt.title("Boxplot of Errors for Tree Height Bins")
            plt.xlabel("Label Bins")
            plt.ylabel("Error")
            plt.xticks(rotation=90)  # Make sure that the labels on x-axis don't overlap

            return plt.gcf()  # return the current figure
            
        except Exception as e:
            # Return empty figure with error message when visualization fails
            print(f"Boxplot visualization failed: {e}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Visualization failed:\n{str(e)[:100]}...', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.axis('off')
            return fig

    return visualization_boxplots

def get_multi_year_evolution_visualization(
        process_variables=None,
        ignore_value=-9999,
        rgb_channels=[3, 2, 1],
):
    """
    Creates a multi-year evolution visualization showing predictions across years and year-over-year changes.
    :param process_variables: A function that takes in inputs, labels, and outputs and returns the processed versions
    :param ignore_value: Value to ignore in the data
    :param rgb_channels: The channels that should be used for the RGB image.
    :param single_month_scaling: Whether to use single month scaling for inputs
    :return: A function that takes in inputs, labels, and outputs and returns a multi-year evolution visualization
    """

    def multi_year_evolution_visualization(inputs, labels, outputs):
        try:
            inputs = inputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()

            # Check for NaN or infinite values
            if not np.isfinite(inputs).all() or not np.isfinite(outputs).all():
                raise ValueError("Data contains NaN or infinite values")

            if process_variables is not None:
                inputs, labels, outputs = process_variables(inputs, labels, outputs)

            # Create figure with 2 rows and 7 columns
            fig, axs = plt.subplots(2, 8, figsize=(32, 8))
            
            inputs_normalized = inputs # Skip normalization because done in dataloader

            # Use the first image from the batch
            batch_idx = 0
            
            # Expected shapes after processing:
            # outputs: batch, 7, 256, 256 (based on user specification)
            # We need to handle this specific shape configuration
            
            # Handle the specific output shape: (batch, years, H, W)
            if outputs.ndim == 4:
                # Shape: (batch, years, H, W) - this is the expected format
                n_years = outputs.shape[1]
            elif outputs.ndim == 5 and outputs.shape[1] == 1:
                # Shape: (batch, 1, years, H, W) - squeeze out the singleton dimension
                outputs = outputs[:, 0, :, :, :]  # Now (batch, years, H, W)
                n_years = outputs.shape[1]
            elif outputs.ndim == 3:
                # Shape: (batch, H, W) - single year case
                n_years = 1
                outputs = outputs[:, np.newaxis, :, :]  # Add year dimension
            else:
                # Fallback - assume 7 years as mentioned in requirements
                n_years = 7
            
            # Ensure we have exactly 7 years for the visualization
            n_years = min(7, n_years)
            
            # Top row: Show predictions for each year (0-35m)
            for year_idx in range(n_years):
                if outputs.ndim >= 3 and outputs.shape[1] > year_idx:
                    prediction = outputs[batch_idx, year_idx, :, :]
                else:
                    # Fallback for single year or insufficient data
                    prediction = outputs[batch_idx, 0, :, :] if outputs.ndim >= 3 else outputs[batch_idx]
                
                im = axs[0, year_idx].imshow(prediction, cmap="viridis", vmin=0, vmax=35)
                axs[0, year_idx].set_title(f"Year {year_idx + 1} Prediction")
                axs[0, year_idx].axis('off')

            # Bottom row: RGB image in first column, then year-over-year changes
            # RGB image (bottom left)
            if inputs_normalized.ndim == 4:
                # Shape: (batch, channels, H, W)
                rgb_image = inputs_normalized[batch_idx, rgb_channels, :, :].transpose(1, 2, 0)
            elif inputs_normalized.ndim == 5:
                # Shape: (batch, channels, months, H, W) - use middle month
                month_idx = inputs_normalized.shape[2] // 2
                rgb_image = inputs_normalized[batch_idx, rgb_channels, month_idx, :, :].transpose(1, 2, 0)
            elif inputs_normalized.ndim == 6:
                # Shape: (batch, channels, years, months, H, W) - use first year, middle month
                year_idx = 0
                month_idx = inputs_normalized.shape[3] // 2
                rgb_image = inputs_normalized[batch_idx, rgb_channels, year_idx, month_idx, :, :].transpose(1, 2, 0)
            else:
                # Fallback: create a placeholder image
                rgb_image = np.zeros((256, 256, 3))
                
            axs[1, 0].imshow(rgb_image)
            axs[1, 0].set_title("RGB Image")
            axs[1, 0].axis('off')

            # Year-over-year changes (columns 1-6)
            for change_idx in range(min(6, n_years - 1)):
                current_year = outputs[batch_idx, change_idx + 1, :, :]
                previous_year = outputs[batch_idx, change_idx, :, :]
                change = current_year - previous_year
                
                im_change = axs[1, change_idx + 1].imshow(change, cmap="RdBu_r", vmin=-2, vmax=2)
                axs[1, change_idx + 1].set_title(f"Change Year {change_idx + 1}→{change_idx + 2}")
                axs[1, change_idx + 1].axis('off')
                
            first_year = outputs[batch_idx, 0, :, :]
            second_year = outputs[batch_idx, 1, :, :]
            last_year = outputs[batch_idx, n_years - 1, :, :]
            diff_first_last = last_year - first_year
            diff_second_last = last_year - second_year

            axs[0, 7].imshow(diff_first_last, cmap="RdBu_r", vmin=-2, vmax=2)
            axs[1, 7].imshow(diff_second_last, cmap="RdBu_r", vmin=-2, vmax=2)
            axs[0, 7].set_title("First Year → Last Year")
            axs[1, 7].set_title("Second Year → Last Year")
            axs[0, 7].axis('off')
            axs[1, 7].axis('off')

            # Add colorbars
            # Colorbar for predictions (top row)
            cbar_ax1 = fig.add_axes([0.96, 0.55, 0.01, 0.35])
            cbar1 = fig.colorbar(im, cax=cbar_ax1)
            cbar1.set_label('Height (m)', rotation=270, labelpad=15)

            # Colorbar for changes (bottom row)
            if n_years > 1:
                cbar_ax2 = fig.add_axes([0.96, 0.1, 0.01, 0.35])
                cbar2 = fig.colorbar(im_change, cax=cbar_ax2)
                cbar2.set_label('Change (m)', rotation=270, labelpad=15)

            plt.subplots_adjust(right=0.95)  # Make room for colorbars

            return fig
            
        except Exception as e:
            # Return empty figure with error message when visualization fails
            print(f"Multi-year evolution visualization failed: {e}")
            fig, ax = plt.subplots(figsize=(28, 8))
            ax.text(0.5, 0.5, f'Visualization failed:\n{str(e)[:100]}...', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.axis('off')
            return fig

    return multi_year_evolution_visualization
