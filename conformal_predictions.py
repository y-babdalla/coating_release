import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(["science", "no-latex"])
# Load the Excel file
file_path = 'data/test_predictions.xlsx'
data = pd.read_excel(file_path, sheet_name='conformal')
actual_data = pd.read_excel(file_path, sheet_name='full')
actual_data.rename(columns={2: 'Time_2h', 8: 'Time_8h', 24: 'Time_24h'}, inplace=True)


# Column names
hours = [2, 8, 24]
adjusted_time_points = list(range(2, 27, 2))
lower_bound_columns = ['2_lower', '8_lower', '24_lower']
upper_bound_columns = ['2_upper', '8_upper', '24_upper']

# Extracting columns for plotting
values = data[hours]
lower_bounds = data[lower_bound_columns]
upper_bounds = data[upper_bound_columns]


def plot_combined_release(data, predicted_data):
    unique_coatings = data['coating'].unique()
    unique_media = data['medium'].unique()
    n_coatings = len(unique_coatings)
    n_media = len(unique_media)
    lower_bound_columns = ['2_lower', '8_lower', '24_lower']
    upper_bound_columns = ['2_upper', '8_upper', '24_upper']
    plt.rcParams.update({'font.size': 18})
    fig, axs = plt.subplots(n_media, n_coatings, figsize=(20, 5 * n_media), sharex=True, sharey=True)
    if n_media == 1:
        axs = [axs]
    for i, medium in enumerate(unique_media):
        for j, coating in enumerate(unique_coatings):
            ax = axs[i][j] if n_media > 1 else axs[j]
            # Plotting actual values
            subset_actual = data[(data['coating'] == coating) & (data['medium'] == medium)]
            means = [subset_actual['Time_2h'].mean(), subset_actual['Time_8h'].mean(), subset_actual['Time_24h'].mean()]
            stds = [subset_actual['Time_2h'].std(), subset_actual['Time_8h'].std(), subset_actual['Time_24h'].std()]
            ci = 1.96 * np.array(stds) / np.sqrt(len(subset_actual))  # 90% confidence interval
            ax.plot(hours, means, label='Actual', color='blue')
            ax.fill_between(hours, np.array(means) - ci, np.array(means) + ci, color='blue', alpha=0.2)

            # Plotting predicted values and confidence intervals
            subset_predicted = predicted_data[(predicted_data['coating'] == coating) & (predicted_data['medium'] == medium)]
            if len(subset_predicted) > 0:
                values = subset_predicted[hours].iloc[0]
                lower_bounds = subset_predicted[lower_bound_columns].iloc[0]
                upper_bounds = subset_predicted[upper_bound_columns].iloc[0]
                ax.plot(hours, values*100, label='Predicted', color='green')
                ax.fill_between(hours, lower_bounds*100, upper_bounds*100, alpha=0.3, color='green')

            font = 18
            ax.set_title(f"{coating} - {medium}", fontsize=font)
            ax.set_xlabel('Time (hours)', fontsize=font)
            if coating == "Aloe vera extract":
                ax.set_ylabel('5-ASA Drug Release (%)', fontsize=font)
            plt.ylim(-5, 100)
            if i == 0 and coating == "Acacia gum":
                ax.legend()
            ax.set_xticks(adjusted_time_points)
            ax.grid(True)

    # plt.rcParams.update({'font.size': 22})
    plt.tight_layout()
    plt.savefig("new/full_release_2.png", dpi=600)
    plt.show()

# Plotting the combined release
plot_combined_release(actual_data, data)