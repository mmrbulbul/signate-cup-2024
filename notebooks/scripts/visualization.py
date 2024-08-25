import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def dist_visualization(X_train, X_test, cont_cols=[]):
    # Calculate the number of rows needed for the subplots
    num_rows = (len(cont_cols) + 2) // 3

    X = pd.concat([X_train, X_test], axis=0)
    # Create subplots for each continuous column
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows*5))

    # Loop through each continuous column and plot the histograms
    for i, col in enumerate(cont_cols):
        # Determine the range of values to plot
        max_val = max(X_train[col].max(), X_test[col].max(), X[col].max())
        min_val = min(X_train[col].min(), X_test[col].min(), X[col].min())
        range_val = max_val - min_val
        
        # Determine the bin size and number of bins
        bin_size = range_val / 20
        num_bins_train = round(range_val / bin_size)
        num_bins_test = round(range_val / bin_size)
        num_bins_original = round(range_val / bin_size)
        
        # Calculate the subplot position
        row = i // 3
        col_pos = i % 3
        
        # Plot the histograms
        sns.histplot(X_train[col], ax=axs[row][col_pos], color='orange', kde=True, label='Train', bins=num_bins_train)
        sns.histplot(X_test[col], ax=axs[row][col_pos], color='green', kde=True, label='Test', bins=num_bins_test)
        sns.histplot(X[col], ax=axs[row][col_pos], color='blue', kde=True, label='Original', bins=num_bins_original)
        axs[row][col_pos].set_title(col)
        axs[row][col_pos].set_xlabel('Value')
        axs[row][col_pos].set_ylabel('Frequency')
        axs[row][col_pos].legend()


    # Remove any empty subplots
    if len(cont_cols) % 3 != 0:
        for col_pos in range(len(cont_cols) % 3, 3):
            axs[-1][col_pos].remove()

    plt.tight_layout()
    plt.show()
    
    
    
def get_corr_plot(corr, title="correlations"):
    fig, ax = plt.subplots(figsize=(10,5))
    plt.title(title, fontsize=14)

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    np.fill_diagonal(mask, False)
    
    # Generate the heatmap including the mask
    heatmap = sns.heatmap(corr,
                        annot=True,
                        annot_kws={"fontsize": 10},
                        fmt='.2f',
                        linewidths=0.5,
                        cmap='RdBu',
                        mask=mask, # the mask has been included here
                        ax=ax)

    # Display our plot
    plt.show()