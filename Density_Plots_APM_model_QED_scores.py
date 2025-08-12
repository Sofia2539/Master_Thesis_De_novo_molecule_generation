#Load dependencies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import argparse


# Function to determine label based on the presence of "APM" in the filename
def extract_label(filename):
    if "APM" in filename:
        return 'With APM'
    else:
        return 'Without APM'

def main(file_paths):
    # Load data into pandas dataframes
    dataframes = [pd.read_csv(file) for file in file_paths]

    # Set the style of the plots
    sns.set(style='whitegrid')

    # Define a color palette
    palette = sns.color_palette("husl", len(dataframes))

    plt.figure(figsize=(12, 8))

    # Create the density plots
    for i, (file, df) in enumerate(zip(file_paths, dataframes)):
        # Determine the label
        label = extract_label(file)
        # Plot the density plot for the current dataframe with a specific color
        sns.kdeplot(df['QED_Score'], label=label, shade=True, color=palette[i])

    # Add labels and legend
    plt.xlabel('QED_Score')
    plt.ylabel('Density')
    plt.title('Density Plot – QED Scores - APM ')
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate density plots for QED_Score from multiple CSV files.')
    parser.add_argument('file_paths', metavar='F', type=str, nargs='+', help='CSV file paths')
    args = parser.parse_args()

    # Run the main function with the provided file paths
    main(args.file_paths)
