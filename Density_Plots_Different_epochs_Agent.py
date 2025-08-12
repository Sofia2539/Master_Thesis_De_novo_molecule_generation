#Load dependencies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import argparse

# Function to extract epoch information from the filename
def extract_epoch(filename):
    # Use regular expression to find the pattern
    match = re.search(r'agent_(\d+)ep_SAScore.csv', filename)
    if match:
        # Return the captured group which is the epoch number
        return match.group(1)
    else:
        # Return 'Unknown' if the pattern is not found
        return 'Unknown'

def main(file_paths):
    # Load data into pandas dataframes
    dataframes = [pd.read_csv(file) for file in file_paths]

    # Set the style of the plots
    sns.set(style='whitegrid')

    # Define a color palette
    palette = sns.color_palette("husl", len(dataframes))

    plt.figure(figsize=(10, 6))

    # Create the density plots
    for i, (file, df) in enumerate(zip(file_paths, dataframes)):
        # Extract the epoch number from the filename
        epoch = extract_epoch(file)
        # Create the label using the extracted epoch number
        label = f'Epoch - {epoch}'
        # Plot the density plot for the current dataframe with a specific color
        sns.kdeplot(df['SAScore'], label=label, shade=True, color=palette[i])

    # Add labels and legend
    plt.xlabel('SAScore')
    plt.ylabel('Density')
    plt.title('Density Plot - SAScores - Different Epochs in Agent ')
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate density plots for SAScores from multiple CSV files.')
    parser.add_argument('file_paths', metavar='F', type=str, nargs='+', help='CSV file paths')
    args = parser.parse_args()

    # Run the main function with the provided file paths
    main(args.file_paths)
