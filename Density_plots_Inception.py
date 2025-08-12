import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Function to determine label based on the presence of "inception" in the filename
def extract_label(filename):
    if "inception" in filename.lower():
        return 'With Inception'
    else:
        return 'Without Inception'

def main():
    # Check if enough command line arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python script_name.py <file_name1> <file_name2> <plot_title>")
        sys.exit(1)

    # Get the filenames and plot title from command line arguments
    file_name1 = sys.argv[1]
    file_name2 = sys.argv[2]
    plot_title = sys.argv[3]

    # Read the CSV files into DataFrames
    df1 = pd.read_csv(file_name1)
    df2 = pd.read_csv(file_name2)

    # Set a seaborn style
    sns.set(style="whitegrid")

    # Define custom color palette for discrete colors
    custom_palette = ["#1f77b4", "#ff7f0e"]

    # Create a density plot for 'SAScore' using sns.kdeplot with fill
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df1['SAScore'], shade=True, color=custom_palette[0], alpha=0.5, label=extract_label(file_name1))
    sns.kdeplot(df2['SAScore'], shade=True, color=custom_palette[1], alpha=0.5, label=extract_label(file_name2))

    # Set plot title and labels
    plt.title(plot_title, fontsize=15)
    plt.xlabel('SAScore', fontsize=12)
    plt.ylabel('Density', fontsize=12)

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    plt.legend()

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()
