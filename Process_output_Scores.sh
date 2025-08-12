# Store the output file name provided as an argument
output_file="$1"

# Extract the part after "output_"
output_dir=$(dirname "$output_file")
file_name=$(basename "$output_file")
prefix="output_"
suffix=".txt"
file_name_no_ext="${file_name#$prefix}"
file_name_no_ext="${file_name_no_ext%$suffix}"

# Check if the output file exists
if [ ! -f "$output_file" ]; then
    echo "Error: Output file '$output_file' not found."
    exit 1
fi

#Create CSV file with headers in the same directory as the output file
output_csv="${output_dir}/scores_smiles_filtered_${file_name_no_ext}.csv"
echo "Scores,SMILES" > "$output_csv"

# Extract relevant data from output file and append to CSV
grep Prior "$output_file" -A10 | grep -v "\-\-" | grep -v SMILES | awk '{print $4","$5}' >> "$output_csv"

echo "Process completed. Output written to $output_csv"
