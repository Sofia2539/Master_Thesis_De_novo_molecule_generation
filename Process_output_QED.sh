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

# Create CSV file with headers in the same directory as the output file
output_csv="${output_dir}/scores_smiles_filtered_QED_${file_name_no_ext}.csv"
echo "QED_Score,SMILES" > "$output_csv"

# Extract SMILES and Scores from the first section
grep -A10 "Prior" "$output_file" | grep -v "\-\-" | grep -v "SMILES" | awk '{print $5}' > tmp_smiles.csv

# Extract QED Scores from the second section
grep -A10 "Matching substructure" "$output_file" | grep -v "Matching substructure" | awk '{print $3}' > tmp_qed.csv

# Combine the two files into the final CSV
paste -d, tmp_qed.csv tmp_smiles.csv | grep -v "^," >> "$output_csv"

# Cleanup temporary files
rm tmp_smiles.csv tmp_qed.csv

echo "Process completed. Output written to $output_csv"
