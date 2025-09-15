#!/bin/bash
# Fix imagecas and extract to a single directory


ranges=(
    "1-200"
    "201-400"
    "401-600"
    "601-800"
    "801-1000"
)

output_dir="imagecas_unzipped"

for range in "${ranges[@]}"; do
    echo "$range"
    
    original_filename="${range}.change2zip"
    temp_zip_filename="${range}.zip"
    fixed_zip_filename="${range}_fixed.zip"

    mv "${original_filename}" "${temp_zip_filename}"
    zip -F "${temp_zip_filename}" --out "${fixed_zip_filename}"
    unzip "${fixed_zip_filename}" -d "${output_dir}"
done
