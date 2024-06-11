#!/bin/bash

# Name of the text file
filename="af_links.txt"

# Read the file line by line
while read -r line
do
    # Perform wget for each line
    wget_output=$(wget -nc "$line")
    if [ $? -ne 0 ]; then
        echo $line >> fail_af_download.out
done < "$filename"







