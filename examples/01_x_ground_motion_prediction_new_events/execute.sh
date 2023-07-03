#!/bin/bash
dir=events

# inotifywait -m "$dir" -e create --format '%w%f' |
#     while IFS=' ' read -r fname
#     do
#         echo $fname
#         [ -f "$fname" ] && chmod +x "$fname"
#     done

inotifywait -r -m $dir -e create |
    while read directory action file; do
        if [[ "$file" =~ .*xml$ ]]; then # Does the file end with .xml?
            echo "$directory$file"
            echo "xml file" # If so, do your thing here!
            python3.8 event_ground_motion_map_generation.py "$directory$file"
        fi
    done
