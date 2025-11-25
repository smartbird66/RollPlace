#!/bin/bash

datasets=(
    "adaptec1" "adaptec2" "adaptec3" "adaptec4" 
    "bigblue1" "bigblue2" "bigblue3" "bigblue4"
)

grid_nums=(
    160 158 108 108
    160 160 233 160
)

total_count=${#grid_nums[@]}

for ((i=0; i<total_count; i++)); do

    current_grid=${grid_nums[$i]}
    current_data=${datasets[$i]}
    
    echo "start processing: $current_data"
    nohup python RollPlace.py --benchmark_folder ispd2005 --dataset $current_data --seed 2005 --max_workers 1 --grid_num $current_grid > ./nohup/${current_data}_2005.log 2>&1 &
    
    sleep 2
done

echo "All commands are now running."