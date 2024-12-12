dirs=$(find ./outputs/ -mindepth 1 -maxdepth 1 -type d | paste -sd ',' -)
python get_result.py $dirs