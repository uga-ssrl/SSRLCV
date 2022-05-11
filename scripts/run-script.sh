#!/bin/bash

export j=$(sbatch scripts/sub-$1.sh | awk '{print $4}')

echo "PENDING ${j}"
while [ $(sacct -j $j -X --format='State' | tail -n1 | awk '{print $1}') == '----------' ]; do :; done
while [ $(sacct -j $j -X --format='State' | tail -n1 | awk '{print $1}') == 'PENDING' ]; do :; done

echo "RUNNING"
unset count
while IFS= read -r line1 || IFS= read -r line2 <&3 || [ $(sacct -j $j -X --format='State' | tail -n1 | awk '{print $1}') == 'RUNNING' ] || let "count=count+1" && [ "$count" -lt "10" ] && sleep 1; do
    if [ ${#line1} != 0 ]; then
        printf '\033[0;34mSTDOUT:\033[0m %s\n' "${line1}";
    elif [ ${#line2} != 0 ]; then
        printf '\033[0;31mSTDERR:\033[0m %s\n' "${line2}";
    fi;
done < log/$1.$j.out 3< log/$1.$j.err

if [ $(sacct -j $j -X --format='State' | tail -n1 | awk '{print $1}') != 'COMPLETED' ]; then
    echo "$(sacct -j $j -X --format='State' | tail -n1 | awk '{print $1}')";
    exit 1;
fi;

echo "DONE"
