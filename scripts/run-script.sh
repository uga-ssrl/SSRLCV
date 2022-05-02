#!/bin/bash

export j=$(sbatch $1 | awk '{print $4}')

echo "PENDING ${j}"
while [ $(sacct -j $j -X | tail -n1 | awk '{print $6}') == '----------' ]; do :; done
while [ $(sacct -j $j -X | tail -n1 | awk '{print $6}') == 'PENDING' ]; do :; done

echo "RUNNING"
while IFS= read -r line1 || IFS= read -r line2 <&3 || [ $(sacct -j $j -X | tail -n1 | awk '{print $6}') == 'RUNNING' ]; do
    if [ ${#line1} != 0 ]; then
        printf '\033[0;34mSTDOUT:\033[0m %s\n' "${line1}";
    elif [ ${#line2} != 0 ]; then
        printf '\033[0;31mSTDERR:\033[0m %s\n' "${line2}";
    fi;
done < log/build.$j.out 3< log/build.$j.err

if [ $(sacct -j $j -X | tail -n1 | awk '{print $6}') != 'COMPLETED' ]; then
    echo "$(sacct -j $j -X | tail -n1 | awk '{print $6}')";
    exit 1;
fi;

echo "DONE"
