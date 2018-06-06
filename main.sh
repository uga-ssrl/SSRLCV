#!/usr/bin/env bash

echo "starting pipeline ..."

# attempt to load ARGS
IMG1=$1 # image 1
IMG2=$2 # image 2
CAMF=$3 # camera file

# functions here
print_usage() { echo "USAGE:"; echo "./main.sh path/to/image.png path/to/image.png path/to/cameras.txt";}

# to  make sure we have what we need
check_args(){
  # get path to first image
  if [ -z "$IMG1" ]
  then
        echo "USER ERROR: no first argument"
        print_usage
        exit 1
  fi

  if [ -z "$IMG2" ]
  then
        echo "USER ERROR: no second argument"
        print_usage
        exit 1
  fi

  if [ -z "$CAMF" ]
  then
        echo "USER ERROR: no third argument"
        print_usage
        exit 1
  fi
}

run_sift_cpu() {
  echo "running sift cpu ..."
  rm -f p01.kp
  rm -f p02.kp
  rm -f matches_raw.txt
  rm -f matches.txt
  echo "computing ${IMG1} ..."
  ./bin/sift_cli $IMG1 > p01.kp
  echo "computing ${IMG2} ..."
  ./bin/sift_cli $IMG2 > p02.kp
  echo "computing matches ..."
  ./bin/match_cli p01.kp p02.kp > matches_raw.txt
  echo "converting raw sift output ..."
  python util/io/raw_matches_to_matches.py
}

run_reprojection(){
  echo "running reprojection ..."
  ./bin/2viewreprojection.x $CAMF matches.txt 0
}

clean_up(){
  echo "cleaning ..."
  rm -f p01.kp
  rm -f p02.kp
  rm -f matches_raw.txt
  rm -f matches.txt
}

###############
# ENTRY POINT #
###############

check_args
run_sift_cpu
run_reprojection
# TODO put reconstruction here
clean_up
