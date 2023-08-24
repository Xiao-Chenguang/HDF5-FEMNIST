#!/bin/bash

while getopts ":d:" opt; do
  case $opt in
    d) digits_only="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

digits_only=${digits_only:-True}
echo "extract digits_only: $digits_only";

# cd current directory
cd "$(dirname "$0")"
echo "Current working directory: $(pwd)"

mkdir data

wget https://s3.amazonaws.com/nist-srd/SD19/by_class.zip -O data/by_class.zip || curl https://s3.amazonaws.com/nist-srd/SD19/by_class.zip -o data/by_class.zip

unzip -q data/by_class.zip -d data/

rm data/by_class.zip

python group_by_writer.py $digits_only

python converter.py $digits_only