# cd current directory
cd "$(dirname "$0")"
echo "Current working directory: $(pwd)"

mkdir data

wget https://s3.amazonaws.com/nist-srd/SD19/by_class.zip -O data/by_class.zip

unzip -q data/by_class.zip -d data/

rm data/by_class.zip

python group_by_writer.py

python converter.py