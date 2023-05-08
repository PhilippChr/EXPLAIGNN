#!/usr/bin/bash 

# initialize root dir
CONVINSE_ROOT=$(pwd)

## check argument length
if [[ $# -lt 1 ]]
then
	echo "Error: Invalid number of options: Please specify the data which should be downloaded."
	echo "Usage: bash scripts/download.sh <DATA_FOR_DOWNLOAD>"
	exit 0
fi

case "$1" in
"explaignn")
    echo "Downloading EXPLAIGNN data..."
    wget http://qa.mpi-inf.mpg.de/explaignn/convmix_data/explaignn.zip
    mkdir -p _data/convmix/
    unzip explaignn.zip -d _data/convmix/
    rm explaignn.zip
    echo "Successfully downloaded EXPLAIGNN data!"
    ;;
"convmix")
    echo "Downloading ConvMix dataset..."
    mkdir -p _benchmarks/convmix
    cd _benchmarks/convmix
    wget http://qa.mpi-inf.mpg.de/explaignn/convmix/train_set.zip
    unzip train_set.zip
    rm train_set.zip
    wget http://qa.mpi-inf.mpg.de/explaignn/convmix/dev_set.zip
    unzip dev_set.zip
    rm dev_set.zip
    wget http://qa.mpi-inf.mpg.de/explaignn/convmix/test_set.zip
    unzip test_set.zip
    rm test_set.zip
    echo "Successfully downloaded ConvMix dataset!"
    ;;
"wikipedia")
    echo "Downloading Wikipedia dump..."
    wget http://qa.mpi-inf.mpg.de/explaignn/convmix_data/wikipedia.zip
    mkdir -p _data/convmix/
    unzip wikipedia.zip -d _data/convmix/
    rm wikipedia.zip
    echo "Successfully downloaded Wikipedia dump!"
    ;;
"data")
    echo "Downloading general repo data..."
    wget http://qa.mpi-inf.mpg.de/explaignn/convmix_data/data.zip
    unzip data.zip -d _data
    rm data.zip
    echo "Successfully downloaded general repo data!"
    ;;
*)
    echo "Error: Invalid specification of the data. Data $1 could not be found."
	exit 0
    ;;
esac
