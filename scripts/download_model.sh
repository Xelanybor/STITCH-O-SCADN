DIR='./checkpoints'
URL='https://drive.google.com/uc?export=download&id=1IrlFQGTpdQYdPeZIEgGUaSFpbYtNpekA'

echo "Downloading pre-trained models..."
mkdir -p $DIR
FILE="$(curl -sc /tmp/gcokie "${URL}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
curl -Lb /tmp/gcokie "${URL}&confirm=$(awk '/_warning_/ {print $NF}' /tmp/gcokie)" -o "$DIR/models.zip"

# echo "Extracting pre-trained models..."
# cd $DIR
# unzip models.zip
# rm models.zip

echo "Download success."