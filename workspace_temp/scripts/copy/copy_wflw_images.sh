# Copy images from ~/Download and untar
FILE=$1
TARGET_DIR=./datasets/$FILE
mkdir -p $TARGET_DIR
cp ~/Downloads/${FILE}_images.tar.gz $TARGET_DIR/.
TAR_FILE_ANNO=$TARGET_DIR/${FILE}_images.tar.gz
tar -zxvf $TAR_FILE_ANNO -C $TARGET_DIR
rm $TAR_FILE_ANNO

