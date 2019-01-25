# only export path of current package

ENHAN_PATH=$PWD

for dir in bin utils; do
  [ -d $PWD/$dir ] && ENHAN_PATH="$ENHAN_PATH:$PWD/$dir"
done

export PATH=$ENHAN_PATH:$PATH