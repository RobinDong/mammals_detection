#!/bin/bash
set -e

# prepare coco
rm -rf coco
cp -r ../../yolo_dataset/coco_backup coco

# prepare basic
rm -rf yolotrain
tar xf ../../yolo_dataset/yolotrain.tar

# copy extra into
for pair in "M/ mammal.dataset.v1.1/" "A/ amphibia.v1.1/" "R/ reptile.v1/"; do
  dir=( $pair )
  for file in `find ../../BARM/V1.2.extra.20240523/${dir[0]} -type f`; do
    cp ${file} yolotrain/${dir[1]}
  done
done

cp -r ../../BARM/V1.2.extra.20240523/B yolotrain/bird
cp -r ../../BARM/V1.2.extra.20240523/F yolotrain/fish

DIRS="yolotrain/mammal.dataset.v1.1/ yolotrain/amphibia.v1.1/ yolotrain/reptile.v1/ yolotrain/bird yolotrain/fish"
# valid
for file in `find ${DIRS} -name "*5.jpg" -type f`; do
  mv ${file} coco/images/val2017/
done
for file in `find ${DIRS} -name "*5.txt" -type f`; do
  name="`basename ${file}`"
  # birds(0) -> 200
  # mammals(1) -> 201
  # reptile(2) -> 202
  # amphibia(3) -> 203
  # fish(4) -> 204
  sed s/^0\ /200\ / ${file} | sed s/^1\ /201\ / | sed s/^2\ /202\ / | sed s/^3\ /203\ / | sed s/^4\ /204\ / > coco/labels/val2017/${name}
  rm ${file}
done

# train
for file in `find ${DIRS} -name "*.jpg" -type f`; do
  mv ${file} coco/images/train2017/
done
for file in `find ${DIRS} -name "*.txt" -type f`; do
  name="`basename ${file}`"
  # birds(0) -> 200
  # mammals(1) -> 201
  # reptile(2) -> 202
  # amphibia(3) -> 203
  # fish(4) -> 204
  sed s/^0\ /200\ / ${file} | sed s/^1\ /201\ / | sed s/^2\ /202\ / | sed s/^3\ /203\ / | sed s/^4\ /204\ / > coco/labels/train2017/${name}
  rm ${file}
done

# build index
cd coco
find ./images -type f|grep train > train2017.txt
find ./images -type f|grep val > val2017.txt
cd -

python3 clear_categories.py

#cd coco
#find ./images -type f|grep train > train2017.txt
#find ./images -type f|grep val > val2017.txt
#rm -rf train2017.cache val2017.cache
#cd -
