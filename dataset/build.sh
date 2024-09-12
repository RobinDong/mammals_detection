#!/bin/bash
set -e

# prepare coco
rm -rf coco
cp -r ../../yolo_dataset/coco_backup coco

# prepare basic

BARM=/media/data2/tetrapod/yolotraining

mkdir -p yolotrain
cp -r ${BARM}/V1.2.extra.20240523/A yolotrain/amphibia
cp -r ${BARM}/V1.2.extra.20240523/B yolotrain/bird
cp -r ${BARM}/fish.v1.20240525 yolotrain/fish
cp -r ${BARM}/V1.2.extra.20240523/M yolotrain/mammal
cp -r ${BARM}/V1.2.extra.20240523/R yolotrain/reptile
cp ${BARM}/amphibia.v1.1/* yolotrain/amphibia/
cp ${BARM}/birds.video.20240727/* yolotrain/bird/
cp ${BARM}/mammal.dataset.v1.1/* yolotrain/mammal/
cp ${BARM}/reptile.v1/* yolotrain/reptile/
cp -r ${BARM}/extra.20240606-18fishes-461pangolins yolotrain/extra
cp -r ${BARM}/shrimpandcrab.V1.20240724 yolotrain/shrimp

DIRS="yolotrain/amphibia yolotrain/bird yolotrain/mammal yolotrain/reptile yolotrain/fish yolotrain/extra yolotrain/shrimp"
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
  # shrimp(5) -> 205
  # others(6) -> 206
  sed s/^0\ /200\ / ${file} | sed s/^1\ /201\ / | sed s/^2\ /202\ / | sed s/^3\ /203\ / | sed s/^4\ /204\ / | sed s/^5\ /205\ / | sed s/^6\ /206\ / > coco/labels/val2017/${name}
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
  # shrimp(5) -> 205
  # others(6) -> 206
  sed s/^0\ /200\ / ${file} | sed s/^1\ /201\ / | sed s/^2\ /202\ / | sed s/^3\ /203\ / | sed s/^4\ /204\ / | sed s/^5\ /205\ / | sed s/^6\ /206\ / > coco/labels/train2017/${name}
  rm ${file}
done

# build index
cd coco
find ./images -type f|grep train > train2017.txt
find ./images -type f|grep val > val2017.txt
cd -

python3 clear_categories.py

echo "Finished!"
