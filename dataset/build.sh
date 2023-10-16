# prepare coco
rm -rf coco
cp -r coco_backup coco

# prepare mammals
rm -rf yolotraing
tar xzf yolotrain.tgz
# valid
for file in `find yolotraing/mammal.dataset.v1/ -name "i_5*.jpg" -type f`; do
  mv ${file} coco/images/val2017/
done
for file in `find yolotraing/mammal.dataset.v1/ -name "i_5*.txt" -type f`; do
  name="`basename ${file}`"
  # birds(0) -> 200
  # mammals(1) -> 201
  sed s/^0\ /200\ / ${file} | sed s/^1\ /201\ / > coco/labels/val2017/${name}
  rm ${file}
done

# train
for file in `find yolotraing/mammal.dataset.v1/ -name "*.jpg" -type f`; do
  mv ${file} coco/images/train2017/
done
for file in `find yolotraing/mammal.dataset.v1/ -name "*.txt" -type f`; do
  name="`basename ${file}`"
  # birds(0) -> 200
  # mammals(1) -> 201
  sed s/^0\ /200\ / ${file} | sed s/^1\ /201\ / > coco/labels/train2017/${name}
  rm ${file}
done

# build index
cd coco
find ./images -type f|grep train > train2017.txt
find ./images -type f|grep val > val2017.txt
cd -

python3 clear_categories.py

cd coco
find ./images -type f|grep train > train2017.txt
find ./images -type f|grep val > val2017.txt
rm -rf train2017.cache val2017.cache
cd -
