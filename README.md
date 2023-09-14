# paper
多标签多分类
将数据data解压会后，放入代码中
运行下面代码
python train.py --arch efficientnet-b0  --pretrained --num-classes 11 --epochs 100 -b 1 -j 2 --output output --val-csv /home/tr/paper/data/train_label_hot.csv --vdata data --csv /home/tr/paper/data/val_label_hot.csv  data
