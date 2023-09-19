# paper
多标签多分类
将数据data解压会后，放入代码中
运行下面代码跑出参数：
python train.py --arch efficientnet-b4  --pretrained --num-classes 11 --epochs 100 -b 10 -j 2 --output output --val-csv /home/tr/paper/data/train_label_hot.csv --vdata data --csv /home/tr/paper/data/val_label_hot.csv  data
运行下面代码跑出得到可视化结果：
python inference.py --arch efficientnet-b4  --num-classes 11 --gpu 1 --num-visualize 9 -b 10--model /home/tr/paper/output/model_best.pth.tar  /home/tr/paper/predict
