# IMPROVING PSEUDO-LABELS SELECTION USING DOMAIN PRIORS FOR SEMI-SUPERVISED DETECTION IN CAPSULE ENDOSCOPY
This is an official implementation for ICIP2025 paper ["IMPROVING PSEUDO-LABELS SELECTION USING DOMAIN PRIORS FOR SEMI-SUPERVISED DETECTION IN CAPSULE ENDOSCOPY"](https://xxxx.pdf). 

<!-- by [Peng Mi](), [Jianghang Lin](https://github.com/HunterJ-Lin), [Yiyi Zhou](), [Yunhang Shen](), [Gen Luo](), [Xiaoshuai Sun](), [Liujuan Cao](), [Rongrong Fu](), [Qiang Xu](), [Rongrong Ji](). -->
<!-- Conference on Computer Vision and Pattern Recognition (CVPR) 2022 Paper.</br> -->

## Introduction

The overall of our **Bloc Diagram**. 
<p align="center">
<img src="blockDiagram.png">
</p>

Our Domain-Tailored Augmentations **DTA**. 
<p align="center">
<img src="DTA.png">
</p>

**Better pseudo-labels than Active Teacher**. 
<p align="center">
<img src="compareWithAT.png">
</p>

## Important notes
- Update "NUM_CLASSES: 8" in 'configs/coco/Base-RCNN-FPN.yaml' to fit the number of class of your dataset.
- You can also update other parameters in the configuration file 'configs/coco/Base-RCNN-FPN.yaml' or 'configs/Base-RCNN-FPN.yaml'
- Update "metadata" variable in 'tools/train_net.py' and 'tools/train_net_sets.py' according to the classes names of your dataset
  
## Installation

- Install **detectron2** following the [instructions](https://detectron2.readthedocs.io/tutorials/install.html).

## Important notes
- Update "NUM_CLASSES: 8" in 'configs/coco/Base-RCNN-FPN.yaml' to fit the number of class of your dataset.
- You can also update other parameters in the configuration file 'configs/coco/Base-RCNN-FPN.yaml' or 'configs/Base-RCNN-FPN.yaml'
- Update "metadata" variable in 'tools/train_net.py' and 'tools/train_net_sets.py' according to the classes names of your dataset
  
## Dataset Preparation

### Custom dataset
Download our folder containing the two datasets (Kvasir Capsule and SEE-AI) we used here :
After downloading extract 
Then, copy the concerned dataset content in the folder "datasets". For testing we provide 9 subsets. The .json files contains the annotations.
The expected files structure is :
### Dataset File structure:
```
datasets/
 coco/
  annotations/
     coco_training.json
     set1_coco_validation.json
     .....
     set9_coco_validation.json
  train/
     images.jpg 
     .....
  val/
     images.jpg
     .....
  valsets/
    set1/
      images.jpg
     .....
    set9/
      images.jpg
```
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`, and how to add new datasets to them.

## Training (34.79% label data for example)
### Step 0、Generate 34.79% label data partition
```
python tools/generate_random_data_partition.py --random_file dataseed/COCO_supervision.txt --random_percent 34.79
```

### Step 1、Train a pick model on 34.79% random data
```
mkdir temp
mkdir temp/coco
mkdir results
mkdir results/coco
mkdir dataseed/coco_pick

python tools/train_net.py \
      --num-gpus 1 \
      --config configs/coco/faster_rcnn_R_50_FPN_sup35_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16  OUTPUT_DIR output/coco/result_initial_faster_rcnn_R_50_FPN_sup35_run1_16bs

```

### Step 2、Use the trained model from step 1 to get the indicator file of the dataset
```
python tools/inference_for_active_pick_TTA_AS.py\
    --static_file temp/coco/static_by_random.json \
    --model_weights output/coco/result_initial_faster_rcnn_R_50_FPN_sup35_run1_16bs/model_best.pth \
    --config configs/coco/faster_rcnn_R_50_FPN_sup35_run1.yaml \
    

python tools/TTA_AS_active_pick_evaluation.py \
    --static_file temp/coco/static_by_random.json/static_by_random.json \
    --indicator_file results/coco/random_maxnorm
    
```

### Step 3、Use the indictor file from step 2 to generate pick data and merge random data
```
python tools/TTA_AS_generate_pick_merge_random_data_partition.py \
    --random_file dataseed/COCO_supervision.txt \
    --random_percent 34.79\
    --indicator_file results/coco/random_maxnorm.txt \
    --pick_percent 35.21\
    --save_file dataseed/coco_pick/pick_maxnorm+random.txt\
    --static_file temp/coco/static_by_random.json/static_by_random.json \
    --reverse True \
```

### Step 4、Train a model from scratch using the 10% data partition from step 3
```
python tools/TTA_SA_GT_train_net.py \
      --num-gpus 1 \
      --config configs/coco/faster_rcnn_R_50_FPN_sup70_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 OUTPUT_DIR output/coco/result_final_faster_rcnn_R_50_FPN_sup70_run1_16bs DATALOADER.RANDOM_DATA_SEED_PATH dataseed/coco_pick/pick_maxnorm+random.txt   
  
```

## Evaluation
```
python tools/train_net.py \
      --eval-only \
      --num-gpus 1 \
      --config configs/coco/faster_rcnn_R_50_FPN_sup70_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16  MODEL.WEIGHTS output/coco/result_final_faster_rcnn_R_50_FPN_sup70_run1_16bs/model_best.pth OUTPUT_DIR output/results

```
## Evaluation for each set :
Note: You should update the "register_coco_instances()" function with the corresponding directions for the images and the annotations .json file of the concerned 'set' in the file  'train_net_sets.py' 
```
python tools/train_net_sets.py \
      --eval-only \
      --num-gpus 1 \
      --config configs/coco/faster_rcnn_R_50_FPN_sup70_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16  MODEL.WEIGHTS output/coco/result_final_faster_rcnn_R_50_FPN_sup70_run1_16bs/model_best.pth  OUTPUT_DIR output/set_name

```


## Results
- The results on **different datasets** is shown as below:

\begin{table*}[htp]
\vspace{-5mm}
\centering
\caption{Models performances comparison on the SEE-AI Dataset. Ground Truth (GT), Pseudo-Labeling (PL). Highest value for each of the metrics are in bold.}
\small
\begin{tabular}{lcccccccccc}
\hline
\hline
 &GT&AS&Annotation& AP & AP50 & AP75 & APs & APm & APl \\ \hline  
 & & & & \multicolumn{6}{c}{SEE-AI Dataset}\\
 \hline
Faster RCNN \cite{ren2015faster} & 100\%& -- & -- &  33.60& 61.30 & 33.20& 6.00 & 15.80&  36.80\\ %\hline
Unbiased Teacher \cite{liu2021unbiased}& 34.79\%& --& 100\% PL& 30.63 &  55.92&28.81& 16.79& 17.37&33.72 \\ %\hline
Active  Teacher \cite{mi2022active}& 34.79\%& 35.21\% & GT & 38.65&69.91&\textbf{37.89}&\textbf{19.82}& 26.21& 41.27 \\ %\hline
 DTA + AS (Ours)& 34.79\%& 35.21\% & GT &\textbf{40.09}&\textbf{73.00}&37.44&15.98&\textbf{26.89}&\textbf{42.34}\\ 
\hline
& & & & \multicolumn{6}{c}{Kvasir-Capsule}\\
\hline
Faster RCNN \cite{ren2015faster} & 100\%& -- & -- & 57.30&81.70& 65.70&23.70& 49.60&61.30 \\ %\hline
Unbiased Teacher \cite{liu2021unbiased}& 34.79\%& --& 100\% PL&  61.35& 91.28& 69.32&39.30&  55.76&  65.20 \\ %\hline
Active Teacher \cite{mi2022active}& 34.79\%& 35.21\% & GT & 63.63& \textbf{94.02}&70.95&  35.33&58.84 &  69.62 \\ %\hline
 DTA + AS (Ours)& 34.79\%& 35.21\% & GT &  \textbf{64.17}  &   93.25 &\textbf{74.32}& \textbf{43.45}& \textbf{58.89} &\textbf{70.73}\\ \hline
\hline
\end{tabular}
\label{tab1}
\end{table*}

<table border="0" width="800">
<tr>
	<td width="25%" align="center"> <b> Models </b> </td>
	<td width="25%" align="center"> <b> Datasets </b> </td>
	<td width="15%" align="center"> <b> Labels </b> </td>
	<td width="25%" align="center"> <b> Supervision(mAP) </b> </td>
	<td width="25%" align="center"> <b> Ours(mAP) </b> </td>
</tr>
	
<tr>
	<td width="25" align="center" rowspan="9"> Res50-FPN</td>
	<td width="25%" align="center" rowspan="5"> COCO </td>
	<td width="15%" align="center"> 1% </td>
	<td width="25%" align="center" rowspan="5"> 37.63 </td>
	<td width="25%" align="center"> 22.20 </td>
</tr>
<tr>
	<td width="15%" align="center"> 2% </td>
	<td width="25%" align="center"> 24.99 </td>
</tr>
<tr>
	<td width="15%" align="center"> 5% </td>
	<td width="25%" align="center"> 30.07 </td>
</tr>
<tr>
	<td width="15%" align="center"> 10% </td>
	<td width="25%" align="center"> 32.58 </td>
</tr>
<tr>
	<td width="15%" align="center"> 20% </td>
	<td width="25%" align="center"> 35.49 </td>
</tr>

<tr>
	<td width="25%" align="center" rowspan="3"> VOC07+12 </td>
	<td width="15%" align="center"> 5% </td>
	<td width="25%" align="center" rowspan="3"> 48.62 </td>
	<td width="25%" align="center"> 41.85 </td>
</tr>
<tr>
	<td width="15%" align="center"> 10% </td>
	<td width="25%" align="center"> 46.77 </td>
</tr>
<tr>
	<td width="15%" align="center"> 15% </td>
	<td width="25%" align="center"> 49.73 </td>
</tr>
	
<tr>
	<td width="25%" align="center"> SODA </td>
	<td width="15%" align="center"> 10% </td>
	<td width="25%" align="center"> 34.52 </td>
	<td width="25%" align="center"> 33.32 </td>
</tr>
</table>
	
## Citing Active Teacher

If you find Active Teacher useful in your research, please consider citing:

```
@InProceedings{ActiveTeacher_2022_CVPR,
	author = {Mi, Peng and Lin, Jianghang and Zhou, Yiyi and Shen, Yunhang and Luo, Gen and Sun, Xiaoshuai and Cao, Liujuan and Fu, Rongrong and Xu, Qiang and Ji, Rongrong},
	title = {Active Teacher for Semi-Supervised Object Detection},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	year = {2022},
}   
```

## License

Active Teacher is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
-   [STAC](https://github.com/google-research/ssl_detection)
-   [unbiased teacher](https://github.com/facebookresearch/unbiased-teacher)
-   [detectron2](https://github.com/facebookresearch/detectron2)
