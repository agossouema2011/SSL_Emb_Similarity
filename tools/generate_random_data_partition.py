import json
import argparse
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

from detectron2.data.build import (
    get_detection_dataset_dicts,
)




def generate(args):
    datasets = []
    for d in args.datasets.split(','):
        if d == '':
            continue
        datasets.append(d)
    dataset_dicts = get_detection_dataset_dicts(datasets)
    print('len of ditaset:',len(dataset_dicts))
    try:
        with open(args.random_file,'r') as f:
                dic=json.load(f)
    except:
        dic={}

    dic[str(args.random_percent)] = {}
    seeds = [int(i) for i in args.random_seeds.split(',')]
    for i in range(10):
        arr = generate_supervised_seed(
            dataset_dicts,
            args.random_percent,
            seeds[i]
        )
        print(len(arr))
        dic[str(args.random_percent)][str(i)] = arr
    with open(args.random_file,'w') as f:
        f.write(json.dumps(dic))


def generate_supervised_seed(
    dataset_dicts, SupPercent, seed
):
    num_all = len(dataset_dicts)
    num_label = int(SupPercent / 100.0 * num_all)

    arr = range(num_all)
    import random
    random.seed(seed)
    return random.sample(arr,num_label)

if __name__ == "__main__":
    register_coco_instances("my_dataset_train", {}, "datasets/coco/annotations/coco_training.json", "datasets/coco/train") # register the custom dataset
    
    # set name for the different class labels
    metadata= MetadataCatalog.get("my_dataset_train").set(thing_classes = ["Angiectasia","Blood-fresh","Erosion","Erythema","Foreign-Body","Lymphangiectasia","Polyp","Ulcer"]); del metadata.thing_classes
   
    parser = argparse.ArgumentParser(description='Generate random data partitions for supervised set')
    parser.add_argument("--random_file",type=str,default='dataseed/COCO_supervision.txt')
    parser.add_argument("--random_percent",type=float,default=5.0)
    parser.add_argument("--datasets",type=str,default='my_dataset_train,') # degault directory for train set'
    parser.add_argument("--random_seeds",type=str,default="0,1,2,3,4,5,6,7,8,9") # Need to set 10 random number seeds for experiments, divided by ','
    args = parser.parse_args()
    generate(args)
