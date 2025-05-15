
def convert_coco_json_to_csv(filename):
    import pandas as pd
    import json
    
    s = json.load(open(filename, 'r'))
    out_file = filename[:-5] + '.csv'
    out = open(out_file, 'w')
    out.write('id,image,label,x1,y1,x2,y2,area\n')

    all_ids = []
    for im in s['images']:
        all_ids.append(im['id'])

    all_ids_ann = []
    for ann in s['annotations']:
        image_id = ann['image_id']
        List_images=s['images']
        ['file_name']
        
        fileName=""
        
        for elt in List_images:
            #print("---------elt=",elt,"-----------\n")
            if  elt["id"]==image_id:
                fileName=elt['file_name']
                print("---------image_id=",image_id,"-----fileName=",fileName,"-----------\n")
                
        all_ids_ann.append(image_id)
        x1 = ann['bbox'][0]
        x2 = ann['bbox'][0] + ann['bbox'][2]
        y1 = ann['bbox'][1]
        y2 = ann['bbox'][1] + ann['bbox'][3]
        label = ann['category_id']
        area=(x2-x1)*(y2-y1)
        out.write('{},{},{},{},{},{},{},{}\n'.format(image_id,fileName, label, x1, y1, x2, y2, area))

    all_ids = set(all_ids)
    all_ids_ann = set(all_ids_ann)
    no_annotations = list(all_ids - all_ids_ann)
    # Output images without any annotations
    for image_id in no_annotations:
        out.write('{},{},{},{},{},{},{},{}\n'.format(image_id, -1,-1, -1, -1, -1, -1,-1))
    out.close()

    # Sort file by image id
    s1 = pd.read_csv(out_file)
    s1.sort_values('id', inplace=True)
    s1.to_csv(out_file, index=False)
    
    
annot_link_for_training="./datasets/coco/annotations/coco_training.json"

convert_coco_json_to_csv(annot_link_for_training)


