from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import myTools as mts

dataDir='/home/users/mireiffe/Documents/Python/Pose2Seg/downloads/coco2017/validation/'
dataType='val2017'
annFile=f'/home/users/mireiffe/Documents/Python/Pose2Seg/downloads/coco2017/raw/instances_{dataType}.json'

coco=COCO(annFile)
annFile =f'/home/users/mireiffe/Documents/Python/Pose2Seg/downloads/coco2017/raw/person_keypoints_{dataType}.json'
coco_kps=COCO(annFile)

def genMask(imgIds):
    catIds = coco.getCatIds(catNms=['All'])
    # imgIds = 39769
    img = coco.loadImgs(imgIds)[0]
    I = plt.imread(dataDir + 'data/' + img['file_name'])

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    mask = coco.annToMask(anns[0])
    for i in range(len(anns)):
        mask += coco.annToMask(anns[i])
    mts.saveFile(mask, f'{dataDir}mask/{imgIds:012d}.pck')