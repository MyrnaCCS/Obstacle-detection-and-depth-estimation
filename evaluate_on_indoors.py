from re import T
import cv2
import numpy as np
import os
from glob import glob

from config import get_config
import lib.EvaluationUtils as EvaluationUtils
from lib.Evaluators import JMOD2Stats
from models.JMOD2 import JMOD2
from models.ODDEG import ODDEG
from tracking.Sort import Sort
from tracking.SortUtils import show_trackers


def preprocess_data(rgb, gt, grd, width_patch=230, height_patch=144, selection_patch_and_resize=False):
    if selection_patch_and_resize:
        i=int((160-height_patch)/2)
        j=int((256-width_patch)/2)
        
        rgb = rgb[i:i+height_patch, j:j+width_patch]
        rgb = cv2.resize(rgb, (256, 160), cv2.INTER_LINEAR)
        
        gt = gt[i:i+height_patch, j:j+width_patch]
        gt = cv2.resize(gt, (256, 160), cv2.INTER_LINEAR)

        grd = grd[i:i+height_patch, j:j+width_patch]
        grd = cv2.resize(grd, (256, 160), cv2.INTER_LINEAR)
    
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    rgb = np.expand_dims(rgb, 0)
    
    gt = np.asarray(gt, dtype=np.float32) / 255.0
    gt *= 7.0

    grd = np.asarray(grd, dtype=np.float32) / 255.0
    
    return rgb, gt, grd


config, unparsed = get_config()

showImages = False

model = JMOD2(config)

model.load_model()

jmod2_stats = JMOD2Stats('jmod2', compute_depth_branch_stats_on_obs=True, iou_thresh=0.5)


for test_path in config.data_test_path:
    if config.tracking:
        sort_tracker = Sort(max_age=config.max_age, min_hits=config.min_hits, iou_threshold=config.iou_threshold)
    
    depth_gt_paths = sorted(glob(os.path.join(config.dataset_path, test_path, 'depth', '*' + '.png')))
    
    rgb_paths = sorted(glob(os.path.join(config.dataset_path, test_path, 'rgb', '*' + '.png')))
    
    grd_paths = sorted(glob(os.path.join(config.dataset_path, test_path, 'ground', '*' + '.png')))
    
    labels_paths = sorted(glob(os.path.join(config.dataset_path, test_path, 'obstacles_3m', '*' + '.txt')))
    
    idx = 0

    for gt_path, rgb_path, grd_path, label_path in zip(depth_gt_paths, rgb_paths, grd_paths, labels_paths):
        rgb_raw = cv2.imread(rgb_path)
        gt = cv2.imread(gt_path, 0)
        gt_obs = EvaluationUtils.get_obstacles_from_file(label_path)
        gt_grd = cv2.imread(grd_path, 0)

        #Normalize input between 0 and 1
        rgb, gt, gt_grd = preprocess_data(rgb_raw, gt, gt_grd, width_patch=128, height_patch=80, selection_patch_and_resize=False)

        #Forward pass to the net
        results = model.run(rgb, gt_grd)
        
        #Add tracking
        if config.tracking:
            trackers, results[1] = sort_tracker.update(results[1])
            for obstacle in results[1]:
                obstacle.compute_depth_stats_from_estimation(results[0])
        
        jmod2_stats.run(results, [gt, gt_obs])
        
        if showImages:
            if results[1] is not None:
                EvaluationUtils.show_detections(rgb_raw, results[1], gt_obs, save_dir=os.path.join('test', test_path), file_name=str(idx).zfill(5)+'.png', sleep_for=10)
            if results[0] is not None:
                EvaluationUtils.show_depth(rgb_raw, results[0], gt, save_dir=os.path.join('test', test_path), file_name=str(idx).zfill(5)+'.png', max_depth=7.5, sleep_for=10)
            if config.tracking:
                show_trackers(rgb_raw, trackers, save_dir=os.path.join('test', test_path), file_name=str(idx).zfill(5)+'.png', sleep_for=10)
        
        idx += 1