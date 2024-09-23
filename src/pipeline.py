# Python libraries
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cvt.common import to_gpu, laplacian_pyramid
from cvt.io import write_pfm
from cvt.geometry import visibility, get_uncovered_mask, edge_mask
from cvt.visualization import visualize_mvs, laplacian_depth_error, laplacian_count, laplacian_uncovered_count, plot_laplacian_matrix

## Custom libraries
from src.evaluation.eval_2d import depth_acc
from src.datasets.BaseDataset import build_dataset

class Pipeline():
    def __init__(self, cfg, config_path, log_path, model_name, training_scenes=None, validation_scenes=None, inference_scene=None):
        self.cfg = cfg
        self.device = self.cfg["device"]
        self.mode = self.cfg["mode"]
        self.inference_scene = inference_scene

        # build the data loaders
        self.build_dataset()

        # set data paths
        self.data_path = os.path.join(self.cfg["data_path"], self.inference_scene[0])
        self.output_path = os.path.join(self.cfg["output_path"], self.inference_scene[0])
        self.depth_path = os.path.join(self.output_path, "depth")
        self.conf_path = os.path.join(self.output_path, "confidence")
        self.rgb_path = os.path.join(self.output_path, "rgb")
        self.reprojection_path = os.path.join(self.output_path, "reprojection")
        self.laplacian_path = os.path.join(self.output_path, "laplacian")
        self.vis_path = os.path.join(self.output_path, "visuals")
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.depth_path, exist_ok=True)
        os.makedirs(self.conf_path, exist_ok=True)
        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.reprojection_path, exist_ok=True)
        os.makedirs(self.laplacian_path, exist_ok=True)
        os.makedirs(self.vis_path, exist_ok=True)
        self.batch_size = 1

    def inference(self):
        with torch.inference_mode():
            self.run(mode="inference", epoch=-1)
        return

    def run(self, mode, epoch):
        data_loader = self.inference_data_loader
        title_suffix = ""

        with tqdm(data_loader, desc=f"MVS-Studio {mode}{title_suffix}", unit="batch") as loader:
            for batch_ind, data in enumerate(loader):
                # compute laplacians

                output = {}
                output["image_laplacian"] = laplacian_pyramid(data["images"][:,0])
                output["depth_laplacian"] = laplacian_pyramid(data["target_depth"])
                output["final_depth"] = torch.clone(data["target_depth"])
                output["confidence"] = torch.ones_like(output["final_depth"])

                # mask out sharp regions
                output["final_depth"] *= torch.where(output["depth_laplacian"] <= 100, 1.0, 0.0)

                # Store network output
                self.save_output(data, output, mode, batch_ind, epoch)
                #visualize_mvs(data, output, batch_ind, self.vis_path, self.cfg["visualization"]["max_depth_error"], mode="gt", epoch=-1)

    def build_dataset(self):
        self.inference_dataset = build_dataset(self.cfg, self.mode, self.inference_scene)
        self.cfg["H"], self.cfg["W"] = self.inference_dataset.H, self.inference_dataset.W
        self.inference_data_loader = DataLoader(self.inference_dataset,
                                     1,
                                     shuffle=False,
                                     num_workers=self.cfg["num_workers"],
                                     pin_memory=True,
                                     drop_last=False)

    def save_output(self, data, output, mode, batch_ind, epoch):
        with torch.set_grad_enabled((torch.is_grad_enabled and not torch.is_inference_mode_enabled)):
            # save confidence map
            conf_map = output["confidence"][0,0].detach().cpu().numpy()
            conf_filename = os.path.join(self.conf_path, f"{batch_ind:08d}.pfm")
            write_pfm(conf_filename, conf_map)
            conf_map = output["confidence"][0,0].detach().cpu().numpy()
            os.makedirs(os.path.join(self.conf_path, "disp"), exist_ok=True)
            conf_filename = os.path.join(self.conf_path, "disp", f"{batch_ind:08d}.png")
            cv2.imwrite(conf_filename, (conf_map * 255))
            # save depth map
            depth_map = output["final_depth"][0,0].detach().cpu().numpy()
            depth_filename = os.path.join(self.depth_path, f"{batch_ind:08d}.pfm")
            write_pfm(depth_filename, depth_map)
            depth_map = (output["final_depth"][0,0].detach().cpu().numpy() - self.cfg["camera"]["near"]) / (self.cfg["camera"]["far"]-self.cfg["camera"]["near"])
            os.makedirs(os.path.join(self.depth_path, "disp"), exist_ok=True)
            depth_filename = os.path.join(self.depth_path, "disp", f"{batch_ind:08d}.png")
            cv2.imwrite(depth_filename, (depth_map * 255))
            # save image
            ref_image = (torch.movedim(data["images"][0,0], (0,1,2), (2,0,1)).detach().cpu().numpy()+0.5) * 255
            img_filename = os.path.join(self.rgb_path, f"{batch_ind:08d}.png")
            cv2.imwrite(img_filename, ref_image[:,:,::-1])
