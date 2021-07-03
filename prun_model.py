import os
import copy
import torch.nn.utils.prune as prune
from dataloader import TrafficSignDataset, Collator
from model.repvgg import create_RepVGG_A0
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt 
import cv2
from utils import cal_acc


class Pruning(object):
    def __init__(self):
        # initialize config
        self.random_seed = 2021
        self.learning_rate = 1e-3
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.valid_every = 2000
        self.print_every = 500
        self.prunning_num_iters = 10
        self.trainning_num_iters = 3000

        # createa dataloader
        self.train_loader, self.val_loader = self.create_data_loader()

        # create model
        self.model = create_RepVGG_A0(num_classes=43, deploy=False)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load('./weights/repvgg.pth.tar', map_location=self.device), strict=False)

        # create loss fucntion and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-09)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.learning_rate, total_steps=self.trainning_num_iters, pct_start=0.1)


    def create_data_loader(self):
        dataset = TrafficSignDataset(image_dir='./Data/myData/', label_file='./Data/labels.csv', target_shape=(32, 32))
        
        # split train and val dataset
        split_ratio = 0.9
        n_train = int(len(dataset) * split_ratio)
        n_val = len(dataset) - n_train
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=Collator(), shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, collate_fn=Collator(), shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

        return train_loader, val_loader

    @staticmethod
    def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):
        num_zeros = 0
        num_elements = 0

        if use_mask == True:
            for buffer_name, buffer in module.named_buffers():
                if "weight_mask" in buffer_name and weight == True:
                    num_zeros += torch.sum(buffer == 0).item()
                    num_elements += buffer.nelement()
                if "bias_mask" in buffer_name and bias == True:
                    num_zeros += torch.sum(buffer == 0).item()
                    num_elements += buffer.nelement()
        else:
            for param_name, param in module.named_parameters():
                if "weight" in param_name and weight == True:
                    num_zeros += torch.sum(param < 0.01).item()
                    num_elements += param.nelement()
                if "bias" in param_name and bias == True:
                    num_zeros += torch.sum(param < 0.01).item()
                    num_elements += param.nelement()

        sparsity = num_zeros / num_elements

        return num_zeros, num_elements, sparsity

    def measure_global_sparsity(self, weight=True, bias=False, conv2d_use_mask=False, linear_use_mask=False):
        num_zeros = 0
        num_elements = 0

        for _, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module_num_zeros, module_num_elements, _ = self.measure_module_sparsity(
                    module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
                num_zeros += module_num_zeros
                num_elements += module_num_elements

            elif isinstance(module, torch.nn.Linear):

                module_num_zeros, module_num_elements, _ = self.measure_module_sparsity(
                    module, weight=weight, bias=bias, use_mask=linear_use_mask)
                num_zeros += module_num_zeros
                num_elements += module_num_elements

        sparsity = num_zeros / num_elements

        return num_zeros, num_elements, sparsity

    def validate(self):
        self.model.eval()
        total_loss = []
        total_acc = []
        val_loss = 0
        val_acc = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images, gts = batch
                images, gts = self.batch_to_device(images, gts)
                outputs = self.model(images)
                loss = self.criterion(outputs, gts)
                acc = cal_acc(outputs, gts)
                total_loss.append(loss.item())
                total_acc.append(acc)
                
                del outputs
                del loss
                
        val_loss = np.mean(total_loss)
        val_acc = np.mean(total_acc)
        self.model.train()
        
        return val_loss, val_acc

    def batch_to_device(self, images, gts):
        images = images.to(self.device, non_blocking=True)
        gts = gts.to(self.device, non_blocking=True)

        return images, gts

    def iterative_pruning(self, conv2d_prune_amount=0.4, linear_prune_amount=0.2, global_pruning=False):
        best_acc = 0

        # compute sparsity of the model
        _, _, sparsity = self.measure_global_sparsity(conv2d_use_mask=False)
        print('The model of sparsity: {.2f} '.format(sparsity))

        # compute the original model of acc on validation data
        val_loss, val_acc = self.validate()
        print('Before pruning: Loss: {.2f} Acc: {.2f}'.format(val_loss, val_acc))

        for i in range(self.prunning_num_iters):
            print('--------------------------------------------------------------------------')
            print("Pruning and Finetuning {}/{}".format(i + 1, self.prunning_num_iters))
            print('--------------------------------------------------------------------------')
            
            if global_pruning:
                parameters_to_prune = []

                for _, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        parameters_to_prune.append((module, "weight"))
                
                prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=conv2d_prune_amount)
            else:
                for _, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        prune.l1_unstructured(module, name="weight", amount=conv2d_prune_amount)
                    elif isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(module, name="weight", amount=linear_prune_amount)
                    
            # on each iteration, re-evaluating prunned model
            val_loss, val_acc = self.validate()
            print("==========================================================================")
            print('After pruning:')
            print('Loss: {.2f} Acc: {.2f}'.format(val_loss, val_acc))

            _, _, sparsity = self.measure_global_sparsity(conv2d_use_mask=True)
            print('The model of sparsity: {.2f} '.format(sparsity))
            print("==========================================================================")

            # fine-tuning
            print('Fine-tuning........................')
            self.fine_tune()

            val_loss, val_acc = self.validate()
            print("val_loss: {:.4f}, val_acc: {:.4f}".format(val_loss, val_acc))

            _, _, sparsity = self.measure_global_sparsity(conv2d_use_mask=True)
            print('The model of sparsity: {.2f} '.format(sparsity))
            print("==============================================================================")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), './weights/prunned_mode.pth.tar')
        
        self.remove_parameters()
        print('Final results: ')
        val_loss, val_acc = self.validate()
        print("val_loss: {:.4f}, val_acc: {:.4f}".format(val_loss, val_acc))

        _, _, sparsity = self.measure_global_sparsity(conv2d_use_mask=False)
        print('The model of sparsity: {.2f} '.format(sparsity))

        # save final weights
        torch.save(self.model.state_dict(), './weights/final_prunned_mode.pth.tar')

    def fine_tune(self):
        self.model.train()
        total_loss = 0
        global_step = 0


        data_iter = iter(self.train_loader)
        for i in range(self.trainning_num_iters):
            self.model.train()
            
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
                
            global_step += 1
            loss = self.train_step(batch)
            total_loss += loss

            if global_step % self.print_every == 0:
                print('step: {:06d}, train_loss: {:.4f}'.format(global_step, total_loss / self.print_every))
                total_loss = 0

    def train_step(self, batch):
        # get the inputs,
        images, gts = batch
        images, gts = self.batch_to_device(images, gts)
    
        # zero the parameter gradients,
        self.optimizer.zero_grad()
    
        # forward + backward + optimize + scheduler,
        outputs = self.model(images)
        loss = self.criterion(outputs, gts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()
        loss_item = loss.item()
        
        return loss_item

    def remove_parameters(self):
        for _, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                try:
                    prune.remove(module, "weight")
                except:
                    pass
                try:
                    prune.remove(module, "bias")
                except:
                    pass
            elif isinstance(module, torch.nn.Linear):
                try:
                    prune.remove(module, "weight")
                except:
                    pass
                try:
                    prune.remove(module, "bias")
                except:
                    pass

        