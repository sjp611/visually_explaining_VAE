import os
import numpy as np
import cv2
from PIL import Image

import argparse
import csv

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from models import VAE_model, VAE_loss
from gradcam import GradCam

from utils import average_meter, get_mnist_index, make_dir
from misc_functions import save_class_activation_images

torch.backends.cudnn.enabled = True

class vae_agent():
    def __init__(self, args):

        #Set dataset
        self.dataset = MNIST(root='data', train=True, download=True, transform = transforms.ToTensor())    
        self.train_indices, self.test_indices = get_mnist_index(self.dataset.targets, args.normal_class)

        #Set dataloader
        self.train_data_loader = torch.utils.data.DataLoader(\
                self.dataset, \
                batch_size=args.batch_size, \
                sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_indices),\
                drop_last = True) 

        self.val_data_loader = torch.utils.data.DataLoader(\
                self.dataset, \
                batch_size=1, \
                sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_indices),\
                drop_last = True) 

        self.test_data_loader = torch.utils.data.DataLoader(\
                self.dataset, \
                batch_size=1, \
                sampler = torch.utils.data.sampler.SubsetRandomSampler(self.test_indices))
            
        #Set model
        self.model = VAE_model(args)
        self.loss_func = VAE_loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.learning_rate)

        #Set model cuda
        if args.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(args.gpu_device)
            torch.cuda.manual_seed_all(args.seed)
        else:
            print("not set cuda")
            exit()

        self.model = self.model.to(self.device) 
        self.loss_func = self.loss_func.to(self.device)

        #args
        self.current_epoch = 0
        self.max_epochs = args.max_epochs
        self.seed = args.seed

        self.kl_c = args.kl_c
        self.rec_c = args.rec_c
        self.latent_dim = args.latent_dim

        self.save_log_name = args.save_log_name
        self.save_model_name = args.save_model_name
        self.train_save_fig_log = args.train_save_fig_log
        self.train_save_fig_counts = args.train_save_fig_counts
        self.test_save_fig_counts = args.test_save_fig_counts
        self.sampling_save_fig_counts = args.sampling_save_fig_counts
    
        
        #Make dirs
        self.experiment_path = os.getcwd() + args.experiments
        self.save_train_image_path = self.experiment_path + 'train_figs/'
        self.save_test_image_path = self.experiment_path + 'test_figs/'
        self.save_sampling_image_path = self.experiment_path + 'sampling/'
        self.save_visual_normal_image_path = self.experiment_path + 'visual_figs/normal/'
        self.save_visual_abnormal_image_path = self.experiment_path + 'visual_figs/abnormal/'
        self.save_visual_anoloc_image_path = self.experiment_path + 'visual_figs/ano_loc/'
        self.save_model_path = self.experiment_path + 'checkpoint/'
        self.save_log_path = self.experiment_path + 'log/'

        make_dir(self.save_train_image_path)
        make_dir(self.save_test_image_path)
        make_dir(self.save_sampling_image_path)
        make_dir(self.save_visual_normal_image_path)
        make_dir(self.save_visual_abnormal_image_path)
        make_dir(self.save_visual_anoloc_image_path)
        make_dir(self.save_model_path)
        make_dir(self.save_log_path)

    def save_checkpoint(self, file_path):
        state = {
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optimizer.state_dict(),
                    'seed': self.seed
                }
        torch.save(state, file_path)

    def load_checkpoint(self, file_path):
        try:
            checkpoint = torch.load(file_path)

            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            self.seed = checkpoint['seed']

            print("successfully load checkpoint")

        except OSError as e:
            print("First time to train")

    def train(self):
        self.model.train()

        for epoch in range(self.max_epochs+1):

            self.current_epoch = epoch
            train_epoch_loss = average_meter()
            rec_epoch_loss = average_meter()
            kl_epoch_loss = average_meter()

            img_count = 0 #generated image count

            for iteration, (x,y) in enumerate(self.train_data_loader):

                x, y = x.to(self.device), y.to(self.device) # x: image, y: label

                x_hat, mu, log_var = self.model(x, train = True)
                train_loss, rec_loss, kl_loss = self.loss_func(x_hat, x, mu, log_var, self.rec_c, self.kl_c)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                train_epoch_loss.update(train_loss.item())
                rec_epoch_loss.update(rec_loss.item())
                kl_epoch_loss.update(kl_loss.item())

                #Save images
                if (epoch % self.train_save_fig_log) == 0:
                    save_path = self.save_train_image_path + str(epoch) + '/'
                    make_dir(save_path)

                    for class_in_batch in y:
                        if img_count >= self.train_save_fig_counts:
                            continue
                        img_count += 1

                        x_np_img = x[class_in_batch].cpu().view(28, 28).data.numpy() * 255
                        x_hat_np_img = x_hat[class_in_batch].cpu().view(28, 28).data.numpy() * 255

                        result = np.concatenate([x_np_img, x_hat_np_img], axis=1)

                        cv2.imwrite(save_path + str(img_count) + "_" + str(class_in_batch.item()) + ".png", result) 

            print("Epoch {:02d}/{:02d} , total_Loss {:9.4f}, rec_Loss {:9.6f}, kl_Loss {}".format(
                epoch, self.max_epochs, train_loss.item(), rec_loss.item(), kl_loss.item() ))
            
            #log
            if not(os.path.exists(self.save_log_path + self.save_log_name)):
                with open(self.save_log_path + self.save_log_name, 'w', newline='') as train_writer_csv:
                    header_list = ['epoch', 'train_loss', 'rec_loss', 'kl_loss']
                    train_writer = csv.DictWriter(train_writer_csv, fieldnames= header_list)
                    train_writer.writeheader()
            with open(self.save_log_path + self.save_log_name, 'a', newline='') as train_writer_csv:
                train_writer = csv.writer(train_writer_csv)
                train_writer.writerow([epoch, str(train_epoch_loss.val), str(rec_epoch_loss.val), str(kl_epoch_loss.val)])

            train_epoch_loss.reset()
            rec_epoch_loss.reset()
            kl_epoch_loss.reset()

            self.save_checkpoint(file_path = self.save_model_path + self.save_model_name)

    def test(self):

        self.load_checkpoint(file_path = self.save_model_path + self.save_model_name)
        self.model.eval()

        for iteration, (x,y) in enumerate(self.test_data_loader):

            if iteration >= self.test_save_fig_counts:
                continue

            x, y = x.to(self.device), y.to(self.device) # x: image, y: label

            x_hat, _, _ = self.model(x, train = False)

            #Save images
            x_np_img = x.cpu().view(28, 28).data.numpy() * 255
            x_hat_np_img = x_hat.cpu().view(28, 28).data.numpy() * 255
            
            result = np.concatenate([x_np_img, x_hat_np_img], axis=1)

            cv2.imwrite(self.save_test_image_path + str(iteration) + "_" + str(y.item()) + ".png", result) 
            print("save file: ", self.save_test_image_path + str(iteration) + "_" + str(y.item()) + ".png")



    def visual_vae(self):
        self.load_checkpoint(file_path = self.save_model_path + self.save_model_name)
        self.model.eval()

        grad_cam = GradCam(self.model, target_layer = 'encoder3')
        

        #normal
        for iteration, (x,y) in enumerate(self.val_data_loader):
            if iteration >= self.test_save_fig_counts:
                continue
            else:

                x, y = x.to(self.device), y.to(self.device) # x: image, y: label

                cam = grad_cam.generate_cam(x, anomaly = False)

                x_np_img = x.squeeze(0).squeeze(0).cpu().data.numpy() * 255
                x_np_pil = Image.fromarray(x_np_img)
                cam_np_img = cam * 255
                
                file_name = str(iteration) + "_" + str(y.item())

                save_class_activation_images(x_np_pil, cam, self.save_visual_normal_image_path + file_name)

                print("file_path: " + self.save_visual_normal_image_path + file_name)

        #abnormal
        for iteration, (x,y) in enumerate(self.test_data_loader):
            if iteration >= self.test_save_fig_counts:
                continue
            else:

                x, y = x.to(self.device), y.to(self.device) # x: image, y: label

                cam = grad_cam.generate_cam(x)

                x_np_img = x.squeeze(0).squeeze(0).cpu().data.numpy() * 255
                x_np_pil = Image.fromarray(x_np_img)
                cam_np_img = cam * 255 

                file_name = str(iteration) + "_" + str(y.item())

                save_class_activation_images(x_np_pil, cam, self.save_visual_abnormal_image_path + file_name)

                print("file_path: " + self.save_visual_abnormal_image_path + file_name)

    def sampling(self):

        self.load_checkpoint(file_path = self.save_model_path + self.save_model_name)
        self.model.eval()

        for sample_cnt in range(self.sampling_save_fig_counts):
            z = torch.randn(1, self.latent_dim)

            z = z.to(self.device)

            sample = self.model.decode(z)

            sample = sample.cpu().view(28,28).data.numpy() * 255

            cv2.imwrite(self.save_sampling_image_path + str(sample_cnt) + ".png", sample) 
            print("save file: ", self.save_sampling_image_path + str(sample_cnt) + ".png")

                       
        
    def anomaly_loc(self):
        self.load_checkpoint(file_path = self.save_model_path + self.save_model_name)
        self.model.eval()

        grad_cam = GradCam(self.model, target_layer = 'encoder3')
        
        for iteration, (x,y) in enumerate(self.test_data_loader):
            if iteration >= self.test_save_fig_counts:
                continue
            else:

                x, y = x.to(self.device), y.to(self.device) # x: image, y: label

                cam = grad_cam.generate_cam(x, anomaly = True)

                x_np_img = x.squeeze(0).squeeze(0).cpu().data.numpy() * 255
                x_np_pil = Image.fromarray(x_np_img)
                cam_np_img = cam * 255
                
                file_name = str(iteration) + "_" + str(y.item())

                save_class_activation_images(x_np_pil, cam, self.save_visual_anoloc_image_path + file_name)

                print("file_path: " + self.save_visual_anoloc_image_path + file_name)
 



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--feature_in", type=int, default=1)
    parser.add_argument("--feature_num", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--normal_class", type=int, default=1)
    parser.add_argument("--rec_c", type=int, default=1)
    parser.add_argument("--kl_c", type=int, default=8)

    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--gpu_device", type=int, default=0)

    parser.add_argument("--train_save_fig_log", type=int, default=100, help='')
    parser.add_argument("--train_save_fig_counts", type=int, default=10, help='')
    parser.add_argument("--test_save_fig_counts", type=int, default=200, help='')
    parser.add_argument("--sampling_save_fig_counts", type=int, default=200, help='')

    parser.add_argument("--experiments", type=str, default='/experiments/')
    parser.add_argument("--save_model_name", type=str, default='checkpoint.pth.tar')
    parser.add_argument("--save_log_name", type=str, default='train_loss.csv')

    args = parser.parse_args()

    main = vae_agent(args)

#    main.train()
#    main.test()
#    main.sampling()
    main.visual_vae()
    main.anomaly_loc()

