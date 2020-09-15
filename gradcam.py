from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import math


class CamExtractor():
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_Encoder(self, x):
        conv_output = None

        for module_pos, module in self.model._modules.items():
            x = module(x)  # Forward

            if module_pos == self.target_layer:
                x.register_hook(self.save_gradient)
                A = x  

                return A, x
                
    def abnormal_forward_pass(self, x):
        A, x = self.forward_pass_on_Encoder(x)

        #abnormal mu, log_var
        latent = x.view(x.size(0), -1)
        mu_y = self.model.mean(latent)
        log_var_y = self.model.variance(latent)

        #normal N(0,1)
        mu_x = torch.zeros_like(mu_y)
        log_var_x = torch.ones_like(log_var_y)

        #normal difference distribution
        mu_dif = mu_x - mu_y
        log_var_dif = log_var_x + log_var_y

        z = self.model.reparameterize(mu_dif, log_var_dif)

        return A, z


    def normal_forward_pass(self, x):
        A, x = self.forward_pass_on_Encoder(x)
        
        #mean, var
        latent = x.view(x.size(0), -1)
        mu = self.model.mean(latent)
        log_var = self.model.variance(latent)
        
        z = self.model.reparameterize(mu, log_var)

        return A, z
        
class GradCam():
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, anomaly=True):

        if anomaly:
            A, z = self.extractor.abnormal_forward_pass(input_image)
        else:
            A, z = self.extractor.normal_forward_pass(input_image)

        self.model.zero_grad()

        A = A.squeeze(0) #16 4 4
        z = z.squeeze(0) #32

        n, w, h = A.shape #n:16, w:4, h:4 
        z_len = int(len(z))

        M = torch.zeros([z_len, w, h]).cuda() # 32, 4, 4
        
        for z_target, z_i in enumerate(z):
        
            z_label = torch.zeros_like(z)
            z_label[z_target] = 1

            z.backward(gradient = z_label, retain_graph=True)

            # Get hooked gradients
            guided_gradients = self.extractor.gradients.cpu().data.numpy()[0]

            #Global Average Pooling
            c, w, h = guided_gradients.shape
            t = w * h 
            a_k = np.sum(guided_gradients, axis=(1,2)) / t

            M_i_linear = torch.zeros_like(A[1, :, :])
            for k in range(n):
                M_i_linear += a_k[k] * A[k, :, :]

            M_i = F.relu(M_i_linear) # 4 * 4

            M[z_target, :, :] += M_i

        M = torch.mean(M, dim=0)
        M = M.cpu().data.numpy() #4 4

        M = (M - np.min(M)) / (np.max(M) - np.min(M))

        M = np.uint8(M * 255)

        M = np.uint8(Image.fromarray(M).resize((input_image.shape[2],
                        input_image.shape[3]), Image.ANTIALIAS))/255

        return M
