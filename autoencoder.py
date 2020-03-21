import torch
from torchsummary import summary
from torch.autograd import Variable
# import pytorch_ssim
import numpy as np

# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
# different losses: https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
# https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_pgan/
# https://github.com/yxlao/reverse-gan.pytorch/blob/master/dcgan_reverse.py
# https://github.com/SubarnaTripathi/ReverseGAN
# https://openreview.net/pdf?id=HJC88BzFl - PRECISE RECOVERY OF LATENT VECTORS FROM GENERATIVE ADVERSARIAL NETWORKS
#	- Try clipping the latent vector within range to see if performance improves
use_gpu = True if torch.cuda.is_available() else False

# trained on high-quality celebrity faces "celebA" dataset
# this model outputs 512 x 512 pixel images
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-512',
                       pretrained=True, useGPU=use_gpu)
# this model outputs 256 x 256 pixel images
# model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
#                        'PGAN', model_name='celebAHQ-256',
#                        pretrained=True, useGPU=use_gpu)

# summary(model, (1,1,512))

num_images = 4
noise, _ = model.buildNoiseData(num_images)
with torch.no_grad():
    generated_images = model.test(noise)

# let's plot these images using torchvision and matplotlib
import matplotlib.pyplot as plt
import matplotlib
import torchvision
grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
grid_images = grid.permute(1, 2, 0).cpu().numpy()
print(grid_images.shape)
matplotlib.image.imsave('output.png', grid_images) 

matplotlib.image.imsave('output.png', grid_images) 
plt.imshow(grid_images)

print(noise, _)

# print(dir(model))

model_g = model.netG
# print(dir(model_g))

for param in model_g.parameters():
    param.requires_grad = False

noise, _ = model.buildNoiseData(num_images)
noise1, _ = model.buildNoiseData(num_images)
predFakeG1 = model_g(noise1)
noise1.requires_grad = True

noise = Variable(noise.to(torch.device("cuda:0")))
noise.retain_grad()

noise.requires_grad = True
noise.requires_grad_(True)



mse_criterion = torch.nn.MSELoss().cuda()# nn.CrossEntropyLoss().cuda(args.gpu)
criterion = pytorch_ssim.SSIM(window_size = 11)#nn.MSELoss()
criterion = criterion.cuda()

# parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                     metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)',
#                     dest='weight_decay')

# optimizer = torch.optim.SGD([noise], 0.1, # model_g.parameters()
#                             momentum=0.9,
#                             weight_decay=1e-4);
optimizer = torch.optim.Adam([noise], lr=3e-4)

noise_numpy = noise.detach().cpu().numpy()
noise1_numpy = noise1.detach().cpu().numpy()

print(np.max(noise_numpy), np.min(noise_numpy), noise_numpy.shape)
print(np.max(noise1_numpy), np.min(noise1_numpy))

for i in range(10000000):
	predFakeG = model_g(noise)
	loss = 1-criterion(predFakeG,predFakeG1)
	loss_1 = mse_criterion(noise1,noise)
	print(i, loss.detach().cpu().numpy(), loss_1.detach().cpu().numpy())
	# compute gradient and do SGD step
	optimizer.zero_grad()
	# noise.zero_grad()
	loss.backward()
	optimizer.step()
	# print(noise.grad)



# plt.show()



