
import torch
from torchsummary import summary
import matplotlib.pyplot as plt

import matplotlib
import numpy as np
from PIL import Image


use_gpu = True if torch.cuda.is_available() else False

# trained on high-quality celebrity faces "celebA" dataset
# this model outputs 512 x 512 pixel images
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-512',
                       pretrained=True, useGPU=use_gpu)
model_g = model.netG

num_images = 1
noise_a, _ = model.buildNoiseData(1)
target = model_g(noise_a)
noise_b, _ = model.buildNoiseData(1)

print(noise_a.cpu().detach().numpy().shape)


# print(dir(model_g))

for param in model_g.parameters():
    param.requires_grad = False
noise_b = noise_b.to(torch.device("cuda:0"))
noise_b.retain_grad()

noise_b.requires_grad = True
noise_b.requires_grad_(True)

criterion = torch.nn.MSELoss().cuda()# nn.CrossEntropyLoss().cuda(args.gpu)


# The major problem is the input is not changing to match the output approriately.
# The error is not properly backpropagated or the input updates are not happening fine
# Probably try with simple GAN and updates

epsilon = 0.01
old_loss = 10

target_img = (target.permute(0, 2, 3, 1).cpu().detach().numpy()[0]+1)/2.0
print(np.min(target_img), np.max(target_img))
target_img = np.clip(target_img, 0, 1.0)
matplotlib.image.imsave("target.png", target_img)

# img = Image.open('050714.jpg')
# img = img.resize((512, 512), Image.LANCZOS )
# img = np.asarray( img)
# img=((img-128)/128.0) - 1
# print(np.min(img), np.max(img))
# img = np.expand_dims(img, axis=0)
# img = np.transpose(img, (0, 3, 1, 2))
# print(img.shape, target.cpu().detach().numpy().shape)
# target = torch.from_numpy(img).float().to(torch.device("cuda:0"))


for i in range(60000):
	noise_b = noise_b.to(torch.device("cuda:0"))
	noise_b.retain_grad()

	train_b = model_g(noise_b)
	loss = criterion(train_b, target)
	loss.backward(retain_graph=True)
	# print(noise_b.grad)

	if(i%100 == 0):
		print(i, loss)
		if(old_loss - loss < epsilon):
			epsilon 
			print(epsilon, loss - old_loss)
			image = (train_b.permute(0, 2, 3, 1).cpu().detach().numpy()[0]+1)/2.0
			image = np.clip(image, 0, 1.0)
			print(image.shape, np.min(image), np.max(image))
			matplotlib.image.imsave("output_"+str(i)+".png", image)

			learnt = noise_b.cpu().detach().numpy()
			expected = noise_a.cpu().detach().numpy()

			# print(learnt, expected)
			print(((learnt - expected)**2).mean())

		old_loss = loss


	sign_data_grad = noise_b.grad.sign()
	# Create the perturbed image by adjusting each pixel of the input image
	noise_b = noise_b - epsilon*sign_data_grad

	# print(sign_data_grad.cpu().detach().numpy())
	# noise_b = noise_b.to(torch.device("cuda:0"))
	# noise_b.retain_grad()

	# noise_b.requires_grad = True
	# noise_b.requires_grad_(True)
	# noise_b.zero_grad()
	# Adding clipping to maintain [0,1] range
	# noise_b = torch.clamp(noise_b, 0, 1)