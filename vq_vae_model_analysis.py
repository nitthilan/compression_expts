from vqvae import VQVAE
from torch import nn, optim

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


import torch

device = 'cuda'

# transform = transforms.Compose(
#     [
#         transforms.Resize(args.size),
#         transforms.CenterCrop(args.size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#     ]
# )

# dataset = datasets.ImageFolder(args.path, transform=transform)
# loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

model = nn.DataParallel(VQVAE()).to(device)

model.module.load_state_dict(torch.load("../vq-vae-2-pytorch/checkpoint/vqvae_018.pt"))

embed_t = model.module.quantize_t.embed.cpu().data.numpy()
embed_b = model.module.quantize_b.embed.cpu().data.numpy()
print(embed_t.shape, embed_b.shape)

X_embed_t = TSNE(n_components=2).fit_transform(embed_t.transpose())
X_embed_b = TSNE(n_components=2).fit_transform(embed_b.transpose())

print(X_embed_t.shape)
# for i in range(512):
# 	for j in range(i, 512):
# 		print(i, j, embed_t)

plt.scatter(X_embed_t[:,0], X_embed_t[:,1])
plt.savefig('top_embedding.png', bbox_inches='tight')

plt.scatter(X_embed_b[:,0], X_embed_b[:,1])
plt.savefig('bot_embedding.png', bbox_inches='tight')
