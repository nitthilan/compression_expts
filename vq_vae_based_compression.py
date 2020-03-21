
import lmdb
import os
import pickle
import numpy as np

from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq
import os


path="../vq-vae-2-pytorch/lmdb_output/vqvae_018.lmdb"


lmdb_env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

if not lmdb_env:
    raise IOError('Cannot open lmdb dataset', path)

num_bins = 512
top_hist = np.zeros(num_bins)
bottom_hist = np.zeros(num_bins)
with lmdb_env.begin(write=False) as txn:
    length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
    print(length)

    for index in range(1000):#length):
	    key = str(index).encode('utf-8')
	    # print(key)

	    row = pickle.loads(txn.get(key))
	    # print(row.top, row.bottom, row.filename)

	    th = np.histogram(row.top, bins=range(num_bins+1))
	    # print(th)

	    top_hist += th[0]
	    bottom_hist += np.histogram(row.bottom, bins=range(num_bins+1))[0]


top_prob = top_hist/np.sum(top_hist)
bot_prob = bottom_hist/np.sum(bottom_hist)
print(np.max(top_prob), np.min(top_prob), np.mean(top_prob), np.median(top_prob))
print(np.max(bot_prob), np.min(bot_prob), np.mean(bot_prob), np.median(bot_prob))

# print(bottom_hist/np.sum(bottom_hist))

# print(np.nonzero(top_hist)[0].shape)

# print(np.nonzero(bottom_hist)[0].shape)


# cumFreq_top = prob_to_cum_freq(top_hist/np.sum(top_hist), resolution=2048)
# cumFreq_bot = prob_to_cum_freq(bottom_hist/np.sum(bottom_hist), resolution=2048)
# # print(cumFreq_top)
# # print(cumFreq_bot)

# with lmdb_env.begin(write=False) as txn:
# 	length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
# 	print(length)

# 	for index in range(1000):#length):
# 		key = str(index).encode('utf-8')
# 		# print(key)

# 		row = pickle.loads(txn.get(key))
# 		# print(row.top, row.bottom, row.filename)

# 		filepath="encoded/encoded_"+str(index)+".txt"
		
# 		top_list = list(map(int,np.reshape(row.top, (-1))))
# 		bottom_list = list(map(int,np.reshape(row.bottom, (-1))))
# 		print(row.top)
# 		# print(bottom_list[:10])
# 		# # print(type(top_list), type([26, 56, 56, 56, 386, 496, 76, 467, 140, 475]))
# 		# top_list_1 = []
# 		# for k in top_list:
# 		# 	top_list_1.append(int(k))
# 		# encode data
# 		encoder = RangeEncoder(filepath)
# 		encoder.encode(top_list, cumFreq_top)
# 		encoder.encode(bottom_list, cumFreq_bot)
# 		encoder.close()
# 		print(index, os.stat(filepath).st_size, os.path.getsize(filepath))







