from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq
import os

data = [2, 0, 1, 0, 0, 0, 1, 2, 2]
prob = [0.5, 0.2, 0.3]

# convert probabilities to cumulative integer frequency table
cumFreq = prob_to_cum_freq(prob, resolution=4)

print(cumFreq)

filepath="output.txt"
# encode data
encoder = RangeEncoder(filepath)
encoder.encode(data, cumFreq)
encoder.close()

# decode data
decoder = RangeDecoder(filepath)
dataRec = decoder.decode(len(data), cumFreq)
decoder.close()

print(os.stat(filepath))
print (dataRec)