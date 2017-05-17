import argparse
import os
import collections
import string
##Parse System Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data/brown_raw',
                    help='location of the data corpus')
parser.add_argument('--vocsize',type=int, default=10000,help='vocabulary_size')
parser.add_argument('--dest',type=str,default='./cleaned',help='Destination Directory')
args = parser.parse_args()

#
files_list = os.listdir(args.data)
wordcount = collections.Counter()

for filename in files_list:
	path = os.path.join(args.data,filename)
	#path = './data/gutenberg/train.txt'
	#Count words
	with open(path, 'rt') as f:
		for line in f:
			words = line.split() 
			wordcount.update(words)
most_common_counts = wordcount.most_common(n=args.vocsize)
most_common_words = [count[0] for count in most_common_counts]

for filename in files_list:
	print('cleaning '+filename)
	path = os.path.join(args.data,filename)
	#print(path)
	file_to_write = open(os.path.join(args.dest,filename),'w')
	#Replace each Unfrequent word with UNKOWN TOKEN
	with open(path, 'rt') as f:
		for line in f:
			words = line.split()
			words_output = []
			for word in words:
				if word in most_common_words:
					words_output.append(word)
				else:
					words_output.append('<UNK>')
			line_output = ' '.join(words_output)
			file_to_write.write(line_output+'\n')
