from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import torch
from torchvision.utils import save_image

from matplotlib import pyplot as plt
import numpy as np
import random
import seaborn as sns
import itertools

random.seed(233)

dataset_zip = np.load(os.path.join('dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), encoding="latin1", allow_pickle=True)

imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
latents_classes = dataset_zip['latents_classes']
metadata = dataset_zip['metadata'][()]

# Define number of values per latents and functions to convert to indices
latents_sizes = metadata['latents_sizes']	#array([ 1,  3,  6, 40, 32, 32])

latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))	
	#array([737280, 245760,  40960,   1024,     32,      1])

'''
	Color: white
	Shape: square, ellipse, heart
	Scale: 6 values linearly spaced in [0.5, 1]
	Orientation: 40 values in [0, 2 pi]
	Position X: 32 values in [0, 1]
	Position Y: 32 values in [0, 1]
	We varied one latent at a time (starting from Position Y, then Position X, etc), 
	and sequentially stored the images in fixed order.
'''
outdir = "dsprites_data" 

shapes = ['0square', '1ellipse', '2heart']

n_per_shape = 245760
n_per_scale = 40960
n_per_rotation = 1024

axis_n = 32
axis_unit = 32//axis_n
orien_n = 10
orien_unit = 40//orien_n

choices = {0:[0], 
			1: [0, 1, 2],
			2: [2, 3],
			3: [i*orien_unit for i in range(orien_n)],
			4: [i*axis_unit for i in range(axis_n)],
			5: [i*axis_unit for i in range(axis_n)]}



def latent_to_index(latents):
  return np.dot(latents, latents_bases).astype(int)

def sample_latent(size=1):
	samples = np.zeros((size, latents_sizes.size))
	# for lat_i, lat_size in enumerate(latents_sizes):
	#   samples[:, lat_i] = np.random.randint(lat_size, size=size)
	for lat_i in range(6):
		samples[:, lat_i] = np.random.randint(choices[lat_i], size=size)
	return samples

def is_train(x, y):
	if x<24 or y<24:
		return True
	else:
		return False

def is_holdout(x, y):
	if x >= 24 and y >= 24:
		return True
	else:
		return False

def generate_all_data(path, method):
	print('start generating')
	all_latents = np.array([])

	if method[:7] == "xy_axis":
		for i in choices[0]:
			for j in choices[1]:
				for k in choices[2]:
					for s in choices[3]:
						if method == 'xy_axis_x_y':
							x = 0
							for y in choices[5]:
								all_latents = np.append(all_latents, np.array([i, j, k, s, x, y]))
							y = 0
							for x in choices[4][1:]:
								all_latents = np.append(all_latents, np.array([i, j, k, s, x, y]))
						if method == 'xy_axis_all':
							for x in choices[4]:
								for y in choices[5]:
									all_latents = np.append(all_latents, np.array([i, j, k, s, x, y]))
						if method == 'xy_axis_x_diagonal':
							for x in choices[4]:
								all_latents = np.append(all_latents, np.array([i, j, k, s, x, x]))
							for y in choices[5][1:]:
								all_latents = np.append(all_latents, np.array([i, j, k, s, 0, y]))
						if method == 'xy_axis_x_y_diagonal':
							for y in choices[5]:
								all_latents = np.append(all_latents, np.array([i, j, k, s, 0, y]))
							for x in choices[4][1:]:
								all_latents = np.append(all_latents, np.array([i, j, k, s, x, 0]))
								all_latents = np.append(all_latents, np.array([i, j, k, s, x, x]))
						if method == "xy_axis_missing_diagonal":
							for x in choices[4]:
								for y in choices[5]:
									if x != y:
										all_latents = np.append(all_latents, np.array([i, j, k, s, x, y]))

		all_latents = all_latents.reshape(len(all_latents)//6, 6)
		all_indices = latent_to_index(all_latents)

		all_imgs = torch.tensor(imgs[all_indices], dtype=torch.int)
		print(all_imgs.shape)

		# save images by class x-y
		xy_str = ['0'+str(i) if i < 10 else str(i) for i in choices[4]]
		print(xy_str)
		for x in xy_str:
			for y in xy_str:
				classfolder = x + '-' + y
				os.makedirs(os.path.join(outdir, path, classfolder))

		for i in range(len(all_latents)):

			latent = all_latents[i]
			x = str(int(latent[4] // 10)) + str(int(latent[4] % 10))
			y = str(int(latent[5] // 10)) + str(int(latent[5] % 10))	
			classfolder = x  + '-' + y

			# if not os.path.exists(os.path.join(outdir, path, classfolder)):
			# 	os.makedirs(os.path.join(outdir, path, classfolder))
			save_image(all_imgs[i], os.path.join(outdir, path, classfolder, str(i) + '.png'))

	elif method[:5] == 'shape':
		if method == 'shape_all':
			all_imgs = torch.tensor(imgs, dtype=torch.int)

			for shape in shapes:
				os.makedirs(os.path.join(outdir, path, shape))

			for i in range(n_per_scale*2, n_per_scale*4, n_per_rotation*4):
				for j in range(n_per_rotation):
					save_image(all_imgs[i+j], os.path.join(outdir, path, '0square', str(i+j) + '.png'))
					save_image(all_imgs[i+j+n_per_shape], os.path.join(outdir, path, '1ellipse', str(i+j) + '.png'))
					save_image(all_imgs[i+j+n_per_shape*2], os.path.join(outdir, path, '2heart', str(i+j) + '.png'))

		if method == 'shape_no_corner':
			if not os.path.exists(os.path.join(outdir, method)):
				os.makedirs(os.path.join(outdir, method))

			all_latents = np.array([])
			train_latents = np.array([])
			holdout_latents = np.array([])
			for k in choices[2]:
				for s in choices[3]:
						for x in range(0, 32):
							for y in range(0, 32):
								all_latents = np.append(all_latents, np.array([0, 0, k, s, x, y]))
								if is_train(x, y):
									train_latents = np.append(train_latents, np.array([0, 0, k, s, x, y]))
								elif is_holdout(x, y):
									holdout_latents = np.append(holdout_latents, np.array([0, 0, k, s, x, y]))



			all_latents = all_latents.reshape(len(all_latents)//6, 6)
			train_latents = train_latents.reshape(len(train_latents)//6, 6)
			holdout_latents = holdout_latents.reshape(len(holdout_latents)//6, 6)
			
			print("finish collecting latents")
			print('all_latents size: ', len(all_latents))
			print('train_latents size: ', len(train_latents))
			print('holdout_latents size: ', len(holdout_latents))

			all_indices = latent_to_index(all_latents)
			train_indices = latent_to_index(train_latents)
			holdout_indices = latent_to_index(holdout_latents)

			all_imgs = {}
			for i in [0, 1, 2]:
				indices = all_indices + i*n_per_shape
				print(len(indices))
				all_imgs[i] = torch.tensor(imgs[indices], dtype=torch.int)

			for j, shape in enumerate(shapes):
				for i in ['0', '1', '2']:
					os.makedirs(os.path.join(outdir, path, 'missing_'+shape, 'train', i))
					os.makedirs(os.path.join(outdir, path, 'missing_'+shape, 'holdout', i))

				train_imgs = torch.tensor(imgs[train_indices+j*n_per_shape], dtype=torch.int)
				holdout_imgs = torch.tensor(imgs[holdout_indices+j*n_per_shape], dtype=torch.int)

				for i in range(len(train_imgs)):
					save_image(train_imgs[i], os.path.join(outdir, path, 'missing_'+shape, 'train', str(j), str(i) + '.png'))
				for k in [0, 1, 2]:
					if k != j:
						for i in range(len(all_imgs[k])):
							save_image(all_imgs[k][i], os.path.join(outdir, path, 'missing_'+shape, 'train', str(k), str(i) + '.png'))

				for i in range(len(holdout_imgs)):
					save_image(holdout_imgs[i], os.path.join(outdir, path, 'missing_'+shape, 'holdout', str(j), str(i) + '.png'))
				print('finish ', shape)
		
		if method == 'shape_no_3_corners':
			if not os.path.exists(os.path.join(outdir, path)):
				os.makedirs(os.path.join(outdir, path))
				for i in ['0', '1', '2']:
					os.makedirs(os.path.join(outdir, path, 'train', i))
					os.makedirs(os.path.join(outdir, path, 'holdout', i))

			train_latents = {}
			holdout_latents = {}
			for i in [0, 1, 2]:
				train_latents[i] = np.array([])
				holdout_latents[i] = np.array([])

			for k in choices[2]:
				for s in choices[3]:
						for x in range(0, 32):
							for y in range(0, 32):
								j = 0
								if is_train(x, y):
									train_latents[j] = np.append(train_latents[j], np.array([0, j, k, s, x, y]))
								elif is_holdout(x, y):
									holdout_latents[j] = np.append(holdout_latents[j], np.array([0, j, k, s, x, y]))
								
								j = 1
								if x<24 or y>=8:
									train_latents[j] = np.append(train_latents[j], np.array([0, j, k, s, x, y]))
								elif x>=24 and y<8:
									holdout_latents[j] = np.append(holdout_latents[j], np.array([0, j, k, s, x, y]))

								j = 2
								if y<24 or x>=8:
									train_latents[j] = np.append(train_latents[j], np.array([0, j, k, s, x, y]))
								elif y>=24 and x<8:
									holdout_latents[j] = np.append(holdout_latents[j], np.array([0, j, k, s, x, y]))
			print("finish collecting latents")
			
			for i in [0, 1, 2]:

				train_latents[i] = train_latents[i].reshape(len(train_latents[i])//6, 6)
				holdout_latents[i] = holdout_latents[i].reshape(len(holdout_latents[i])//6, 6)
				
				train_indices = latent_to_index(train_latents[i])
				holdout_indices = latent_to_index(holdout_latents[i])
				print('train size: ', len(train_indices))
				print('holdout size: ', len(holdout_indices))

				train_imgs = torch.tensor(imgs[train_indices], dtype=torch.int)
				holdout_imgs = torch.tensor(imgs[holdout_indices], dtype=torch.int)

				for k in range(len(train_imgs)):
					save_image(train_imgs[k], os.path.join(outdir, path, 'train', str(i), str(k) + '.png'))
				
				for k in range(len(holdout_imgs)):
					save_image(holdout_imgs[k], os.path.join(outdir, path, 'holdout', str(i), str(k) + '.png'))
			
	# with open(os.path.join(args.outdir, str(ts), 'params.json'), 'w') as fp:
	# json.dump(vars(args), fp, indent=4, sort_keys=True)

def generate_valid_data(train_path, valid_path, method, p=0.1):
	if method[:5] == 'shape':
		for shape in shapes:
			os.makedirs(os.path.join(outdir, valid_path, shape))
	if method == 'shape_all':
		valid_size = int(20480*p)
		indices = range(n_per_scale*2, n_per_scale*4, n_per_rotation*4)
		allowed_indices = [[i+j for j in range(n_per_rotation)] for i in indices]
		allowed_indices = list(itertools.chain.from_iterable(allowed_indices))
		for shape in shapes:
			for i in random.sample(allowed_indices, valid_size):
				shutil.move(os.path.join(outdir, train_path, shape, str(i)+'.png'), os.path.join(outdir, valid_path, shape, str(i)+'.png'))
			print('finish ', shape)

if __name__ == "__main__":
	path = "shape_no_corner_easy_new"	# shape_no_corner_easy, shape_no_3_corners_easy
	method = "shape_no_corner"	# shape_no_corner, shape_no_3_corners
	train_path = '{}_train'.format(path)
	test_path = '{}_test'.format(path)
	valid_path = '{}_valid'.format(path)
	# os.makedirs(train_path)
	# os.makedirs(test_path)
	# generate_data(60000, train_path, 'train')
	# generate_data(10000, test_path, 'test')
	# debug
	# generate_data(80000, train_path, 'full')
	# generate_data(6000, path, 'full')

	generate_all_data(path, method)
	# generate_valid_data(path, valid_path, method)



# python3 dsprites-dataset-master/dsprites_generator.py