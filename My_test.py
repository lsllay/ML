import numpy as np
import operator
import os
import random
import matplotlib.pyplot as plt 

from os import listdir
import struct


#read image
def read_image(filename):
	pfile = open(filename,'rb')
	cfile = pfile.read()

	offset = 0
	head = struct.unpack_from('>IIII',cfile,offset)
	offset += struct.calcsize('>IIII')
	imageNum = head[1]
	rows = head[2]
	cols = head[3]	

	#28*28
	images = np.empty((imageNum,784))
	image_size = rows * cols;
	#单个图片的format
	fmt  = '>' + str(image_size) + 'B'

	for i in range(imageNum):
		images[i] = np.array(struct.unpack_from(fmt,cfile,offset))
		offset += struct.calcsize(fmt)

	return images

def read_label(filename):
	pfile = open(filename,'rb')
	cfile = pfile.read()


	head = struct.unpack_from('>II',cfile,0)
	offset = struct.calcsize('>II')

	labelNum = head[1]
	bitsString = '>' + str(labelNum) + 'B'
	label = struct.unpack_from(bitsString,cfile,offset)
	return label
	

def KNN(inX,dataSet,labels,K):
	Size_dataSet = dataSet.shape[0]

	distance0 = np.tile(inX,(Size_dataSet)).reshape((60000,784)) - dataSet
	distance1 = distance0 ** 2
	distance2 = distance1.sum(axis=1)
	distance = distance2 ** 0.5

	sortedDistIndices = distance.argsort()
	classCount = {}
	classCountlist= {}

	for i in range(K):
		value_label = labels[sortedDistIndices[i]]
		classCount[value_label] = classCount.get(value_label,0)+1
		classCountlist[value_label] = classCountlist.get(value_label,list())+([int(sortedDistIndices[i])])
	
	sortedclasscount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True) 

	
	return sortedclasscount,classCountlist
# 	---------------------------------------------------------------------

def test_KNN(k=30,times=10):
	train_image = "MINIST/train-images.idx3-ubyte"
	test_image = "MINIST/t10k-images.idx3-ubyte"
	train_label = "MINIST/train-labels.idx1-ubyte"
	test_label = "MINIST/t10k-labels.idx1-ubyte"

	train_x = read_image(train_image)
	train_y = read_label(train_label)

	test_x = read_image(test_image)
	test_y = read_label(test_label)

	#decide test_NUM---------------------
	#testRatio = 1
	#train_row = train_x.shape[0]
	#test_row = train_y.shape[0]

	#testNum = int(test_row*testRatio)
	#------------------------------------
	
	# errorCount = 0	 	
	for i in range(times):
		m = random.randint(0,test_x.shape[0])
		result_label,result_index = KNN(test_x[m],train_x,train_y,k)
		most_label = result_label[0][0]
		img_index_list = result_index[most_label]
		#plot the img
		#test_x[i],train_x[img_index_list[0~2]],label:test_y[i]
		fig,ax = plt.subplots(nrows = 1,ncols = 4,sharex = True,sharey = True)
		ax = ax.flatten()

		img0 = test_x[m].reshape(28,28)
		img1 = train_x[img_index_list[0]].reshape(28,28)
		img2 = train_x[img_index_list[1]].reshape(28,28)
		img3 = train_x[img_index_list[2]].reshape(28,28)

		
		ax[0].imshow(img0,cmap='Greys',interpolation='nearest')
		ax[1].imshow(img1,cmap='Greys',interpolation='nearest')
		ax[2].imshow(img2,cmap='Greys',interpolation='nearest')
		ax[3].imshow(img3,cmap='Greys',interpolation='nearest')



		ax[0].set_xticks([])
		ax[0].set_yticks([])
		plt.tight_layout()
		plt.show()
		print('返回结果是:%s,真实结果是:%s' % (result_label[0][0],test_y[m]))

# test_KNN()
# print('\n')
# test_KNN(100,10)
# print('\n')
# test_KNN(200,10)
# print('\n')
# test_KNN(800,10)


def test_KNN1(k=30,times=10):
	train_image = "MINIST/train-images.idx3-ubyte"
	test_image = "MINIST/t10k-images.idx3-ubyte"
	train_label = "MINIST/train-labels.idx1-ubyte"
	test_label = "MINIST/t10k-labels.idx1-ubyte"

	train_x = read_image(train_image)
	train_y = read_label(train_label)

	test_x = read_image(test_image)
	test_y = read_label(test_label)


	errorCount = 0	 	
	for i in range(times):
		m = random.randint(0,test_x.shape[0])
		result_label,result_index = KNN(test_x[m],train_x,train_y,k)		
		if(result_label[0][0]!=test_y[m]):
			errorCount  += 1
	return errorCount 


def misrate():
	x = [30,40,60,100,200,400,600,800,1000]
	y = list()
	for i in x:
		print(i)
		y.append([(test_KNN1(i,500))])

	
	fig  = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(x,y,'bo')
	plt.xlabel("K")
	plt.ylabel("ErrorCount")
	plt.show();
misrate()
