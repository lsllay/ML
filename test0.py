#Perceptron


import numpy  as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


#input: DataSet数据集
#		times max训练的次数
#output:weight训练出的法向量
#		NotSeparated 是否可分
weight_log = []

def train(DataSet,times): 
	global weight_log
	numsample = DataSet.shape[0] #sample点的个数
	num_d = DataSet.shape[1] # x维数
	weight = np.zeros(num_d)#(w,b)
	labelSet = DataSet[:,-1]
	newDataSet = np.hstack((DataSet[:,:-1],np.ones((numsample,1))))#(x,1)

	# print(newDataSet)
	NotSeparated = True
	ax = Plot_Train(DataSet)
	weight_log=[]
	while(times > 0 and NotSeparated ):
		m = 0
		for i in range(numsample):
			if(labelSet[i] * np.sum(weight*newDataSet[i,:]) <= 0):
				weight = weight + labelSet[i] * newDataSet[i,:]
				weight_log.append(weight)
				m = m + 1
		if(m==0):
			NotSeparated = False
		times -= 1
	return ax

#input:  numLines: 数据集的个数
#        realWeight: 真正的法向量
#output: 数据集DataSet(x,y)
def MakeData(numLines,realWeight):
	w = np.array(realWeight)
	num_d = len(realWeight)
	DataSet = np.zeros((numLines*2,num_d))

	for i in range(numLines):
		x = np.random.rand(1,num_d-1)*20 - 10 #二维
		x0 = np.append(x,1)
		innerProduct = np.sum(w*x0)
		if(innerProduct<=0):
			DataSet[i] = np.append(x,-1)
		else:
			DataSet[i] = np.append(x,1)
	#
	for i in range(numLines):
		x = np.random.rand(1,num_d-1)*20 - 10 #二维
		x0 = np.append(x,1)
		innerProduct = np.sum(w*x0)
		if(innerProduct<=0):
			DataSet[i+numLines] = np.append(x,1)
		else:
			DataSet[i+numLines] = np.append(x,-1)

	return DataSet

#input:  DataSet:数据集
#		 weight:训练出的法向量
#output: figure
def Plot_Train(DataSet):
	#散点图
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111)
	ax.set_title('Linear separable DataSet')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.axis("equal")
	idx_1 = np.where(DataSet[:,-1]==1)
	idx_2 = np.where(DataSet[:,-1]==-1)

	p1 = ax.scatter(DataSet[idx_1,0],DataSet[idx_1,1],marker='o', color='g', label=1, s=20)
	p2 = ax.scatter(DataSet[idx_2,0],DataSet[idx_2,1],marker='x', color='r', label=2, s=20)
	return ax

def Show_Weight(weight,ax):
	a = weight[0] 
	b = weight[1] 
	c = weight[2] 
	# ax + by + c = 0
	# y = -(ax+c)/b
	ys =(-(a*(-10)+c)/b,-(a*10+c)/b)
	ax.add_line(Line2D((-10, 10), ys, linewidth=1, color='blue'))
	

def Cost(DataSet,weight):
	labelSet = DataSet[:,-1]
	newDataSet = np.hstack((DataSet[:,:-1],np.ones((DataSet.shape[0],1))))#(x,1)
	result = labelSet * np.sum(weight*newDataSet) #计算ax+by+c
	# for i in range(len(result)):
	# 	if(result[i]>0):
	# 		result[i]=0
	np.where(result < 0,result,0)
	return np.sqrt(sum(result)**2/(weight[0]**2+weight[1]**2))

from matplotlib import animation


DataSet = MakeData(200,[2,1,0])
ax = train(DataSet,3)
x = np.arange(-10, 10, 0.1)
line, = ax.plot(x,np.zeros(x.shape))
def animate_update(weight):
    x = line.get_xdata()
    a,b,c = weight
    line.set_ydata((-c-a*x)/b)
    return line,
def initial():
    line.set_ydata([0]*len(x))
    return line,
ani = animation.FuncAnimation(fig=plt.gcf(),func=animate_update,frames=weight_log,
	init_func=initial,interval=200,blit=True, repeat=False) 
ani.save("./train2.gif",)
plt.show()	

