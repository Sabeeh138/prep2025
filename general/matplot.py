import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = ['c','c++','python','java','kotlin']
y = [10,20,30,40,50]
z = [5,15,25,35,45]

# BAR PLOT
plt.bar(x,y,color='y',edgecolor='r',width=0.5,label='Graph 1')
plt.bar(x,z,color='b',edgecolor='r',width=0.5,label='Graph 2')
plt.legend()
plt.show()

# SCATTER PLOT
plt.scatter(x,y,c='y',edgecolor='r',marker='*')
plt.show()

# HIST PLOT
plt.hist(y,color='b',edgecolor='r',bins=20,orientation='horizontal')
plt.show()
plt.hist(y,color='b',edgecolor='r',histtype='step')
plt.show()

# PIE CHART
ex = [0.4,0,0,0,0]
plt.pie(y,labels=x,explode=ex,autopct="%0.01f%%")
plt.show()

# STACK PLOT
a1 = [2,3,2,5,4]
a2 = [2,3,4,5,6]
a3 = [1,3,2,4,2]
l = ['area 1','area 2','area 3']
plt.stackplot(y,a1)
plt.show()
plt.stackplot(y,a1,a2,a3,labels=l)
plt.legend()
plt.show()

# BOX PLOT
plt.boxplot(y,widths=0.3,label='Python',patch_artist=True,showmeans=True,)
plt.show()
plt.boxplot(y,vert=False,widths=0.3)
plt.show()
lists = [y,z]
plt.boxplot(lists,labels=['Python','c++'],showmeans=True)
plt.show()

# STEP PLOT
plt.step(y,z,color = 'r',marker='o')
plt.grid()
plt.show()

# LINE GRAPH
plt.plot(y,z)
plt.text(30,10,'Hello There')
plt.annotate('Python',xy=(20,20),xytext=(30,30),arrowprops=dict(facecolor='black'))
plt.show()

# STEM DIAGRAM
plt.stem(y,z,linefmt=':',markerfmt='r*')
plt.show()

# FILL BETWEEN
plt.plot(y,z,color= 'red')
plt.fill_between(y,z)
plt.show()

plt.plot(y,z,color= 'red')
plt.fill_between(x=[20,40],y1 = 20, y2 = 40)
plt.show()

n1 = np.array(y)
n2 = np.array(z)

plt.plot(y,z,color= 'red')
plt.fill_between(n1,n2,color = 'g', where=(n1 >= 20) & (n1 <= 40))
plt.show()