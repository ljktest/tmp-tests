import scitools.easyviz as sea
import numpy as np
import scitools.filetable as ft
import matplotlib.pyplot as plt

# Lambda comparison
# Results in Softs/MethodesParticulaires/Resultats_simu/Comp_lmanda
fileDt=(('drag129_fixe'),('drag129_var'),('drag65_fixe'),('drag65_var'))
fileListLayer=(('drag_01'),('drag_02'),('drag_03'))
fileListLambda=(('drag_05'),('drag_06'),('drag_07'),('drag_09'),('drag_11'))
fileListLambda2=(('d129_4'),('d129_5'),('drag_06'),('d129_7'),('d129_12'))
fileListLambda3=(('d129_5'),('d257_5'))#,('d257_7'))
fileListLambda4=(('drag_06'),('d257_6'))#,('d257_7'))

legendLayer=('layer=0.1', 'layer=0.2','layer=0.3')
legendLambda=('lambda=1e5','lambda=1e6','lambda=1e7','lambda=1e9','lambda=1e11', 'Ref from Folke : 0.6726')
legendLambda2=('lambda=1e4','lambda=1e5','lambda=1e6','lambda=1e7','lambda=1e12','Ref from Folke : 0.6726')
legendLambda3=('lambda=1e5','257 - lambda=1e5','Ref from Folke : 0.6726')
legendLambda3=('lambda=1e6','257 - lambda=1e6','Ref from Folke : 0.6726')
plt.hold('off')
plt.xlabel('time')
plt.hold('on')
plt.ylabel('drag')
plt.axis([0,70,0.3,1])
plt.grid('on')

for filename in fileListLambda3:
	print ("my file is ", filename)
	file=open(filename)
	table=ft.read(file)
	time=table[:,0]
	drag=table[:,1]
	file.close()
	plt.plot(time,drag,'--')
plt.axhline(y=0.6726,xmin=0,xmax=22,color='r')
plt.legend(legendLambda3)
#plt.hold('on')

plt.savefig('DragRe133_CompLambda3.pdf')
plt.show()
