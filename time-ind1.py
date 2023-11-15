from qutip import*
import matplotlib.pyplot as plt
import numpy as np
import math

import qutip.settings as settings
from qutip import __version__
from qutip.fastsparse import fast_csr_matrix, fast_identity

from qutip.sparse import (sp_eigs, sp_expm, sp_fro_norm, sp_max_norm,
                          sp_one_norm, sp_L2_norm)
from qutip.dimensions import type_from_dims, enumerate_flat, collapse_dims_super
from qutip.cy.spmath import (zcsr_transpose, zcsr_adjoint)


nn=3
h=6.63*10**(-34)
ghz=10**(9)
kb=1.38*10**(-23)
T=17*10**(-3)
bt=1/(kb*T)
e=1.6*10**(-19)

#6.33 for param1, 2.0 for param2, 13.46 for param3, 1.86 for param4 

#param set 1
ec=1.2
ecj=4
ej=6
el=0.038

#param set 2
#ec=0.15
#ecj=10
#ej=5.0
#el=0.13

#param set 3
#ec=0.65
#ecj=1.75
#ej=10.8
#el=0.79

#param set 4
#ec=0.04
#ecj=20
#ej=10.0
#el=0.04

cc=e**2/(2*ec*h*ghz)
cj=e**2/(2*ecj*h*ghz)
#print(cc)
#print(cj)
csig=cc+cj
ecsig= e**2/(2*csig*h*ghz)
#print(ecsig)


xi=(2*ecsig/ej)**(0.5)
xi2=cc/(2*csig)


a1=destroy(nn,(1-nn)/2)
c1=create(nn,(1-nn)/2)

a2=destroy(nn,(1-nn)/2)
c2=create(nn,(1-nn)/2)

a3=destroy(nn,(1-nn)/2)
c3=create(nn,(1-nn)/2)


iota=complex(0,1)
no1=iota*(c1-a1)/(2*(xi**(0.5)))
#print(iota*identity(2))
#print("no1=",no1)
no2=iota*(c2-a2)/(2*(xi**(0.5)))
#print(a1*c1+c1*a1)
no3=iota*(c3-a3)/(2*(xi**(0.5)))

no4=tensor([identity(nn),identity(nn),identity(nn)])

#print(iota*(c1-a1))
eigen1=(no1).eigenstates()
eigen2=(no2).eigenstates()
eigen3=(no3).eigenstates()
#print(1/(2*(xi**(0.5))))
print(eigen1[1])
m1=0
m2=0
m3=0
m12=0


for r in np.arange(nn-1):
	#m12=m12+0.5*(tensor([eigen1[1][r],eigen1[1][r+1].dag()])+tensor([eigen1[1][r+1],eigen1[1][r].dag()]))
	m1=m1+(1/2)*(eigen1[1][r]*eigen1[1][r+1].dag()+eigen1[1][r+1]*eigen1[1][r].dag())
			
for r in np.arange(nn-1):
	m2=m2+(1/2)*(eigen2[1][r]*eigen2[1][r+1].dag()+eigen2[1][r+1]*eigen2[1][r].dag())
			
	
for r in np.arange(nn-1):
	m3=m3+(1/2)*(eigen3[1][r]*eigen3[1][r+1].dag()+eigen3[1][r+1]*eigen3[1][r].dag())
			

listSup=[]
ham=[]
U=[]
m11=0
m22=0
m33=0
for r in np.arange(nn-1):
	m11=m11+0.5*(-iota)*(eigen1[1][r]*eigen1[1][r+1].dag()-eigen1[1][r+1]*eigen1[1][r].dag())
			
for r in np.arange(nn-1):
	m22=m22+0.5*(-iota)*(eigen2[1][r]*eigen2[1][r+1].dag()-eigen2[1][r+1]*eigen2[1][r].dag())
			
for r in np.arange(nn-1):
	m33=m33+0.5*(-iota)*(eigen3[1][r]*eigen3[1][r+1].dag()-eigen3[1][r+1]*eigen3[1][r].dag())
			

#print(tensor([eigen1[1][r],eigen2[1][r],eigen3[1][r]])*(tensor([eigen1[1][r+1].dag(),eigen2[1][r].dag(),eigen3[1][r].dag()])))
#print(m11)

phi3=np.pi/2-m3-(m3**3)/6-3*(m3**5)/40-15*(m3**7)/(7*48)-35*(m3**9)/1152-63*(m3**11)/(32*11*8)
phi1=np.pi/2-m1-(m1**3)/6-3*(m1**5)/40-15*(m1**7)/(7*48)-35*(m3**9)/1152-63*(m3**11)/(32*11*8)
phi2=np.pi/2-m2-(m2**3)/6-3*(m2**5)/40-15*(m2**7)/(7*48)-35*(m3**9)/1152-63*(m3**11)/(32*11*8)

cosphi1=m1
cosphi2=m2
cosphi3=m3

sinphi1=m11
sinphi2=m22
sinphi3=m33

print(phi1*no1-no1*phi1)

file4 = open("energy_val_7.txt","w")

listSup=[]
listSup2=[]
listSup3=[]
listSup5=[]
listSup4=[]
listSup6=[]
listSup7=[]

phibegin=-3
phiend=3
phi_inc=0.2
tval2 = np.arange(phibegin,phiend,phi_inc)

count=0
eigen=[]
for phie20 in np.arange(phibegin,phiend,phi_inc):
	phie10=phie20
	phie30=phie20
	cphi1=tensor([cosphi1,cosphi2*cosphi2,cosphi3])-tensor([sinphi1,(sinphi2*cosphi2+cosphi2*sinphi2)/2,cosphi3])+tensor([cosphi1,(sinphi2*cosphi2+cosphi2*sinphi2)/2,sinphi3])-tensor([sinphi1,sinphi2*sinphi2,sinphi3])

	cphi2=math.cos(xi2*(phie10+phie30))*no4
	sphi2=math.sin(xi2*(phie10+phie30))*no4
	
	sphi1=tensor([sinphi1,cosphi2*cosphi2,cosphi3])+tensor([cosphi1,(sinphi2*cosphi2+cosphi2*sinphi2)/2,cosphi3])+tensor([sinphi1,(sinphi2*cosphi2+cosphi2*sinphi2)/2,sinphi3])+tensor([cosphi1,sinphi2*sinphi2,sinphi3])
	
	
	ham1=-2*ej*(cphi1*cphi2-sphi1*sphi2)+(el/2)*(tensor([identity(nn),identity(nn),2*phi3**2])+tensor([identity(nn),4*phi2**2,identity(nn)])-tensor([identity(nn),4*phi2,phi3])-tensor([identity(nn),identity(nn),(phie10-phie30)*phi3])+(1/4)*((phie10-phie30)**2+(phie10+2*phie20+phie30)**2)*tensor([identity(nn),identity(nn),identity(nn)])+tensor([identity(nn),(phie10+2*phie20+phie30)*2*phi2,identity(nn)])-tensor([identity(nn),identity(nn),(phie10+2*phie20+phie30)*phi3]))

	kden=(8*cc*cj*(csig**3)*h*ghz)
    #A
	ham21=((2*cj*(16*(cc**3)+39*(cc**2)*cj+29*cc*(cj**2)+7*(cj**3)))/kden)*e*e*tensor([no1*no1,identity(nn),identity(nn)])
	#D
	ham22=(-(4*cj*(csig**2*(cc+7*csig)))/kden)*e*e*tensor([no1,no2,identity(nn)])
	#B
	ham23=((2*cj*(csig**2*(cc+7*csig)))/kden)*e*e*tensor([identity(nn),no2*no2,identity(nn)])
    #F
	ham31=(-2*cj*csig*((14*cc*cc+29*cc*cj+14*cj*cj)*e*e)/kden)*tensor([no1,identity(nn),no3])
	#E
	ham32=((28*cj*csig**3*e*e)/kden)*tensor([identity(nn),no2,no3])
	#C
	ham41=((csig**2*(14*cc*cc+27*cc*cj+14*cj*cj)*e*e)/kden)*tensor([identity(nn),identity(nn),no3*no3])
	ham0=ham1+ham21+ham22+ham23+ham31+ham32+ham41
		
	Energy2 = (ham0).eigenenergies()
	eigen=(ham0).eigenstates()
	#print(eigen[0][1]-eigen[0][0]+6.33)
	listSup2.append(eigen[0][5]+6.33)
	listSup3.append(eigen[0][0]+6.33)
	listSup4.append(eigen[0][1]+6.33)
	listSup5.append(eigen[0][2]+6.33)
	listSup6.append(eigen[0][3]+6.33)
	listSup7.append(eigen[0][6]+6.33)
	listSup.append(eigen[0][4]+6.33)
	#print(ham0)
	#print(eigen[0])
	#listSup2.append(eigen[0])
	
	count=count+1
	#print(tensor([cosphi1,identity(nn), identity(nn)]))
	np.savetxt(file4,Energy2,fmt='%1.10f')
#print(listSup2)
#print(eigen[0])
#plt.plot(tval2,listSup2,marker=".",markersize=1,linewidth=0.3)
#plt.show()
plt.plot(tval2,listSup3,marker=".",markersize=2,linewidth=1.0, c='b')
plt.plot(tval2,listSup4,marker=".",markersize=2,linewidth=1.0, c='r')
plt.plot(tval2,listSup5,marker=".",markersize=2,linewidth=1.0, c= 'k')
plt.plot(tval2,listSup6,marker=".",markersize=2,linewidth=1.0, c= 'm')
plt.plot(tval2,listSup2,marker=".",markersize=2,linewidth=1.0, c='g')
plt.plot(tval2,listSup,marker=".",markersize=2,linewidth=1.0, c= 'tab:cyan')
#plt.plot(tval2,listSup7,marker=".",markersize=2,linewidth=1.0, c= 'g')
#plt.plot(tval2,listSup,marker=".",markersize=1,linewidth=0.3)
plt.xlabel(r'$\varphi_{\rm ext}$', fontsize=20)
plt.ylabel("$E$ (h GHz)", fontsize=20)
plt.rcParams['font.size'] = 16
#plt.rc('xtick', labelsize=18) 
#plt.rc('ytick', labelsize=18)
#plt.savefig("test1.png",dpi=300)
plt.subplots_adjust(top=0.960,bottom=0.150,left=0.150,right=0.960,hspace=0.305,wspace=0.2)
plt.savefig('fig3c.eps', format='eps', dpi=1200)
plt.show()

'''
plt.plot(tval2,listSup3,marker=".",markersize=1,linewidth=0.3)
plt.plot(tval2,listSup4,marker=".",markersize=1,linewidth=0.3)
plt.plot(tval2,listSup5,marker=".",markersize=1,linewidth=0.3)
plt.plot(tval2,listSup6,marker=".",markersize=1,linewidth=0.3)
plt.plot(tval2,listSup2,marker=".",markersize=1,linewidth=0.3)
plt.plot(tval2,listSup,marker=".",markersize=1,linewidth=0.3)
plt.show()
r2=math.exp(-bt*ghz*h*eigen[0][0])

#print(r2)
dhn=0
for r in range(nn**3):
	dhn=dhn+math.exp(-bt*h*ghz*eigen[0][r])

#print(r2/dhn)
i=0
for r in range(nn**3):
	i=i+1
	r2=math.exp(-bt*h*ghz*eigen[0][r])
	
	print(r2,i)

'''
#nn=5 : 20
#nn=7: 43
#nn=11: 125
#nn=9: 75
