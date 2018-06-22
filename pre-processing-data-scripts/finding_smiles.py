import pandas as pd
import numpy as np
from salty import check_name
import pickle

density_all=pd.read_csv('density.csv') #all 30k data
density_all=density_all.drop(['Molar volume, m<SUP>3</SUP>/mol','Specific volume, m<SUP>3</SUP>/kg'],axis=1)
salts=np.array(density_all.salt_name)
unique_salts=np.unique(salts)
len(unique_salts)

print('there are '+str(len(unique_salts)) + ' unique salts')

#decide if pickling

two_without_split=[]
for i in unique_salts:    #making a list to use when comparing for descriptor table
    A=i.split()
    if len(A)==2:
        two_without_split.append(i)
    else:
        pass
two_without_split[0]

salts=[]
two=[]
three=[]
four=[]
more=[]

for i in unique_salts:
    A=i.split()              #sorting 2,3,4 and more ions into separate lists
    if len(A)==2:
        two.append(A)
    elif len(A)==3:
        three.append(A)
    elif len(A)==4:
        four.append(A)
    else:
        more.append(A)

print('There are '+ str(len(two))+' salts of 1 each') #looks fine and clean
print('There are '+ str(len(three))+' salts of 2 cations/1anion or 2 anions/1 cation each') #confirm which belongs to what
print('There are '+ str(len(four))+' salts of 2 each')
print('There are '+ str(len(more))+' salts of 2 or more each')

cation2=[]
anion2=[]
error2_anion=[]
error2_cation=[]

for i in two:
    cation2.append(i[0])
    anion2.append(i[1])

for i in anion2:   #CHECK CHECK_NAME FUNC FOR MISSING ANION OR CATION
    #print(i)
    try:
        check_name(i)
    except:
        UnboundLocalError
        error2_anion.append(i)
        #print(i)
for i in cation2:
    #print(i)
    try:
        check_name(i)
    except:
        UnboundLocalError
        error2_cation.append(i)

print('There are '+ str(len(set(error2_anion)))+' unique missing anions from the data base')
#print(error2_anion)
print('There are '+ str(len(set(error2_cation)))+ ' unique missing cations from the data base')
#len(set(error2_anion))
anion3=[]  #anion3 is the list for salts with 3 ions
cation3=[] #cation3 is the list for salts with 3ions
count=0
for i in three:
    if 'sulfate' in i:
        anion3.append(i[1]+' '+i[2])
        cation3.append(i[0])
        count+=1
    elif 'phosphate' in i:
        anion3.append(i[1]+' '+i[2])
        cation3.append(i[0])
        count+=1
    elif 'phosphonate' in i:
        anion3.append(i[1]+' '+i[2])
        cation3.append(i[0])
        count+=1
    elif 'carbonate' in i:
        anion3.append(i[1]+' '+i[2])
        cation3.append(i[0])
        count+=1

print('handling '+ str(count)+ ' out of '+str(len(three)))

for n,i in enumerate(anion3): #fixing the space to make it compatible with database
    if i=='diethyl phosphate':
        anion3[n]='diethylphosphate'
    elif i=='dimethyl phosphate':
        anion3[n]='dimethylphosphate'


error3_anion=[]
for i in anion3:
        try:
            check_name(i)
        except:
            UnboundLocalError
            error3_anion.append(i)

error3_cation=[]
for i in cation3:
        try:
            check_name(i)
        except:
            UnboundLocalError
            error3_cation.append(i)



print('There are '+ str(len(set(error3_anion)))+ ' missing anions from the data base')
#error3_anion
print('There are '+ str(len(set(error3_cation)))+ ' missing cations from the data base')
#error3_cation
