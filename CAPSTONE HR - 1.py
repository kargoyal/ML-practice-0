#!/usr/bin/env python
# coding: utf-8

# # Import all important libraries

# In[25]:


import numpy as ny
import pandas as ps
import matplotlib.pyplot as ml
import seaborn as sb


# # Read CSV

# In[70]:


DF=ps.read_csv(r"C:\Users\ShinChan\Downloads\HR_Employee_Attrition_Data.csv")


# # Exploratory DATA Analysis

# In[71]:


DF.head(3)


# In[72]:


DF.select_dtypes(include="int64").shape


# In[73]:


(DF.select_dtypes(exclude="int64")).shape


# In[74]:


DF.shape


# In[75]:


DF.columns


# In[76]:


sb.countplot(DF.Attrition[DF["Gender"]=="Male"])


# In[77]:


DF.Attrition.value_counts()


# In[78]:


sb.boxplot(DF.Age,DF.Attrition)


# In[79]:


ml.figure(figsize=(16,4))
sb.countplot(DF.JobRole)
ml.xticks(rotation=45,size=13)
ml.xlabel("JOB Role",size=15)
ml.ylabel("COUNT",size=15)
ml.show()


# In[80]:


ml.figure(figsize=(16,4))
sb.countplot(DF.Age[DF["Attrition"]=="Yes"])
ml.xticks(rotation=45,size=13)
ml.xlabel("JOB Role",size=15)
ml.ylabel("COUNT",size=15)
ml.show()


# In[81]:


ml.figure(figsize=(16,4))
sb.countplot(DF.EducationField[DF["Attrition"]=="Yes"])
ml.xticks(rotation=45,size=13)
ml.xlabel("JOB Role",size=15)
ml.ylabel("COUNT",size=15)
ml.show()


# In[82]:


DF.describe()


# In[83]:


DF.info()


# In[84]:


un={}
for i in DF.columns:
    un[i]=DF[i].nunique()
un


# # DATA Preprocessing

# In[85]:


sb.kdeplot(DF.Age[DF["Attrition"]=="Yes"],color="g")
sb.kdeplot(DF.Age[DF["Attrition"]=="No"],color="r",label="Aged")


# In[86]:


# NULL Value Detection
DF.isna().sum()


# In[87]:


# DROP UNIVARIANCE CLOUMNS
Drp=[]
for i in DF.columns:
    if DF[i].nunique() == 1 or DF[i].nunique() == len(DF):
        Drp.append(i)
Drp    


# In[88]:


DF.drop(Drp,inplace=True,axis=1)


# In[89]:


DF.shape


# In[90]:


CR=DF.select_dtypes(include="int64").corr()


# In[91]:


cr={}
for i in CR.columns:
    for j in CR.index:
        if (CR.loc[i,j]>0.75):
            if (i==j) or (j in cr.keys()) or (i in cr.keys()) :
                continue
            else:
                cr[i]=[j,CR.loc[i,j]]
cr        


# In[92]:


# DROP CORELATED COLUMNS
DF.drop(columns=["MonthlyIncome","PercentSalaryHike"],inplace=True,axis=1)


# In[93]:


DF.shape


# In[94]:


# OUTLIERS Detection
DS=DF.describe()
ds=[]
for i in DS.columns:
    if int(DS.loc["mean",i]) in range(int(DS.loc["50%",i]-DS.loc["std",i]),int(DS.loc["50%",i]+DS.loc["std",i]+1)):
        continue
    else:
        ds.append(i)
ds


# In[95]:


sb.distplot(DF.Age)


# # TRAIN - TEST

# In[98]:


from scipy import stats
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder,StandardScaler


# In[99]:


# Z SCORE or standardization
SD=ps.DataFrame(stats.zscore(DF.select_dtypes(include="int64")),columns=DF.select_dtypes(include="int64").columns)
for i in SD.columns:
    DF[i]=SD[i]


# In[167]:


# outliers & # SKEWNESS Remomval
print(DF.Age.skew())
DF.Age.kurt()


# In[101]:


for i in SD.columns:
    for j in DF.index:
        if (SD.loc[j,i]>3) or (SD.loc[j,i]<-3):
            DF.drop(index=[j],axis=0,inplace=True)
            


# In[102]:


DF.shape


# In[103]:


LD=DF.select_dtypes(include="object")


# In[104]:


ld=ps.DataFrame()
for i in DF.select_dtypes(include="object").columns:
    DF[i]=LabelEncoder().fit_transform(LD[i])
    ld[i]=LabelEncoder().fit_transform(LD[i])
ld.corr()


# In[107]:


X=DF.drop(columns=["Attrition"],axis=1)
Y=DF.Attrition
x,a,y,b=train_test_split(X,Y,test_size=0.2,random_state=7)


# # Logistic Regression

# In[108]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,auc,classification_report,recall_score
from sklearn.model_selection import cross_validate,cross_val_score


# In[109]:


LG=LogisticRegression().fit(x,y)


# In[110]:


b1=LG.predict(a)


# In[111]:


accuracy_score(b,b1)


# In[114]:


print(classification_report(b,b1))
confusion_matrix(b,b1)


# In[117]:


cross_val_score(LogisticRegression(),X,Y)
cross_validate(LogisticRegression(),X,Y)


# ## WITHOUT Preprocessing

# In[118]:


DP=ps.read_csv(r"C:\Users\ShinChan\Downloads\HR_Employee_Attrition_Data.csv")


# In[119]:


LC=DP.select_dtypes(include='object')


# In[120]:


for i in DP.select_dtypes(include="object").columns:
    DP[i]=LabelEncoder().fit_transform(LC[i])


# In[169]:


U=DP.drop(columns=["Attrition"])
V=DP.Attrition
u,j,v,k=train_test_split(U,V,random_state=3,test_size=0.2)


# In[138]:


u,j,v,k=train_test_split(M,N,random_state=3,test_size=0.2)


# In[170]:


MM=LogisticRegression().fit(u,v)


# In[140]:


k1=MM.predict(j)


# In[141]:


accuracy_score(k,k1)


# In[142]:


print(classification_report(k,k1))
confusion_matrix(k,k1)


# ### Understanding Purpose

# In[143]:


DR=DP.drop(columns=["Attrition",'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours',"MonthlyIncome","PercentSalaryHike"],axis=1)
t=DR.loc[500,:].to_numpy().reshape(1,28)
LG.predict(t)


# In[144]:


p=LG.predict(DR.to_numpy().reshape(2940,28))


# In[145]:


LG.predict(X.loc[2503,:].to_numpy().reshape(1,28))


# In[146]:


MM.predict(M.loc[2707,:].to_numpy().reshape(1,34))


# In[147]:


accuracy_score(DP.Attrition,p)


# In[156]:


print(classification_report(DP.Attrition,p))
confusion_matrix(DP.Attrition,p)


# # NAIVE BAYES 

# In[149]:


from sklearn.naive_bayes import MultinomialNB


# In[150]:


NB=MultinomialNB().fit(u,v)


# In[152]:


k2=NB.predict(j)


# In[153]:


accuracy_score(k,k2)


# In[155]:


print(classification_report(k,k2))
confusion_matrix(k,k2)


# # DF With Preprocessing

# In[158]:


m,f,n,g=train_test_split(DR,DP.Attrition,random_state=6,test_size=0.2)


# In[164]:


MM1=MultinomialNB().fit(m,n)
g1=MM1.predict(f)


# In[165]:


accuracy_score(g,g1)


# In[171]:


print(classification_report(g,g1))
confusion_matrix(g,g1)


#                                    ...... To Be Continued ......

# In[ ]:




