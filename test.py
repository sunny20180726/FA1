import random

allData.shape

random.random
a = []
for i in range(0,30000):
    a.append(random.randint(0,365))
b = {"ListingGap":a,
     "Idx":allData["Idx"]}

bb = pd.DataFrame(b)


result = pd.merge(allData,bb,on="Idx")
result.to_csv('allData_00.csv',encoding = 'gbk')


allData11 = pd.read_csv('allData_00.csv', header=0, encoding='gbk')
allData11["ListingGap"]