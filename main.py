
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

import PdHelper
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import Utils
pd.options.mode.chained_assignment = None

EXPECTED_SCORE = 90.0 # we expect to forecast correctly at least 90%

def count_estimated_Ages(listAges):
    #Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
    count=0
    numBaby=0
    for a in listAges:
        if a-int(a)>0 and a>1:
            count+=1
        if a<1:
            numBaby+=1
    print("num of estimated Ages: " + str(count) +"/"+str(len(listAges)))
    print("num of babies: " + str(numBaby) + "/" + str(len(listAges)))

def normalize_ages2(age):
    return age//5*5
def normalize_ages(age):
    if age is None:
        return None
    """
        map age to group
        Infant: 0-1 yrs
        Toddler = 2 - 4 yrs
        Child = 5 - 12 yrs
        Teen = 13 - 19 yrs
        Adult = 20 - 39 yrs
        Middle Age Adult = 40 - 59 yrs
        Senior Adult = 60+
    """
    AGE_RANGES = [(0, 1, "0_Infant"), (1, 4, "1_Toddler"), (4, 12, "2_Child"), (12, 19, "3_Teen"), (19, 39, "4_Adult"),
                  (39, 64, "5_Middle"), (65, 1000, "6_65+")]
    for r in AGE_RANGES:
        if age<=r[1]:
            return r[2]

def show_Ages_Distribution_chart(listAges):
    #put ages to groups with range 5
    r5Ages = list(map(lambda a:a//5*5,listAges))
    r5AgesPopulation = {}
    for a in r5Ages:
        if r5AgesPopulation.get(a) is None:
            r5AgesPopulation.update({a:1})
        else: r5AgesPopulation[a]+=1

    # distribute ages into 7 groups
    g7Ages = list(map(lambda a: normalize_ages(a), listAges))
    g7AgesPopulation = {}
    for a in g7Ages:
        if g7AgesPopulation.get(a) is None:
            g7AgesPopulation.update({a:1})
        else: g7AgesPopulation[a]+=1
    g7AgesPopulation=dict(sorted(g7AgesPopulation.items()))

    plt.figure(figsize=(10, 5))
    # 1st chart
    plt.subplot(2, 1, 1)
    # creating the bar plot
    plt.bar(r5AgesPopulation.keys(), r5AgesPopulation.values(), color='blue',
            width=0.4)
    plt.xlabel("Ages")
    plt.ylabel("number of Passenger")
    plt.title("Age Distribution")

    # 2nd chart
    plt.subplot(2, 1, 2)

    plt.bar(g7AgesPopulation.keys(), g7AgesPopulation.values(), color='blue',
            width=0.4)

    plt.xlabel("Ages")
    plt.ylabel("number of Passenger")
    plt.title("Age Distribution")

    plt.show()

def normalize_fare(fare):
    if fare is None:
        return None
    return 200 if fare>=200 else int(fare)//5*5
def show_Fare_Distribution_chart(listFare):
    plt.figure(figsize=(10, 5))
    normListFare=list(map(lambda a: 200 if a>=200 else a // 2 * 2, listFare))
    farePopulation = Utils.group_list(normListFare)

    plt.bar(list(map(lambda x:x[0],farePopulation)), list(map(lambda x:x[1],farePopulation)), color='blue')
    plt.xlabel("Passenger Fare")
    plt.ylabel("number of Passenger")
    plt.title("Fare Distribution")
    plt.show()

def show_loss_Distribution_chart(listLoss):
    # loss in range [0,1] -> will be convert to range [0,100]
    plt.figure(figsize=(10, 5))
    normListLoss=list(map(lambda l: int(l*100)//2*2,listLoss))

    print("correct point = ")
    print(100*len(list(filter(lambda x:x<50,normListLoss)))/len(normListLoss))

    distribution = Utils.group_list(normListLoss)

    plt.bar(list(map(lambda x:x[0],distribution)), list(map(lambda x:x[1],distribution)), color='blue')
    plt.xlabel("loss in percentage %")
    plt.ylabel("distribution")
    plt.title("Loss function Distribution")
    plt.show()
def print_correlate_survival_age(df):
    df=df.dropna(subset=["Age","Survived"])
    df['Age']=df['Age'].apply(normalize_ages)
    #90% passenger >=65 yrs die
    PdHelper.print_full(df.groupby(['Age',"Survived"]).size())

def print_correlate_survival_cabin(df):
    cp_df=df.dropna(subset=["Cabin"])

    mapCabin={}
    for idx in cp_df.index:
        cabins=cp_df["Cabin"][idx].split(' ')
        sur=cp_df["Survived"][idx]
        for c in cabins:
            if c not in mapCabin:
                mapCabin.update({c: (0,0)}) # cabin suvival sum
            s,sum=mapCabin.get(c)
            sum+=1
            s+=sur
            mapCabin.update({c: (s, sum)})
    listCabin =[]
    for k,v in mapCabin.items():
        listCabin.append((k,v[0],v[1],v[0]*100/v[1]))
    listCabin.sort(key=lambda x:-x[3])
    print(len(listCabin))
    print("cabin,num-survival,sum,rate")
    for l in listCabin:
        print(l)
    # PdHelper.print_full(cp_df.groupby(['Cabin',"Survived"]).size())

def print_correlate_survival_fare(df):
    df=df.dropna(subset=["Fare","Survived"])
    # 95% passenger has fare =0 die
    df=df.groupby(['Fare', "Survived"]).size().reset_index(name="Size")
    df=df[df['Size']>=2].sort_values(by=['Size','Fare'],ascending=False)
    df['Rate']=100*df['Size']/df.groupby('Fare')['Size'].transform(np.sum)
    PdHelper.print_full(df)
    print(df[df['Rate']==100]['Fare'].tolist())

def print_correlate_survival_ticket(df):
    cp_df=df.dropna(subset=["Ticket","Survived"])

    gpList = Utils.group_list(cp_df["Ticket"].tolist())
    freqList=list(map(lambda x: x[1],gpList))
    grFreqList=Utils.group_list((freqList))
    print("(X,Y) DESCRIBE THAT THERE ARE Y TICKET NUMBER HAVING X PASSENGER")
    print(grFreqList)

    cp_df=cp_df.groupby(["Ticket","Survived"]).size().reset_index(name="Size")
    cp_df["Sum"]=cp_df.groupby(["Ticket"])["Size"].transform(np.sum)
    cp_df["Rate"]=100*cp_df["Size"]/cp_df.groupby("Ticket")["Size"].transform(np.sum)

    cp_df=cp_df[cp_df["Sum"]>=2].sort_values(by=['Rate','Ticket'],ascending=False).reset_index()
    PdHelper.print_full(cp_df)
    print(cp_df[cp_df['Rate']==100]['Ticket'].tolist())

def explore_correlate_survival_one_field(df):
    columns=["Pclass","Sex","Age","SibSp","Parch","Embarked","Fare"]

    for col in columns:
        cpDf = df.dropna(subset=[col]).copy()
        cpDf=cpDf.groupby([col,"Survived"]).size().reset_index(name='Size')
        cpDf['Rate'] = 100*cpDf['Size']/cpDf.groupby([col])["Size"].transform('sum')
        cpDf=cpDf.sort_values(by=['Size'], ascending=False)
        cpDf=cpDf[(cpDf['Size']>=10)]
        PdHelper.print_full(cpDf)

def explore_correlate_survival_two_fields(df):
    columns=["Pclass","Sex","Age","SibSp","Parch","Embarked","Fare"]

    for col1 in columns:
        for col2 in columns:
            if col1 <= col2:
                continue
            cpDf = df.dropna(subset=[col1,col2]).copy()
            cpDf=cpDf.groupby([col1,col2,"Survived"]).size().reset_index(name='Size')
            cpDf['Rate'] = 100*cpDf['Size']/cpDf.groupby([col1,col2])["Size"].transform('sum')
            cpDf=cpDf.sort_values(by=['Size'], ascending=False)
            cpDf=cpDf[(cpDf['Size']>=10) & (cpDf['Rate']>=EXPECTED_SCORE-5)]
            if len(cpDf)>0:
                PdHelper.print_full(cpDf)
    """
        some rules are found out from this function:
        1) ~100% passengers with fare<5 are from Southamton (Embarked = S) and no family members onboard (SibSp=0 & Parch=0),
            probably they all are seamens, Titanic employees = 95% die
        2) if a man go with his family, more likely he die 75%
        3) women with ticket start with prefix "PC" are 95% survival
        4) women with fare >=60 are 95% survival (there are 2 dead outliers with the same (cabin,fare,ticket) )
        5) Passengers having the same ticket (maybe in the same room) with size>=5 and pclass !=1 are ~100% dead together
        6) passengers in 6_65+ yrs likely die
        7) passengers having the same (ticket&fare) tend to die or alive together (not sure)
    """

def count_passenger_matching_pattern(df):
    cp_df=df.copy()
    cnt=0
    for idx in cp_df.index:
        isMatched=0
        if cp_df["Fare"][idx]<5:
            isMatched=1
        elif cp_df["Sex"][idx]=='male' and (cp_df["SibSp"][idx]>0 or cp_df["Parch"][idx]>0):
            isMatched=1
        elif cp_df["Sex"][idx]=='female' and cp_df["Ticket"][idx].startswith("PC"):
            isMatched=1
        elif cp_df["Sex"][idx]=='female' and cp_df["Fare"][idx]>=60:
            isMatched=1
        cnt+=isMatched
    return cnt

def calc_predict_prob(listscores):
    prob0=0.5
    prob1=0.5
    for (v,p) in listscores:
        if v ==1:
            prob1=1-(1-prob1)*(1-p)
        elif v==0:
            prob0=1-(1-prob0)*(1-p)
        else:
            assert(False)

    if prob0>=prob1:
        return 1-prob0
    else:
        return prob1

def filter_passengers_same_fare_ticket(df):
    def get_most_freq_values(df,colname,threshold_sum,threshold_rate):
        cp_df=df.copy()
        cp_df = cp_df.groupby([colname, "Survived"]).size().reset_index(name="Size")
        cp_df["Sum"] = cp_df.groupby([colname])["Size"].transform(np.sum)
        cp_df["Rate"] = 100 * cp_df["Size"] / cp_df.groupby(colname)["Size"].transform(np.sum)

        cp_df = cp_df[cp_df["Sum"] >= threshold_sum].sort_values(by='Sum', ascending=False).reset_index()
        return cp_df[cp_df['Rate'] >= threshold_rate][colname].tolist()

    mostFreqFareList = get_most_freq_values(df,'Fare',2,0)
    mostFreqTicketList = get_most_freq_values(df, 'Ticket', 2, 0)

    common_df = df[df['Fare'].isin(mostFreqFareList) & df['Ticket'].isin(mostFreqTicketList)]
    common_df = common_df.sort_values(by=['Ticket','Fare']).reset_index()
    return common_df

def get_mapTicket2Survial(df):
    ticketlist = ['W./C. 6608', 'S.O./P.P. 3', 'PP 9549', 'PC 17755', 'PC 17593', 'PC 17572', '367230', '364516',
                     '35281', '347742', '347054', '345773', '29106', '28403', '2678', '2668', '2666', '2665', '250649',
                     '250647', '250644', '239865', '113760', '110465']
    mapTicket2Survival = {}
    cp_df=df.copy()
    for idx in cp_df.index:
        if cp_df['Ticket'][idx] in (ticketlist):
            mapTicket2Survival.update({cp_df['Ticket'][idx]: cp_df['Survived'][idx]})
    return mapTicket2Survival

def get_mapRateSurvialOnTicket(df):
    cp_df = df.copy()
    cp_df["Ticket"] = cp_df["Ticket"].fillna("")
    res={}
    for idx in cp_df.index:
        if cp_df['Ticket'][idx] not in res:
            res.update({cp_df['Ticket'][idx]:(0,0)})    #ticket , num survival, sum ticket
        survival,sum=res.get(cp_df['Ticket'][idx])
        sum+=1
        survival+=cp_df['Survived'][idx]
        res.update({cp_df['Ticket'][idx]: (survival,sum)})
    return res

def get_mapRateSurvialOnCabin(df):
    cp_df = df.copy()
    cp_df["Cabin"] = cp_df["Cabin"].fillna("")

    mapCabin = {}
    for idx in cp_df.index:
        cabins = cp_df["Cabin"][idx].split(' ')
        sur = cp_df["Survived"][idx]
        for c in cabins:
            if c not in mapCabin:
                mapCabin.update({c: (0, 0)})  # cabin, num suvival, sum
            s, sum = mapCabin.get(c)
            sum += 1
            s += sur
            mapCabin.update({c: (s, sum)})
    return mapCabin


def map_data_2_score_vector(df):
    #the idea is if else manually like decision tree based on some rules
    # submission score = 0.73
    scoreVector= {}
    cp_df=df.copy()
    cp_df["Age"]=cp_df["Age"].fillna(20)
    cp_df["Fare"] = cp_df["Fare"].fillna(15)
    cp_df["Age"]=cp_df["Age"].apply(normalize_ages)
    cp_df["Fare"] = cp_df["Fare"].apply(normalize_fare)
    cp_df["SibSp"]=cp_df["SibSp"].fillna("")
    cp_df["Parch"] = cp_df["Parch"].fillna("")
    cp_df["Sex"] = cp_df["Sex"].fillna("")
    cp_df["Ticket"] = cp_df["Ticket"].fillna("")
    cp_df["Pclass"] = cp_df["Pclass"].fillna("")
    cp_df["Cabin"] = cp_df["Cabin"].fillna("")
    freqTicket=Utils.group_list(cp_df["Ticket"].tolist())
    mostFreqTicket=list(map(lambda x:x[0],filter(lambda t: t[1]>=5,freqTicket)))
    print("most frequence ticket number")
    print(mostFreqTicket)

    train_df=pd.read_csv("/home/vangtrangtan/Desktop/titanic/train.csv")
    mapTicket2Suvival = get_mapTicket2Survial(train_df)
    mapRateSurviveOnTicket =get_mapRateSurvialOnTicket(train_df)
    mapRateSurviveOnCabin=get_mapRateSurvialOnCabin(train_df)
    cnttic=0
    cntcabin=0
    cntpredictable=0
    cntunpredictable=0
    cntticandcabin=0
    for idx in cp_df.index:
        scores=[]
        if cp_df["Fare"][idx]<5:
            scores.append((0,0.9))
        if cp_df["Sex"][idx]=='female' and cp_df["Ticket"][idx].startswith("PC"):
            scores.append((1, 0.9))
        # if cp_df["Sex"][idx]=='female' and cp_df["Fare"][idx]>=60:
        #     scores.append((1, 0.9))
        if cp_df["Sex"][idx] == 'female' and cp_df["Fare"][idx] >= 20 and cp_df["Pclass"][idx]<=2:
            scores.append((1, 0.8))
        if cp_df["Age"][idx] == '6_65+':
            scores.append((0, 0.9))
        # if cp_df["Ticket"][idx] in (mostFreqTicket):
        #     scores.append((0,0.88))
        # if cp_df["SibSp"][idx]>=4:
        #     scores.append((0, 0.85))
        # if (cp_df["Sex"][idx]=='male') and cp_df["Pclass"][idx]==3:
        #     scores.append((0, 0.9))
        #(passengerID,list scores,tag predictable or not)
        if len(scores)>0:
            scoreVector.update({cp_df["PassengerId"][idx]:(scores,"predictable")})
            cntpredictable+=1
        else:
            cntunpredictable+=1

            if cp_df["Fare"][idx]>=55:
                scores.append((1,0.85))

            if cp_df["Sex"][idx]=="male":
                if cp_df["Pclass"][idx]==3:
                    scores.append((0,0.8))
            else:
               scores.append((1, 0.78))
            if cp_df["Pclass"][idx]==3:
                scores.append((0, 0.75))
            # if cp_df["Pclass"][idx]==1:
            #     scores.append((1, 0.7))

            if cp_df["Ticket"][idx] in mapRateSurviveOnTicket:
                tic=cp_df["Ticket"][idx]
                if cp_df["Cabin"][idx] != "" and cp_df["Cabin"][idx] in mapRateSurviveOnCabin:
                   cntticandcabin+=1
                sur,sum = mapRateSurviveOnTicket.get(tic)
                cnttic+=1
                if sur/sum<=0.5:
                    scores.append((0, 0.7))
                else:
                    scores.append((1, 0.7))

            if cp_df["Cabin"][idx] != "" and cp_df["Cabin"][idx] in mapRateSurviveOnCabin:
                cntcabin+=1
                cabins = cp_df["Cabin"][idx].split(' ')
                rates =[]
                for c in cabins:
                    if c in mapRateSurviveOnCabin:
                        x,y=mapRateSurviveOnCabin.get(c)
                        rates.append(x*100/y)
                rates.sort()
                if rates[0]<=25:
                    scores.append((0, 0.7))
                if rates[-1]>=75:
                    scores.append((1, 0.7))

            if len(scores)==0:
                if cp_df["Sex"][idx] == "male":
                    scores.append((0, 0.8))
                else:
                    scores.append((1, 0.78))

            scoreVector.update({cp_df["PassengerId"][idx]: (scores, "unpredictable")})
    #check result
    print("cuttic = " +str(cnttic))
    print("cntcabin = " + str(cntcabin))
    print("cntpredictable = " + str(cntpredictable))
    print("cntunpredictable = " + str(cntunpredictable))
    print("cntunpredictable cntticandcabin= " + str(cntticandcabin))
    return
    cnt=0
    v=[]
    for k in scoreVector:
        if scoreVector.get(k)[1]=="predictable":
            cnt+=1
            x=scoreVector.get(k)[0]
            x.sort()
            zipv=''
            for u in x:
                zipv+=str(u[0])
            v.append(zipv)
    gpv=Utils.group_list(v)
    print("non empty scores = "+str(cnt))
    print(gpv)

    finalScores={}

    listId=[]
    listPredict=[]
    for k,v in scoreVector.items():
        finalScores.update({k:(calc_predict_prob(v[0]),v[1])})
        listId.append(k)
        listPredict.append(0 if calc_predict_prob(v[0])<0.5 else 1)

    # # print result to submit
    # print("write to output csv")
    # output_df = pd.DataFrame({'PassengerId':listId,'Survived':listPredict})
    # PdHelper.write_df_to_csv(output_df,'/home/vangtrangtan/Desktop/titanic/res.csv')
    # return
    lossList=[]
    for idx in cp_df.index:
        loss=abs(finalScores.get(cp_df["PassengerId"][idx])[0]-cp_df["Survived"][idx])
        tag=finalScores.get(cp_df["PassengerId"][idx])[1]
        lossList.append((cp_df["PassengerId"][idx],loss,tag))

    predictable_loss=list(map(lambda p: p[1],filter(lambda v:v[2]=="predictable",lossList)))
    unpredictable_loss = list(map(lambda p: p[1], filter(lambda v: v[2] == "unpredictable", lossList)))
    all_loss=list(map(lambda p: p[1], lossList))


    show_loss_Distribution_chart(all_loss)


def normalize_df(cp_df):
    cp_df["Age"] = cp_df["Age"].fillna(20)
    cp_df["Fare"] = cp_df["Fare"].fillna(15)
    cp_df["Age"] = cp_df["Age"].apply(normalize_ages)
    cp_df["Age"] = cp_df["Age"].replace(
        {"0_Infant": 0, "1_Toddler": 1, "2_Child": 2, "3_Teen": 3, "4_Adult": 4, "5_Middle": 5, "6_65+": 6})
    cp_df["Fare"] = cp_df["Fare"].apply(normalize_fare)
    cp_df["SibSp"] = cp_df["SibSp"].fillna(0)
    cp_df["Parch"] = cp_df["Parch"].fillna(0)
    cp_df["Sex"] = cp_df["Sex"].fillna("male")
    cp_df["Sex"] = cp_df["Sex"].replace({"male": 0, "female": 1})
    # cp_df["Ticket"] = cp_df["Ticket"].fillna("")
    cp_df["Pclass"] = cp_df["Pclass"].fillna("")
    # cp_df["Cabin"] = cp_df["Cabin"].fillna("")

def calc_ticket_mates_survive(df):
    cp_df=df.copy()
    cp_df['Ticket'].dropna()
    survivemap={}   # Ticket: (num survive - num dead)
    for idx in cp_df.index:
        t=cp_df['Ticket'][idx]
        if t not in survivemap:
            survivemap.update({t:0})
        v=survivemap.get(t)
        v+= 1 if cp_df['Survived'][idx]==1 else -1
        survivemap.update({t:v})
    return survivemap

def fill_ticket_mates_survive_column(df,matesSurviveMap):
    df['Ticket']=df['Ticket'].fillna(0)
    listMatesSurvive=[]
    for id in df['Ticket'].tolist():
        if id in matesSurviveMap:
            listMatesSurvive.append(matesSurviveMap.get(id))
        else:
            listMatesSurvive.append(0)
    df=df.assign(TicketMateSurvive=listMatesSurvive)
    return df

def decisionTree(df):
    #the idea is that we split data into two sets:
    # - unpredictable-by-tree set which belongs to gini >=0.4 tree nodes
    # - predictable-by-tree set
    # unpredictable-by-tree set contains 75% data, we predict based on whether their ticket mates are survival or not, predict accuracy ~ 60%
    # predictable-by-tree set contains 25% data, train and predict by tree, predict accuracy ~ 90%
    # what's next ? try to improve accuracy of unpredictable-by-tree set (currently i have no idea hehe)

    isSUbmit = 0
    cp_df = df.copy()

    normalize_df(cp_df)
    ticketMateSuviveMap = calc_ticket_mates_survive(cp_df)
    cp_df1 = cp_df[(cp_df["Sex"] == 1) & (cp_df["Pclass"] == 3) & (cp_df["Fare"] <= 22.5)]
    cp_df2 = cp_df[(cp_df["Sex"] == 0) & (cp_df["Pclass"] == 1)]
    unpredictable_df=pd.concat([cp_df1,cp_df2])
    predictable_df=cp_df[~cp_df['PassengerId'].isin(unpredictable_df['PassengerId'].tolist())]
    print("len predictable "+ str(len(predictable_df)))
    print("len unpredictable " + str(len(unpredictable_df)))



    cp_df=predictable_df[['Age',"Fare","SibSp",'Parch','Pclass','Sex','Survived','Ticket']]
    target=cp_df["Survived"]
    X_train,X_test,target_train,target_test=train_test_split(cp_df,target,random_state=100,test_size=0.3 if not isSUbmit else 0.01)


    X_train = fill_ticket_mates_survive_column(X_train, ticketMateSuviveMap)
    X_test = fill_ticket_mates_survive_column(X_test, ticketMateSuviveMap)
    X_train=X_train.drop(['Ticket','Survived','TicketMateSurvive'],axis=1)
    X_test=X_test.drop(['Ticket','Survived','TicketMateSurvive'],axis=1)

    clf = DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=4,min_samples_split=15,random_state=100)
    clf.fit(X_train, target_train)
    # print(target_train)

    if isSUbmit:
        test_df=pd.read_csv('/home/vangtrangtan/Desktop/titanic/test.csv',usecols=['Age',"Fare","SibSp",'Parch','Pclass','Sex','PassengerId','Ticket'])
        test_df=fill_ticket_mates_survive_column(test_df, ticketMateSuviveMap)
        normalize_df(test_df)
        cp_df1 = test_df[(test_df["Sex"] == 1) & (test_df["Pclass"] == 3) & (test_df["Fare"] <= 22.5)]
        cp_df2 = test_df[(test_df["Sex"] == 0) & (test_df["Pclass"] == 1)]
        unpredictable_df = pd.concat([cp_df1, cp_df2])
        predictable_df = test_df[~test_df['PassengerId'].isin(unpredictable_df['PassengerId'].tolist())]
        print("len predictable " + str(len(predictable_df)))
        print("len unpredictable " + str(len(unpredictable_df)))
        # test_df.pop('Ticket')
        predictable_pIds=predictable_df.pop('PassengerId')
        predictable_test_df=predictable_df[['Age',"Fare","SibSp",'Parch','Pclass','Sex']]
        result = clf.predict(predictable_test_df)
        predictable_output_df=pd.DataFrame({'PassengerId':predictable_pIds.tolist(),'Survived':result})

        result2=[]
        cntall=0
        cntyes=0
        cntno=0
        for id in unpredictable_df.index:
            cntall+=1
            if unpredictable_df['Ticket'][id] not in ticketMateSuviveMap:
                result2.append(0)
            else:
                x=ticketMateSuviveMap.get(unpredictable_df['Ticket'][id])
                result2.append(0 if x<=0 else 1)
                if x<=0:
                    cntno+=1
                else:
                    cntyes+=1
        print("all="+str(cntall)+"yes"+str(cntyes)+"no"+str(cntno))
        unpredictable_output_df=pd.DataFrame({'PassengerId':unpredictable_df['PassengerId'].tolist(),'Survived':result2})
        all_output =pd.concat([unpredictable_output_df,predictable_output_df])
        all_output=all_output.sort_values(by='PassengerId')
        # PdHelper.print_full(all_output)
        print("write to output csv")
        PdHelper.write_df_to_csv(all_output,'/home/vangtrangtan/Desktop/titanic/res.csv')
    else:
        print(cp_df.head(10))
        predictions = clf.predict(X_test)
        print(accuracy_score(target_test, predictions))
        # print(predictions)

    text_representation = tree.export_text(clf)
    print(text_representation)

    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(clf,
                       feature_names=cp_df.columns,
                       class_names=['Dead','Alive'],
                       filled=True)
    plt.show()


main_df = pd.read_csv("/home/vangtrangtan/Desktop/titanic/train.csv")
cp_df=main_df.copy()
PdHelper.print_all_column_stats(main_df)
cp_df=main_df.copy()
# 1st approach, submission score = 0.73
# map_data_2_score_vector(cp_df)

# 2nd approach, submission score = 0.78
decisionTree(cp_df)
