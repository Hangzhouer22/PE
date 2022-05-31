#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Library to suppress warnings 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np                  # Scientific Computing
import pandas as pd                 # Data Analysis
import matplotlib.pyplot as plt     # Plotting
import seaborn as sns               # Statistical Data Visualization


# In[3]:


matches  = pd.read_csv('/Users/yutaoyan/Desktop/Worldcupplayers/WorldCupMatches.csv')
players  = pd.read_csv('/Users/yutaoyan/Desktop/Worldcupplayers/WorldCupPlayers.csv')
cups     = pd.read_csv('/Users/yutaoyan/Desktop/Worldcupplayers/WorldCups.csv')


# In[4]:


matches.head()


# In[5]:


players.head(7)


# In[6]:


cups.tail(7)


# In[7]:


# display the dimension of the cups data
cups.shape


# In[8]:


players.info()


# In[9]:


matches.dtypes


# In[10]:


players.describe()


# In[11]:


matches.describe()


# In[12]:


cups.describe()


# In[13]:


# describe the categorical data
cups.describe(include=object)
# Note: If we do not pass include=object to the describe(), it would return statistics for numeric variable


# In[14]:


cups.isnull().sum()


# In[15]:


players.isnull().sum()


# In[17]:


def missing(players):
    print (round((players.isnull().sum() * 100/ len(players)),2).sort_values(ascending=False))

missing(players)


# In[18]:


players["Position"].value_counts()


# In[19]:


players["Position"].replace(' ', np.NaN).head(3)


# In[20]:


df = players['Position'].value_counts().index[0]


# In[21]:


players['Position'].fillna(df,inplace=True)
players["Position"].head()


# In[22]:


players["Event"].replace(' ', np.NaN).head(3)


# In[23]:


df1 = players['Event'].value_counts().index[0]
players['Event'].fillna(df1,inplace=True)
players["Event"].head(2)


# In[24]:


def missing(players):
    print (round((players.isnull().sum() * 100/ len(players)),2).sort_values(ascending=False))

missing(players)


# In[25]:


cups.corr()


# In[26]:


# To get a correlation matrix
# Ploting correlation plot
f,ax = plt.subplots(figsize=(10, 10))

# plotting the heat map
# corr: give the correlation matrix
# cmap: colour code used for plotting
# vmax: gives maximum range of values for the chart
# vmin: gives minimum range of values for the chart
# annot: prints the correlation values in the chart
# annot_kws={"size": 12}): Sets the font size of the annotation

sns.heatmap(cups.corr(), annot=True, linewidths=.5, fmt= '.4f',ax=ax)
# specify name of the plot
plt.title('Correlation between features')
plt.show()


# In[27]:


matches.boxplot(column='Home Team Goals',by = 'Away Team Goals')


# In[28]:


matches["total_goals"] = matches["Home Team Goals"] + matches["Away Team Goals"]

plt.figure(figsize=(13,8))
sns.boxplot(y=matches["total_goals"],
            x=matches["Year"])
plt.grid(True)
plt.title("Total goals scored during game by year",color='b')
plt.show()


# In[29]:


Q1 = matches.quantile(0.25)
Q3 = matches.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[30]:


matches = matches[~((matches < (Q1 - 1.5 * IQR)) |(matches > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[31]:


matches.boxplot(column='Home Team Goals',by = 'Away Team Goals')


# In[32]:


matches["total_goals"] = matches["Home Team Goals"] + matches["Away Team Goals"]

plt.figure(figsize=(13,8))
sns.boxplot(y=matches["total_goals"],
            x=matches["Year"])
plt.grid(True)
plt.title("Total goals scored during game by year",color='b')
plt.show()


# In[33]:


sns.distplot(cups.Year)


# In[34]:


plt.figure(figsize=(13, 4))
sns.countplot(cups.Country, order = cups.Country.value_counts().index);


# In[35]:


plt.figure(figsize=(14, 14))

sns.pairplot(cups, diag_kind='kde');


# In[36]:


cups["AverageGoal"] = cups.GoalsScored/cups.MatchesPlayed


# In[37]:


cups.plot(kind='scatter', x='Year', y="GoalsScored",alpha = .8,color = 'blue',figsize= (6,6))
plt.xlabel('Year')             
plt.ylabel("GoalsScored")
plt.title('Scatter Plot') 
plt.show()


# In[38]:


ax = plt.gca()

cups.plot(kind='line', x = "Year",y = "GoalsScored", color = "green", ax=ax,grid = True,figsize = (7,7))
cups.plot(kind='line', x = "Year",y = "MatchesPlayed", color = 'red', ax=ax,grid = True)
cups.plot(kind='line', x = "Year",y = "QualifiedTeams", color = 'b', ax=ax,grid = True)
plt.legend(loc = "upper left")
plt.show()


# In[39]:


import seaborn as sns

sns.lmplot(x='Year', y='GoalsScored', hue='Country', 
           data=cups.loc[cups['Country'].isin(['Uruguay', 'Italy', 'France', 'Brazil', 'Switzerland', 'Chile', 'England', 'Mexico', 'Germany', 'Argentina', 'Spain', 'Mexico'])], 
           fit_reg=False)


# In[40]:


f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cups.corr(), annot=True, linewidths=.5, fmt= '.4f',ax=ax)
plt.show()


# In[41]:


cups[np.logical_and(cups['GoalsScored']>100, cups['Year']>1930 )]


# In[42]:


def cup(count=5):
    """returns a list of top (count) cups that finished with the highest amount of goals (default:5)"""
    ecg=cups.sort_values(by=['GoalsScored'],ascending=False).head(count)
    return ecg
cup()


# In[43]:


def goals(country=16):
    Country=cups.at[country-1,'Country']
    year=cups.at[country-1,'Year']
    goals=cups.at[country-1,'GoalsScored']
    matches=cups.at[country-1,'MatchesPlayed']
    def AvgGoal(goals, matches):
        avg=goals/matches
        return avg
    print(Country, year)
    print("Average goal per match:",AvgGoal(goals,matches))
goals()


# In[44]:


def ulkeler(*args):
    for i in args:
        print(i)
countries=tuple(cups.iloc[:, cups.columns.get_loc('Country')])
ulkeler(countries)

dict=cups.set_index('Country').to_dict()['Winner']
def f(**kwargs):
    for key, value in kwargs.items():
        print(key, " ", value)
    
f(**dict)


# In[45]:


cups.Attendance = cups.Attendance.astype(str)
cups.Attendance = [c.replace('.', '') for c in cups.Attendance]
cups.Attendance = cups.Attendance.astype(int)


# In[46]:


matches.columns = [c.replace(' ', '') for c in matches.columns]
matches.columns = [c.replace('-', '') for c in matches.columns]
matches.columns


# In[47]:


cups_data1 = cups.copy()


# In[48]:


cups_data1.head(2)


# In[49]:


cups_data = cups_data1.set_index(["Country","Winner"])
cups_data.head(4)


# In[50]:


cups_data1["GoalMean"]=[round(cups_data1.GoalsScored[i]/cups_data1.MatchesPlayed[i],2) for i in range(len(cups_data1.GoalsScored))]
cups_data1.head()


# In[51]:


print(cups_data1.index.name)
cups_data1.index.name="IndexName"
cups_data1.head(3)


# In[52]:


cups_data2 = cups_data1.set_index(["Winner","Year"]) 
cups_data2.head(3)


# In[53]:


cups_data3=cups_data1.set_index(["Winner","GoalMean"])
cups_data3.head()


# In[54]:


cups_data3.groupby("GoalsScored").mean().head()


# In[55]:


plt.figure(figsize=(13,7))
cups["Year1"] = cups["Year"].astype(str)
ax = plt.scatter("Year1","GoalsScored",data=cups,
            c=cups["GoalsScored"],cmap="inferno",
            s=900,alpha=.7,
            linewidth=2,edgecolor="k")

#plt.colorbar()
plt.xticks(cups["Year1"].unique())
plt.yticks(np.arange(60,200,20))
plt.title('Total goals scored by year',color='b')
plt.xlabel("year")
plt.ylabel("total goals scored")
plt.show()


# In[56]:


plt.figure(figsize=(12,7))

sns.barplot(cups["Year"],cups["MatchesPlayed"],linewidth=1,
            edgecolor=[c for c in "k"*len(cups)],color="b",label="Total matches played")

sns.barplot(cups["Year"],cups["QualifiedTeams"],linewidth=1,
            edgecolor=[c for c in "k"*len(cups)],color="r",label="Total qualified teams")

plt.legend(loc="best",prop={"size":13})
plt.title("Qualified teams by year",color='b')
plt.grid(True)
plt.ylabel("total matches and qualified teams by year")
plt.show()


# In[57]:


matches.dropna().head(2)


# In[58]:


h_att = matches.sort_values(by="Attendance",ascending=False)[:10]
h_att = h_att[['Year', 'Datetime','Stadium', 'City', 'HomeTeamName',
              'HomeTeamGoals', 'AwayTeamGoals', 'AwayTeamName', 'Attendance', 'MatchID']]
h_att["Stadium"] = h_att["Stadium"].replace('Maracan� - Est�dio Jornalista M�rio Filho',"Maracanã Stadium")
h_att["Datetime"] = h_att["Datetime"].str.split("-").str[0]
h_att["mt"] = h_att["HomeTeamName"] + " .Vs.  " + h_att["AwayTeamName"]

plt.figure(figsize=(10,9))
ax = sns.barplot(y =h_att["mt"],x = h_att["Attendance"],palette="gist_ncar",
                 linewidth = 1,edgecolor=[c for c in "k"*len(h_att)])
plt.ylabel("match teams")
plt.xlabel("Attendance")
plt.title("Matches with highest number of attendace",color='b')
plt.grid(True)
for i,j in enumerate(" stadium : "+h_att["Stadium"]+" , Date :" + h_att["Datetime"]):
    ax.text(.7,i,j,fontsize = 12,color="white",weight = "bold")
plt.show()


# In[59]:


mat_c = matches["City"].value_counts().reset_index()
plt.figure(figsize=(10,8))
ax = sns.barplot(y=mat_c["index"][:15],x = mat_c["City"][:15],palette="plasma",
                 linewidth=1,edgecolor=[c for c in "k"*15])
plt.xlabel("number of matches")
plt.ylabel("City")
plt.grid(True)
plt.title("Cities with maximum world cup matches",color='b')

for i,j in enumerate("Matches  :" + mat_c["City"][:15].astype(str)):
    ax.text(.7,i,j,fontsize = 13,color="w")
plt.show()


# In[60]:


ct_at = matches.groupby("City")["Attendance"].mean().reset_index()
ct_at = ct_at.sort_values(by="Attendance",ascending=False)
ct_at

plt.figure(figsize=(10,10))

ax = sns.barplot("Attendance","City",
            data=ct_at[:20],
            linewidth = 1,
            edgecolor = [c for c in "k"*20],
            palette  = "Spectral_r")

for i,j in enumerate(" Average attendance  : "+np.around(ct_at["Attendance"][:20],0).astype(str)):
    ax.text(.7,i,j,fontsize=12)
plt.grid(True)

plt.title("Average attendance by city",color='b')
plt.show()


# In[61]:


cups["Winner"]=cups["Winner"].replace("Germany FR","Germany")
cups["Runners-Up"]=cups["Runners-Up"].replace("Germany FR","Germany")
c1  = cups.groupby("Winner")["Year1"].apply(" , ".join).reset_index()
c2  = cups.groupby("Winner")['Year'].count().reset_index()
c12 = c1.merge(c2,left_on="Winner",right_on="Winner",how="left")
c12 = c12.sort_values(by = "Year",ascending =False)

plt.figure(figsize=(10,8))
ax = sns.barplot("Year","Winner",data=c12,
            palette="jet_r",
            alpha=.8,
            linewidth=2,
            edgecolor=[c for c in "k"*len(c12)])
for i,j in enumerate("Years : " + c12["Year1"]):
    ax.text(.1,i,j,weight = "bold")

plt.title("Teams with the most world cup final victories")
plt.grid(True)
plt.xlabel("count")
plt.show()


# In[62]:


cou = cups["Winner"].value_counts().reset_index()
cou_w = cou.copy()
cou_w.columns = ["country","count"]
cou_w["type"] = "WINNER"

cou_r = cups["Runners-Up"].value_counts().reset_index()
cou_r.columns = ["country","count"]
cou_r["type"] = "RUNNER - Up"

cou_t = pd.concat([cou_w,cou_r],axis=0)

plt.figure(figsize=(8,10))
sns.barplot("count","country",data=cou_t,
            hue="type",palette=["lime","r"],
            linewidth=1,edgecolor=[c for c in "k"*len(cou_t)])
plt.grid(True)
plt.legend(loc="center right",prop={"size":14})
plt.title("Final results by nation",color='b')
plt.show()


# In[63]:


def label(matches):
    if matches["HomeTeamGoals"] > matches["AwayTeamGoals"]:
        return "Home team win"
    if matches["AwayTeamGoals"] > matches["HomeTeamGoals"]:
        return "Away team win"
    if matches["HomeTeamGoals"] == matches["AwayTeamGoals"]:
        return "DRAW"

matches["outcome"] = matches.apply(lambda matches:label(matches),axis=1)
plt.figure(figsize=(9,9))
matches["outcome"].value_counts().plot.pie(autopct="%1.0f%%",fontsize =14,
                                           colors = sns.color_palette("husl"),
                                           wedgeprops={"linewidth":2,"edgecolor":"white"},
                                           shadow=True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("# Match outcomes by home and away teams",color='b')
plt.show()


# In[ ]:




