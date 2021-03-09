# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:09:08 2020

@author: Yi Ming
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None
plt.style.use('ggplot')

champs = pd.read_csv('E:\Onedrive\OneDrive - Arizona State University\桌面\LOL_Project\LOL\champs.csv')
champs.shape

participants = pd.read_csv('E:\Onedrive\OneDrive - Arizona State University\桌面\LOL_Project\LOL\participants.csv')
participants.shape

matches = pd.read_csv('E:\Onedrive\OneDrive - Arizona State University\桌面\LOL_Project\LOL\matches.csv')
matches.shape

stats1 = pd.read_csv('E:\Onedrive\OneDrive - Arizona State University\桌面\LOL_Project\LOL\stats1.csv')
stats1.shape

stats2 = pd.read_csv('E:\Onedrive\OneDrive - Arizona State University\桌面\LOL_Project\LOL\stats2.csv')
stats2.shape

stats = stats1.append(stats2)
stats.shape

# =============================================================================
# Data Cleaning
# =============================================================================
df = pd.merge(participants, stats, how = 'left', on = ['id'], suffixes=('', '_y'))
df = pd.merge(df, champs, how = 'left', left_on = 'championid', right_on = 'id', suffixes=('', '_y'))
df = pd.merge(df, matches, how = 'left', left_on = 'matchid', right_on = 'id', suffixes=('', '_y'))

def final_position(row):
    if row['role'] in ('DUO_SUPPORT', 'DUO_CARRY'):
        return row['role']
    else:
        return row['position']

df['adjposition'] = df.apply(final_position, axis = 1) 

df['team'] = df['player'].apply(lambda x: '1' if x <= 5 else '2')
df['team_role'] = df['team'] + ' - ' + df['adjposition']

# remove matchid with duplicate roles, e.g. 3 MID in same team, etc
remove_index = []
for i in ('1 - MID', '1 - TOP', '1 - DUO_SUPPORT', '1 - DUO_CARRY', '1 - JUNGLE', '2 - MID', '2 - TOP', '2 - DUO_SUPPORT', '2 - DUO_CARRY', '2 - JUNGLE'):
    df_remove = df[df['team_role'] == i].groupby('matchid').agg({'team_role':'count'})
    remove_index.extend(df_remove[df_remove['team_role']!=1].index.values)
    
# remove unclassified BOT, correct ones should be DUO_SUPPORT OR DUO_CARRY
remove_index.extend(df[df['adjposition'] == 'BOT']['matchid'].unique())
remove_index = list(set(remove_index))

print('# matches in dataset before cleaning: {}'.format(df['matchid'].nunique()))
df = df[~df['matchid'].isin(remove_index)]
print('# matches in dataset after cleaning: {}'.format(df['matchid'].nunique()))
# matches in dataset before cleaning: 184069
# matches in dataset after cleaning: 148638
df = df[['id', 'matchid', 'player', 'name', 'adjposition', 'team_role', 'win', 'kills', 'deaths', 'assists', 'turretkills','totdmgtochamp', 'totheal', 'totminionskilled', 'goldspent', 'totdmgtaken', 'inhibkills', 'pinksbought', 'wardsplaced', 'duration', 'platformid', 'seasonid', 'version']]
df.head(10)
# =============================================================================
# Ward Place win or loss 
# =============================================================================
df_v = df.copy()
# put upper and lower limit
df_v['wardsplaced'] = df_v['wardsplaced'].apply(lambda x: x if x<30 else 30)
df_v['wardsplaced'] = df_v['wardsplaced'].apply(lambda x: x if x>0 else 0)

plt.figure(figsize = (15,10))
sns.violinplot(x="seasonid", y="wardsplaced", hue="win", data=df_v, split=True, inner = 'quartile')
plt.title('Wardsplaced by season: win vs loss')
#We can see that players placed more wards in recent seasons, no matter win or loss. Wards play a more important part nowadays

# =============================================================================
# Correlations win vs factor
# =============================================================================
df_corr = df._get_numeric_data()
df_corr = df_corr.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

mask = np.zeros_like(df_corr.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(10, 150, as_cmap=True)

plt.figure(figsize = (15,10))
sns.heatmap(df_corr.corr(), cmap = cmap, annot = True, fmt = '.2f', mask = mask, square=True, linewidths=.5, center = 0)
plt.title('Correlations - win vs factors (all games)')

# =============================================================================
# Less Than 25mins Correlation
# =============================================================================
df_corr_2 = df._get_numeric_data()
# for games less than 25mins
df_corr_2 = df_corr_2[df_corr_2['duration'] <= 1500]
df_corr_2 = df_corr_2.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

mask = np.zeros_like(df_corr_2.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(10, 150, as_cmap=True)

plt.figure(figsize = (15,10))
sns.heatmap(df_corr_2.corr(), cmap = cmap, annot = True, fmt = '.2f', mask = mask, square=True, linewidths=.5, center = 0)
plt.title('Correlations - win vs factors (for games last less than 25 mins)')
# =============================================================================
# More than 40mins 
# =============================================================================
df_corr_3 = df._get_numeric_data()
# for games more than 40mins
df_corr_3 = df_corr_3[df_corr_3['duration'] > 2400]
df_corr_3 = df_corr_3.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

mask = np.zeros_like(df_corr_3.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(10, 150, as_cmap=True)

plt.figure(figsize = (15,10))
sns.heatmap(df_corr_3.corr(), cmap = cmap, annot = True, fmt = '.2f', mask = mask, square=True, linewidths=.5, center = 0)
plt.title('Correlations - win vs factors (for games last more than 40 mins)')

# =============================================================================
# KDA vs Win rate Top&Bot 10 champs
# =============================================================================
pd.options.display.float_format = '{:,.1f}'.format

df_win_rate = df.groupby('name').agg({'win': 'sum', 'name': 'count', 'kills': 'mean', 'deaths': 'mean', 'assists': 'mean'})
df_win_rate.columns = ['win matches', 'total matches', 'K', 'D', 'A']
df_win_rate['win rate'] = df_win_rate['win matches'] /  df_win_rate['total matches'] * 100
df_win_rate['KDA'] = (df_win_rate['K'] + df_win_rate['A']) / df_win_rate['D']
df_win_rate = df_win_rate.sort_values('win rate', ascending = False)
df_win_rate = df_win_rate[['total matches', 'win rate', 'K', 'D', 'A', 'KDA']]
print('Top 10 win rate')
print(df_win_rate.head(10))
print('Bottom 10 win rate')
print(df_win_rate.tail(10))

# =============================================================================
# scatter plot
# =============================================================================
df_win_rate.reset_index(inplace = True)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))

df_win_rate['color map'] = df_win_rate['win rate'].apply(lambda x: 'green' if x > 50 else 'red')

ax = df_win_rate.plot(kind = 'scatter', x = 'total matches', y = 'win rate', color = df_win_rate['color map'].tolist(), figsize = (15,10), title = 'win rate vs # matches by champions')

label_point(df_win_rate['total matches'], df_win_rate['win rate'], df_win_rate['name'], ax)
# =============================================================================
# win rate with role
# =============================================================================
pd.options.display.float_format = '{:,.1f}'.format

df_win_rate_role = df.groupby(['name','adjposition']).agg({'win': 'sum', 'name': 'count', 'kills': 'mean', 'deaths': 'mean', 'assists': 'mean'})
df_win_rate_role.columns = ['win matches', 'total matches', 'K', 'D', 'A']
df_win_rate_role['win rate'] = df_win_rate_role['win matches'] /  df_win_rate_role['total matches'] * 100
df_win_rate_role['KDA'] = (df_win_rate_role['K'] + df_win_rate_role['A']) / df_win_rate_role['D']
df_win_rate_role = df_win_rate_role.sort_values('win rate', ascending = False)
df_win_rate_role = df_win_rate_role[['total matches', 'win rate', 'K', 'D', 'A', 'KDA']]

# occur > 0.01% of matches
df_win_rate_role = df_win_rate_role[df_win_rate_role['total matches'] > df_win_rate_role['total matches'].sum()*0.0001]
print('Top 10 win rate with role (occur > 0.01% of total # matches)')
print(df_win_rate_role.head(10))
print('Bottom 10 win rate with role (occur > 0.01% of total # matches)')
print(df_win_rate_role.tail(10))
# =============================================================================
# get matchup
# =============================================================================
df_2 = df.sort_values(['matchid','adjposition'], ascending = [1,1])

df_2['shift 1'] = df_2['name'].shift()
df_2['shift -1'] = df_2['name'].shift(-1)

def get_matchup(x):
    if x['player'] <= 5:
        if x['name'] < x['shift -1']:
            name_return = x['name'] + ' vs ' + x['shift -1']
        else:
            name_return = x['shift -1'] + ' vs ' + x['name']
    else:
        if x['name'] < x['shift 1']:
            name_return = x['name'] + ' vs ' + x['shift 1']
        else:
            name_return = x['shift 1'] + ' vs ' + x['name']
    return name_return

df_2['match up'] = df_2.apply(get_matchup, axis = 1)
df_2['win_adj'] = df_2.apply(lambda x: x['win'] if x['name'] == x['match up'].split(' vs')[0] else 0, axis = 1)

df_2.head(10)
# =============================================================================
# Top 5 counter
# =============================================================================
print()
print()
df_matchup = df_2.groupby(['adjposition', 'match up']).agg({'win_adj': 'sum', 'match up': 'count'})
df_matchup.columns = ['win matches', 'total matches']
df_matchup['total matches'] = df_matchup['total matches'] / 2
df_matchup['win rate'] = df_matchup['win matches'] /  df_matchup['total matches']  * 100
df_matchup['counter score'] = df_matchup['win rate'] - 50
df_matchup['counter score (ND)'] = abs(df_matchup['counter score'])
df_matchup = df_matchup[df_matchup['total matches'] > df_matchup['total matches'].sum()*0.0001]

df_matchup = df_matchup.sort_values('counter score (ND)', ascending = False)
df_matchup = df_matchup[['total matches', 'counter score']]                   
df_matchup = df_matchup.reset_index()

print('counter score +/- means first/second champion dominant:')

for i in df_matchup['adjposition'].unique(): 
        print('\n{}:'.format(i))
        print(df_matchup[df_matchup['adjposition'] == i].iloc[:,1:].head(5))
# =============================================================================
# 预测counter
# =============================================================================

#def get_best_counter(champion, role):
 #   df_matchup_temp = df_matchup[(df_matchup['match up'].str.contains(champion)) & (df_matchup['adjposition'] == role)]
  #  df_matchup_temp['champion'] = df_matchup_temp['match up'].apply(lambda x: x.split(' vs ')[0] if x.split(' vs ')[1] == champion else x.split(' vs ')[1])
  # df_matchup_temp['advantage'] = df_matchup_temp.apply(lambda x: x['counter score']*-1 if x['match up'].split(' vs ')[0] == champion else x['counter score'], axis = 1)
  #  df_matchup_temp = df_matchup_temp[df_matchup_temp['advantage']>0].sort_values('advantage', ascending = False)
  #  print('Best counter for {} - {}:'.format(role, champion))
  #  print(df_matchup_temp[['champion', 'total matches', 'advantage']])
  #  return


#champion = 'Jayce'
#role = 'TOP'
#get_best_counter(champion, role)


# =============================================================================
# 
# =============================================================================
# pivot data to what we want in format

df_3 = df[['matchid', 'player', 'name', 'team_role', 'win']]

df_3 = df_3.pivot(index = 'matchid', columns = 'team_role', values = 'name')
df_3 = df_3.reset_index()
df_3 = df_3.merge(df[df['player'] == 1][['matchid', 'win']], left_on = 'matchid', right_on = 'matchid', how = 'left')
df_3 = df_3[df_3.columns.difference(['matchid'])]
df_3 = df_3.rename(columns = {'win': 'T1 win'})

df_3.head(10)
# =============================================================================
# 
# =============================================================================

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn.model_selection import train_test_split

# remove missing data
print('Before drop missing data: {}'.format(len(df_3)))
df_3 = df_3.dropna()
print('After drop missing data: {}'.format(len(df_3)))

y = df_3['T1 win']
X = df_3[df_3.columns.difference(['T1 win'])]

# label string to numeric
le_t = X.apply(le.fit)
X_t_1 = X.apply(le.fit_transform)

enc = preprocessing.OneHotEncoder()
enc_t = enc.fit(X_t_1)
X_t_2 = enc_t.transform(X_t_1)

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_t_1, y, random_state=0)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_t_2, y, random_state=0)

# =============================================================================
# 
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

import seaborn as sns

print('Accuracy on dataset converted from label to integer category:')

clf_lr = LogisticRegression(random_state = 0).fit(X_train_1, y_train_1)
acc_lr = clf_lr.score(X_test_1, y_test_1)
print('logistic regression : {}'.format(acc_lr))

clf_bnb = BernoulliNB().fit(X_train_1, y_train_1)
acc_bnb = clf_bnb.score(X_test_1, y_test_1)
print('naive bayes : {}'.format(acc_bnb))

print('\n')

# category with just 0 / 1, no magnitude meaning in category like above approach
print('Accuracy on dataset converted from label to binary category:')

clf_lr = LogisticRegression(random_state = 0).fit(X_train_2, y_train_2)
acc_lr = clf_lr.score(X_test_2, y_test_2)
print('logistic regression : {}'.format(acc_lr))

clf_bnb = BernoulliNB().fit(X_train_2, y_train_2)
acc_bnb = clf_bnb.score(X_test_2, y_test_2)
print('naive bayes : {}'.format(acc_bnb))






#Predictive Model
def get_best_counter(champion, role):
    df_matchup_temp = df_matchup[(df_matchup['match up'].str.contains(champion)) & (df_matchup['adjposition'] == role)]
    df_matchup_temp['champion'] = df_matchup_temp['match up'].apply(lambda x: x.split(' vs ')[0] if x.split(' vs ')[1] == champion else x.split(' vs ')[1])
    df_matchup_temp['advantage'] = df_matchup_temp.apply(lambda x: x['counter score']*-1 if x['match up'].split(' vs ')[0] == champion else x['counter score'], axis = 1)
    df_matchup_temp = df_matchup_temp[df_matchup_temp['advantage']>0].sort_values('advantage', ascending = False)
    print('Best counter-picks for {} - {}:'.format(role, champion))
    print(df_matchup_temp[['champion', 'total matches', 'advantage']])
    #print(type(df_matchup_temp))
    #csv 
    df_matchup_temp.to_csv(r'C:\ExportCSV\counterpick_Jayce_TOP.csv', index = False)
    return



champion = 'Jayce'
role = 'TOP'
get_best_counter(champion, role)













