# coding: utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import lightgbm as lgb
import pymorphy2
import re

load = pd.read_csv('val.csv')
test = pd.read_csv(r'task-for-hiring-data/test_data.csv')
with open("stop_words_ru.txt", encoding='utf-8') as f:
    stop_words = [x.replace("\n", "") for x in f]
    
y = load.is_bad
data = pd.DataFrame(load.loc[:, ('price')])
data_test = pd.DataFrame(test.loc[:, ('price')])

for df in [load, test]:
    df.datetime_submitted = pd.to_datetime(df.datetime_submitted)
    df['weekday_submitted'] = df.datetime_submitted.dt.weekday
    df['month_submitted'] = df.datetime_submitted.dt.month
    df['year_submitted'] = df.datetime_submitted.dt.year
    d = pd.get_dummies(df[['subcategory', 'category', 'region', 'city', 'weekday_submitted', 'month_submitted', 
                         'year_submitted']].astype('str'))
    df[d.columns] = d

for column in data.columns:
    if column not in data_test.columns:
        df[column] = 0 

morph = pymorphy2.MorphAnalyzer()

def lemmatize(text):
    words = text.split()
    res = list()
    int_list = []
    for word in words:
        if any(i.isdigit() for i in word) and len(word) > 3:
            int_word = ''
            for w in word:
                if w.isdigit():
                    int_word = int_word + w 
            int_list.append(int_word)
        p = morph.parse(word)[0]
        res.append(p.normal_form)
    res = res + int_list
    res = [i for i in res if i not in stop_words]
    res = ' '.join(res)
    return res        

def del_s(_text):
    punctuation = "!#$%^&*()_+<>?:.,;/-"    
    for c in _text:
        if c in punctuation:
            _text = _text.replace(c, "")
    return _text

def find_(string, rule):
    if re.search(rule, string):
        return 1
    else:
        return 0

def count_int(text):
    counter = 0
    for t in text:
        if t.isdigit():
            counter += 1
    return counter
    
tel = r'[\+]?[(-]?[0-9]{3}[-)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
sabaka = '@'

for df in [load, test]:
    df['new_description'] = df.description.apply(lambda x: del_s(x))
    df['new_description'] = df['new_description'].apply(lambda x: lemmatize(x))
    df['telephone'] = df['new_description'].apply(lambda x: find_(x, tel))
    df['@'] = df['new_description'].apply(lambda x: find_(x, sabaka))
    df['count_int'] = df['new_description'].apply(lambda x: count_int(x))
    
data = data.join(load.loc[:, (load.columns[9:])])
data_test = data_test.join(test.loc[:, (load.columns[9:])])

tfidf = TfidfVectorizer(max_features=5000)
values = tfidf.fit_transform(data.new_description)
feature_names = tfidf.get_feature_names()
data = data.join(pd.DataFrame(values.toarray(), columns=feature_names))

values_test = tfidf.transform(data_test.new_description)
data_test = data_test.join(pd.DataFrame(values_test.toarray(), columns=feature_names))
train = data.drop('new_description', axis=1).fillna(0)
data_test = data_test.drop('new_description', axis=1).fillna(0)
data_test = data_test.loc[:, (train.columns)] 

params = {'boosting_type': 'dart', 'class_weight': None, 'colsample_bytree': 0.7524080214748146, 'eval_metric': 'auc', 
          'learning_rate': 0.01778279410038923, 'max_depth': 49, 'min_child_samples': 5, 'n_estimators': 1150, 
          'num_leaves': 85, 'reg_alpha': 0.003359818286283781, 'reg_lambda': 0.12742749857031335, 
          'subsample': 0.95, 'subsample_freq': 6}
clf = lgb.LGBMClassifier(n_jobs=4, objective='binary', **params)
clf.fit(train.values, y)

if __name__ == '__main__':
    target_prediction = pd.DataFrame()
    target_prediction['index'] = range(data_test.shape[0])
    target_prediction['prediction'] = [x[1] for x in clf.predict_proba(data_test.values)]
    
    mask_prediction = pd.DataFrame()
    mask_prediction['index'] = range(test.shape[0])
    mask_prediction['start'] = None
    mask_prediction['end'] = None

    target_prediction.to_csv('task-for-hiring-data/target_prediction.csv', index=False)
    mask_prediction.to_csv('task-for-hiring-data/mask_prediction.csv', index=False)
