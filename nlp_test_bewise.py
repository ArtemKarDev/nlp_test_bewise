## Подготовка данных

import numpy as np
import pandas as pd
import re

# для векторизации текста
from sklearn.feature_extraction.text import CountVectorizer
# загрузим библиотеку для расчетов похожести
from sklearn.metrics.pairwise import cosine_similarity

get_ipython().system('pip install pymorphy2')
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

get_ipython().system('pip install natasha')
from natasha import (Segmenter,
                     MorphVocab,
                     NamesExtractor,

                     NewsEmbedding,
                     NewsMorphTagger,
                     NewsSyntaxParser,
                     NewsNERTagger,

                     PER,
                     Doc)

from yargy import Parser, rule, or_, and_
from yargy.interpretation import fact
from yargy.predicates import gram
from yargy.pipelines import morph_pipeline
from slovnet import NER

pd.set_option('max_rows', 99)
pd.set_option('max_colwidth', 400)
pd.describe_option('max_colwidth')

data = pd.read_csv('test_data.csv')

def change_role(data):
    if data == 'manager':
        return 'client'
    return 'manager'

data['role'] = data['role'].apply(lambda x: change_role(x))

def filter_text(text):
    """
    принимает текст,
    на выходе обработанны текст
    """
    text = str(text)

    text = text.lower()
    #
    #if len()
    #text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(.+[.]jpg)|(.+[.]pdf)', '', text)
    text = [c for c in text if c in 'абвгдеёжзийлкмнопрстуфхцчшщъыьэюя- ']
    text = ''.join(text)
    return text

def lemmatizer(text):
    lemm_text = []
    for word in text.split():
        p = morph.parse(word)[0]
        lemm_text.append(p.normal_form)
    return ' '.join(lemm_text)

data['filtered_text'] = data['text'].apply(lambda x: filter_text(x))
data['filtered_text'] = data['filtered_text'].apply(lambda x: lemmatizer(x))

# Создадим строки с которыми будем искать похожесть
# список со списками - чем длинее фраза тем длинее список
# 
greeting_words = [['привет','здравствуйте'], ['добрый день','день добрый']]
greeting_words2 = ['добрый день','день добрый']
parting_words = [['досвидания','',''], ['до свидания','хорошего дня',' хорошего вечера']]
parting_words2 = ['до свидания','хорошего дня',' хорошего вечера']
#introduced_words = ['','','']


wordtovec1 = CountVectorizer(ngram_range=(1,1))
wordtovec2 = CountVectorizer(ngram_range=(2,2))

def get_matrix(data_fit, data):
    text_matrix1 = wordtovec1.fit_transform(data_fit)
    words_matrix1 = wordtovec1.transform(data[0])

    text_matrix2 = wordtovec2.fit_transform(data_fit)
    words_matrix2 = wordtovec2.transform(data[1])

    text_matrix = np.column_stack((text_matrix1.toarray(), text_matrix2.toarray()))
    words_matrix = np.column_stack((words_matrix1.toarray(), words_matrix2.toarray()))
    return text_matrix, words_matrix


## Менеджер приветствовал

# передадим в функцию расчета векторизированнную матрицу  
# и сравниваться она будет со созданой строкой 
#  - для расчета похожести 
words_matrix, greeting_words_matrix = get_matrix( data['filtered_text'] , greeting_words)
cosine_sim_w2v = cosine_similarity(words_matrix, greeting_words_matrix)
#cosine_sim_w2v.shape

# добавим столбец с отобраным из вектора похожести максимальным значением 
# и колонку с логическими значениямим требуемой задачей - greeting

data['greeting_vec'] = cosine_sim_w2v.tolist()
data['greeting'] = data['greeting_vec'].apply(lambda x: True if  max(x) > 0.01 else None)
# сохраним индексы этих реплик
greeting_indx = data[data['role'] == 'manager'].loc[~data['greeting'].isna()].index.to_list()

## Менеджер попрощался

# Лемматизирую фразы
for i in range(0,len(parting_words)):
    for j in range(0,len(parting_words[i])):
        parting_words[i][j] = lemmatizer(parting_words[i][j])

words_matrix, parting_words_matrix = get_matrix( data['filtered_text'] , parting_words)
cosine_sim_w2v = cosine_similarity(words_matrix, parting_words_matrix)
data['parting_vec'] = cosine_sim_w2v.tolist()
data['parting'] = data['parting_vec'].apply(lambda x: True if  max(x) > 0.01 else None)
# сохраним индексы этих реплик
parting_indx = data[data['role'] == 'manager'].loc[~data['parting'].isna()].index.to_list()


## Менеджер представился

# инициализируем объекты 
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()

morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)

# сложная функция, написана практически методом 'тыка'
# но у неё получается определять имена в словах с маленькой буквы
# а затем она капитализирует слово и проверяет на наличие тагга PER(персона) 

def get_name(data):

    list_name = []
    doc = Doc(data)
    macth = names_extractor(data)
    for i in macth:
        if i.fact.first:
            list_name.append(i.fact.first)
    if list_name:
        jam = []
    for name in list_name:
        doc = Doc(name.capitalize())
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)  
        for item in doc.spans:
            if item.type == 'PER':
          #print(item)
                jam.append(item.text)
        if len(jam) > 0:
            return ' '.join(jam)

intro_words = ['зовут','это']
lemm_intro_words = [lemmatizer(i) for i in intro_words]

# функция которая ищет совпадения слов из реплики с заданнными
def where_words(text, words):
    for i in words:
        if i in text.lower().split():
            return True
    return False 

# отберем индексы первых фраз менеджера в каждом диалоге
first_rep_manag = data[(data['line_n'] < 4) & (data['role'] == 'manager')].index.to_list()
# проверим реплики по отбранным индексам на наличие слов предворяющих самопредставление
intro_rep_manag = data.loc[first_rep_manag]['filtered_text'].apply(lambda x: where_words(x, lemm_intro_words))



data['name'] = None
data.loc[first_rep_manag, 'name'] = data.loc[first_rep_manag,'filtered_text'].apply(lambda x: get_name(x))
intro_rep_manag_indx = data[~(data['name'].isna()) & intro_rep_manag].index.to_list()


## Наименование оргнизации

def return_org(data):
    doc = Doc(data)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    jam=list()
    for item2 in doc.spans:
        if item2.type == 'ORG':
          #print(item2)
            jam.append(item2.text)
    if len(jam) > 0: 
        return ' '.join(jam)

data['company'] = None

#data['company'] = data[data['role'] == 'manager']['text'].apply(lambda x: return_org(x))
#data[~(data['company'].isna())]

Orgname = fact('Org', ['orgform', 'name'])

ORGFORM = morph_pipeline(['компания'])

ORGNAME = and_(gram('NOUN'))#.interpretation(Name.orgnm)

ORGANIZATION = or_( 
    rule(ORGFORM, ORGNAME),
    rule(ORGFORM, ORGNAME, ORGNAME),
)

ORG = Parser(ORGANIZATION)

#Передаем парсеру объект «организация» и запускаем поиск:
orgparser = Parser(ORGANIZATION)

def orgs_extract(text, parser):
    result =None
    for match in parser.findall(text):
        result = ' '.join([_.value for _ in match.tokens])
    return result 

data['company'] = data[data['role'] == 'manager']['text'].apply(lambda x: orgs_extract(x,orgparser) or return_org(x))

dlg_id = sorted(data['dlg_id'].value_counts().index) 

name_table = data[~(data['name'].isna())][['dlg_id','name']].groupby('dlg_id').head(1)
company_table = data[~(data['company'].isna())][['dlg_id','company']].groupby('dlg_id').head(1)
n_c_table = name_table.merge(company_table, how='outer').set_index('dlg_id')

manager_check = pd.pivot_table(data=data[data['role']=='manager'][['dlg_id','greeting','parting']], index='dlg_id',  aggfunc='count')
manager_check = n_c_table.merge(manager_check, how='outer', left_index=True, right_index=True)

check_count = manager_check.groupby('name')['name'].agg(['count'])
check_gp = manager_check.groupby('name')[['greeting','parting']].agg(['sum'])
check_com = manager_check.groupby('name')[['company']].agg(['first'])

check = check_com.merge(check_count.merge(check_gp, how='outer',left_index=True, right_index=True),how='outer',left_index=True, right_index=True)
check.columns = ['company','count','greeting','parting',]
check = check.fillna('__') 

def get_case(word, value, case):
    word = morph.parse(word)[0]
    v1, v2, v3 = word.inflect({'sing', 'nomn'}), word.inflect({'gent'}), word.inflect({'plur', 'gent'})
    if value % 10 == 1 and value % 100 != 11:
        variant = v1.word
    elif value % 10 >= 2 and value % 10 <= 4 and         (value % 100 < 10 or value % 100 >= 20):
        variant = v2.word
    else:
        variant = v3.word

# word = morph.parse(word)[0]
# gent = word.inflect({'gent'})
# gent.word
    return variant

print('Отчет!')
print('========================================')
# циклом по именам в индексе таблицы check
for i,name in enumerate(check.index):
    count = check.loc[name]['count']
    company = ' '.join([i.capitalize() for i in (check.loc[name]['company']).split()])
    greeting = check.loc[name]['greeting']
    parting = check.loc[name]['parting']
    print(f'Менеджер {name} из {company} провел {count} {get_case("диалог",count,"im")}: поприветствовал в {greeting} и попрощался в {parting}.')
  

data = data.drop(['filtered_text','greeting_vec','parting_vec'], axis=1)
data.to_csv('result.csv')
print('В таблицу добавленны столбцы! Файл сохранён под именем result.csv')


data.drop(['filtered_text','greeting_vec','parting_vec'], axis=1)

