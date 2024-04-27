import os 
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import wikipedia
import random
from global_variables.paths import PATH_RAW_DATA, PATH_DICT_RAW_DATA

counter = 1
wiki_dict = []
def wiki():
    '''
    This script helps me create a dataset from different sentences.
    '''

    global counter # Word counter
    global wiki_dict
    wikipedia.set_lang('ru')

    with open(PATH_RAW_DATA, 'r', encoding='utf-8') as t:
        data = t.read()
        data = data.replace('\\ufeff', '')

    wiki_keywords = random.choice(data.split(' '))

    with open(PATH_DICT_RAW_DATA, 'r', encoding='utf-8') as d:
        dictionary = d.read()
        dictionary = dictionary.replace('\\ufeff', '')
        dictionary = dictionary.split(' ')        

    if wiki_keywords in dictionary: wiki()
    else: 
        with open(PATH_DICT_RAW_DATA, 'a+', encoding='utf-8') as d:
            d.write(' ' + wiki_keywords)

    print(wiki_keywords)

    try:
        print(counter)
        python_page = wikipedia.page(wiki_keywords)
        with open(PATH_RAW_DATA, 'a+', encoding='utf-8') as f:
            f.write(python_page.summary)
        counter += 1
        wiki()

    except wikipedia.exceptions.DisambiguationError:
        wiki()

    except wikipedia.exceptions.PageError:
        wiki()
    
    except wikipedia.exceptions.WikipediaException:
        wiki()

if __name__ == '__main__':
    wiki()