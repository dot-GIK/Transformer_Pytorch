import wikipedia
import random

counter = 1
wiki_dict = []
def main():
    '''
    This script helps me create a dataset from different sentences.
    '''

    global counter # Word counter
    global wiki_dict
    wikipedia.set_lang('ru')

    with open('dataset.txt', 'r', encoding='utf-8') as t:
        data = t.read()
        data = data.replace('\\ufeff', '')

    wiki_keywords = random.choice(data.split(' '))

    with open('dictionary.txt', 'r', encoding='utf-8') as d:
        dictionary = d.read()
        dictionary = dictionary.replace('\\ufeff', '')
        dictionary = dictionary.split(' ')        

    if wiki_keywords in dictionary: main()
    else: 
        with open('dictionary.txt', 'a+', encoding='utf-8') as d:
            d.write(' ' + wiki_keywords)

    print(wiki_keywords)

    try:
        print(counter)
        python_page = wikipedia.page(wiki_keywords)
        with open('dataset.txt', 'a+', encoding='utf-8') as f:
            f.write(python_page.summary)
        counter += 1
        main()

    except wikipedia.exceptions.DisambiguationError:
        main()

    except wikipedia.exceptions.PageError:
        main()
    
    except wikipedia.exceptions.WikipediaException:
        main()

if __name__ == '__main__':
    main()