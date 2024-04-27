import os
'''
Пути к файлам с подготовленными данными. 
'''

# Путь к папке prepared_datasets.
PATH_FOLDER = os.path.abspath('src\\datasets\\prepared_datasets')

# Путь к файлам внутри.
PATH_RAW_DATA = os.path.join(PATH_FOLDER, 'raw_data.txt')
PATH_DICT_RAW_DATA = os.path.join(PATH_FOLDER, 'dict_raw_data.txt')
PATH_DATASET = os.path.join(PATH_FOLDER, 'dataset.pt')
PATH_TOK_INP = os.path.join(PATH_FOLDER, 'tokenizer_input.pkl')
PATH_TOK_OUT = os.path.join(PATH_FOLDER, 'tokenizer_output.pkl')
PATH_SAVE_MODEL = os.path.join(PATH_FOLDER, 'transformer')