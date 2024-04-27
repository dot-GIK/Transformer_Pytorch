class InvalidDataType(Exception):
    def __init__(self, variable, dtype: type) -> None:
        super().__init__(f"Переменная '{variable}' ожидает тип данных '{dtype}', был введен тип данных'{type(variable)}'.")


class TextLengthError(Exception):
    def __init__(self, length: int, max_words: int) -> None:
        super().__init__(f"Длина текста не соответствует установленной '{max_words}'; размер вашего текста '{length}'.")


class DatasetBatchSizeError(Exception):
    def __init__(self, length: int, batch_size: int) -> None:
        super().__init__(f"Размер датасета - 'length', должен делиться на размер партий - 'batch_size'.")


class IncorrectDatasetSize(Exception):
    def __init__(self, max_len: int, batch_size: int, seq_len: int, num_of_datasets: int) -> None:
        super().__init__(f"Размере датасета - 'max_len = {max_len}', должен быть кратен группе - 'batch_size * seq_len * num_of_datasets = {batch_size * seq_len * num_of_datasets}', в вашем случае остаток равен '{max_len % (batch_size * seq_len * num_of_datasets)}'.")