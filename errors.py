class InvalidDataType(Exception):
    def __init__(self, variable, dtype: type) -> None:
        super().__init__(f"Переменная '{variable}' ожидает тип данных '{dtype}', был введен тип данных'{type(variable)}'.")


class TextLengthError(Exception):
    def __init__(self, length: int, max_words: int) -> None:
        super().__init__(f"Длина текста не соответствует установленной '{max_words}'; размер вашего текста '{length}'.")


class DatasetBatchSizeError(Exception):
    def __init__(self, length: int, batch_size: int) -> None:
        super().__init__(f"Размер датасета - 'length', должен делиться на размер партий - 'batch_size'.")