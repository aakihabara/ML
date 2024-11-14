from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Предобработка данных
def preprocessing(data_frame):
    df = data_frame.copy()

    label_encoders = {}

    for column in df.columns:
        # Проверка, является ли столбец численным типом
        if df[column].dtype != 'int64' and df[column].dtype != 'float64':
            # Создание кодировщика для каждого столбца
            label_encoder = LabelEncoder()
            # Преобразование значений столбца в закодированные
            df[column] = label_encoder.fit_transform(df[column])
            # Сохранение кодировщика для текущего столбца
            label_encoders[column] = label_encoder

    info_values = df.iloc[:, :-1]  # Все столбцы, кроме последнего
    result = df.iloc[:, -1]  # Последний столбец

    # Разделение данных на тестовую и обучающую части
    train_x, test_x, train_y, test_y = train_test_split(info_values, result, test_size=0.1,
                                                        random_state=100)
    train_test_list = train_x, test_x, train_y, test_y
    return train_test_list, label_encoders


if __name__ == '__main__':
    print('Эта программа доступна только из файла ui_manager.py')

