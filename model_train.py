import pandas as pd
import numpy as np
import joblib

# Обучение модели
def training(train_x, train_y, model_type, model_method):
    model = None
    match model_type:
        case 'classificator':
            match model_method:
                case 'decision_trees':
                    from sklearn.tree import DecisionTreeClassifier
                    model = DecisionTreeClassifier()
                case 'random_forest':
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
        case 'predictor':
            match model_method:
                case 'decision_trees':
                    from sklearn.tree import DecisionTreeRegressor
                    model = DecisionTreeRegressor()
                case 'random_forest':
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(train_x, train_y)

    return model


# Выгрузка на диск модели и её словаря кодировщика в виде файлов
def dump_model_files(model, label_encoders):
    joblib.dump(model, 'last_model.pkl')
    joblib.dump(label_encoders, 'last_label_encoders.pkl')


# Процесс предсказания
def predict_value(values_list, data_frame, model, label_encoders, model_type):
    try:
        # Получение имен столбцов
        all_cols = data_frame.columns.tolist()
        # Преобразование словаря со значениями для предсказания в дата фрейм Pandas
        df_predicted_var = pd.DataFrame([values_list], columns=all_cols[:-1])
        for i in df_predicted_var.columns:
            save_val = df_predicted_var.at[0, i] # Сохранение значения в случае возникновения ошибки
            if data_frame[i].dtype != 'float64' and data_frame[i].dtype != 'int64':
                if model_type == 'predictor':
                    from sklearn.preprocessing import LabelEncoder
                    # Получение кодировщика для текущего столбца
                    previous_values = label_encoders[i].classes_
                    if type(label_encoders[i].classes_[0]) == np.datetime64:
                        # Если кодировщик типа datetime, преобразование значение в этот тип
                        new_value = pd.to_datetime(df_predicted_var[i])
                    else:
                        # Иначе тип object
                        new_value = df_predicted_var[i]
                    # Создание нового списка для кодировщика
                    new_list = pd.concat([pd.Series(previous_values), new_value])
                    # обучение кодировщика с новым списком
                    label_encoder = LabelEncoder()
                    label_encoder.fit(new_list)
                    label_encoders[i] = label_encoder
                # Преобразование значения с помощью кодировщика
                df_predicted_var[i] = label_encoders[i].transform(df_predicted_var[i])
            else:
                continue
        # Предсказание модели и её выгрузка
        predicted_result = model.predict(df_predicted_var)
        dump_model_files(model, label_encoders)
        # Если используется классификация, то итоговое значение преобразуется с помощью кодировщика
        if model_type == 'classificator':
            predicted_result_decoded = \
                label_encoders[data_frame.columns[-1]].inverse_transform(predicted_result)
            return predicted_result_decoded[0]
        else:
            print(predicted_result[0])
            if predicted_result[0] is None:
                return 'Неизвестно'
            else:
                return predicted_result[0]
    except ValueError:
        from tkinter import messagebox
        if model_type == 'classificator':
            messagebox.showinfo('Ошибка классификации', f"Значение '{str(save_val)}' из столбца '{i}' "
                                f"не было найдено в объекта данных. Попробуйте другое значение")
        else:
            messagebox.showinfo('Ошибка регрессии', f"Значение '{str(save_val)}' из столбца '{i}' "
                                f"было введено некорректно.")
        return 'Неизвестно'


if __name__ == '__main__':
    print('Эта программа доступна только из файла ui_manager.py')

