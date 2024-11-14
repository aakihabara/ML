from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt



# Вычисление метрик, которые зависят от типа модели
def get_metrics(test_x, test_y, model_type, model):
    if model is not None:
        predicted = model.predict(test_x)
        score = None

        match model_type:
            case 'classificator':
                score = metrics.accuracy_score(test_y, predicted)
                absolute_error = None
            case 'predictor':
                score = metrics.r2_score(test_y, predicted)
                absolute_error = metrics.mean_absolute_error(test_y, predicted)

        return score, absolute_error


# График точности для тестовых данных
def print_graphic(test_x, test_y, model_type, model):
    import pandas as pd
    fig = plt.figure('Окно графика', figsize=(10, 6))
    plt.title('График точности тестовых предсказаний')
    predicted = model.predict(test_x)
    df = pd.DataFrame({'Фактические': test_y, 'Предсказанные': predicted})
    counts = df.groupby(['Фактические', 'Предсказанные']).size().reset_index(name='Count')
    sns.scatterplot(data=counts, x='Фактические', y='Предсказанные', size='Count', sizes=(10, 500), legend=False, alpha=0.5, color='blue')
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k-', lw=2)
    match model_type:
        case 'decision_trees':
            plt.title('График точности для метода дерева решений')
        case 'random_forest':
            plt.title('График точности для метода случайного леса')
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')

    for i in range(counts.shape[0]):
        plt.text(counts['Фактические'][i], counts['Предсказанные'][i], counts['Count'][i], fontsize=12, ha='center', color='red')
    
    fig.show()


if __name__ == '__main__':
    print('Эта программа доступна только из файла ui_manager.py')

