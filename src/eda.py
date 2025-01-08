import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(file_path='Credit_Default.csv'):
    # Загрузка данных
    df = pd.read_csv(file_path)
    
    print("Первые 5 строк:")
    print(df.head())
    print("\nОписательная статистика:")
    print(df.describe())
    
    sns.pairplot(df, hue='Default')
    plt.suptitle('Попарное распределение признаков')
    plt.tight_layout()
    plt.savefig('pairplot.png')
    plt.close()
    

    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Корреляционная матрица')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    

    plt.figure(figsize=(8, 6))
    df['Default'].value_counts().plot(kind='bar')
    plt.title('Распределение классов')
    plt.xlabel('Класс (Default)')
    plt.ylabel('Количество')
    plt.tight_layout()
    plt.savefig('class_balance.png')
    plt.close()
    

    default_rate = df['Default'].mean() * 100
    print(f"\nДоля дефолтов: {default_rate:.2f}%")
    
    correlation_with_default = correlation_matrix['Default'].sort_values(ascending=False)
    print("\nКорреляция признаков с дефолтом:")
    print(correlation_with_default)

    def generate_eda_insights(df):
        """
        Генерирует аналитические выводы по результатам EDA

        Args:
            df (pd.DataFrame): DataFrame с данными

        Returns:
            str: Текстовый отчет с выводами
        """
        # Доля дефолтов
        default_rate = df['Default'].mean() * 100

        # Корреляция признаков с дефолтом
        correlation_with_default = df.corr()['Default'].sort_values(ascending=False)



    report = f"""# Отчет по разведочному анализу данных (EDA)

    ## Ключевые показатели

    ### Доля дефолтов
    - **{default_rate:.2f}%** клиентов имеют дефолт по кредиту

    ### Корреляционный анализ

    #### Влияние признаков на вероятность дефолта (от сильного к слабому):

    1. **Возраст (Age)**: 
    - Сильная отрицательная корреляция ({correlation_with_default['Age']:.4f})
    - Чем моложе клиент, тем выше риск дефолта

    2. **Отношение кредита к доходу (Loan to Income)**:
    - Умеренная положительная корреляция ({correlation_with_default['Loan to Income']:.4f})
    - Высокое отношение кредита к доходу повышает риск дефолта

    3. **Сумма кредита (Loan)**:
    - Умеренная положительная корреляция ({correlation_with_default['Loan']:.4f})
    - Чем больше сумма кредита, тем выше риск дефолта

    4. **Доход (Income)**:
    - Практически нет корреляции ({correlation_with_default['Income']:.4f})
    - Доход слабо влияет на вероятность дефолта

    ## Рекомендации

    1. При оценке кредитного риска особое внимание уделять:
    - Возрасту заемщика
    - Отношению суммы кредита к доходу

    2. Молодые заемщики требуют более тщательной оценки кредитоспособности

    3. Необходимо контролировать соотношение суммы кредита к доходу
    """
        
    print(report)

if __name__ == "__main__":
    perform_eda()