
import mysql.connector
from mysql.connector import Error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split  # Corrected import
from sklearn.metrics import mean_squared_error


def connect_to_data():
    try:
        db = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            passwd="2431",
            database="reg_finance",
            connect_timeout=60
        )
        mycursor = db.cursor()
        return mycursor, db
    except Error as e:
        # В случае ошибки выводим сообщение и пытаемся переподключиться
        print(f"Ошибка при подключении к базе данных: {e}")
        if db:
            db.close()
        return None, None


def create_df(table):
    mycursor, db = connect_to_data()

    mycursor.execute("USE reg_finance")
    mycursor.execute(f'SELECT * FROM {table}')
    info = mycursor.fetchall()
    column_names = [i[0] for i in mycursor.description]
    df = pd.DataFrame(info, columns=column_names)

    if 'код_региона' in df.columns:
        df.set_index('код_региона', inplace=True)
    if 'код_округа' in df.columns:
        df.set_index('код_округа', inplace=True)
    if table == 'коды_регионов':
        df['название'] = df['название'].astype(str)
        return df
    if table == 'коды_регионов':
        df['название'] = df['название'].astype(str)
        return df
    if table == 'коды_округов':
        df['название'] = df['название'].astype(str)
        return df
    db.close()
    df = convert_df_to_numeric(df)
    return df


def numeric_convert(value):
    if isinstance(value, str):
        # Удаляем все пробелы
        value = value.replace(' ', '')

        # Удаляем все после запятой, если запятая присутствует
        if '.' in value:
            value = value.split('.')[0]
            value = value.replace('.', '')

        try:
            return int(value)  # Преобразуем значение в int
        except (ValueError, TypeError):
            return None
    else:
        return int(value)  # Преобразуем значение в int


def convert_df_to_numeric(df):
    return df.map(numeric_convert)


# поиск названия региона
def find_reg_name(reg, df):
    if df is not None and reg in df.index:
        region_name = df.loc[reg, 'название']
        return region_name
    else:
        return "Регион не найден"



def delete_data():
    """Удаляет данные из указанной таблицы."""
    mycursor, db = connect_to_data()
    if not mycursor:
        return

    try:
        table_name = input("Введите имя таблицы, из которой хотите удалить данные: ").strip()
        if table_name not in table_mapping:
            print(f"Таблица '{table_name}' не найдена.")
            return

        confirmation = input(
            f"Вы уверены, что хотите УДАЛИТЬ ВСЕ данные из таблицы '{table_name}'? (y/n): ").strip().lower()
        if confirmation == 'y':
            mycursor.execute("USE reg_finance")  # Убедитесь, что выбрана правильная база данных
            mycursor.execute(f"DELETE FROM {table_name}")
            db.commit()
            print(f"Все данные из таблицы '{table_name}' успешно удалены.")
        else:
            print("Удаление отменено.")
    except Error as e:
        print(f"Ошибка при удалении данных: {e}")
        db.rollback()  # Откатываем изменения в случае ошибки
    finally:
        if db:
            db.close()


def add_data():
    """Добавляет данные в указанную таблицу из DataFrame."""
    mycursor, db = connect_to_data()
    if not mycursor:
        return

    try:
        table_name = input("Введите имя таблицы, в которую хотите добавить данные: ").strip()
        if table_name not in table_mapping:
            print(f"Таблица '{table_name}' не найдена.")
            return
        df = create_df(table_name)  # Получаем таблицу чтобы получить названия столбцов
        column_names = list(df.columns)
        values = []
        print("Введите значения для каждого столбца. Оставьте поле пустым для значения NULL.")
        for col in column_names:
            value = input(f"Введите значение для '{col}': ").strip()
            if value == "":
                values.append(None)
            else:
                try:
                    values.append(numeric_convert(value))
                except ValueError:
                    print(f"Некорректный формат числа для столбца '{col}'. Пожалуйста, введите число.")
                    return
        placeholders = ', '.join(['%s'] * len(column_names))
        sql = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({placeholders})"
        mycursor.execute("USE reg_finance")
        mycursor.execute(sql, values)
        db.commit()
        print("Данные успешно добавлены.")

    except Error as e:
        print(f"Ошибка при добавлении данных: {e}")
        db.rollback()
    finally:
        if db:
            db.close()


def edit_data():
    """Редактирует данные в указанной таблице."""
    mycursor, db = connect_to_data()
    if not mycursor:
        return

    try:
        table_name = input("Введите имя таблицы, в которой хотите изменить данные: ").strip()
        if table_name not in table_mapping:
            print(f"Таблица '{table_name}' не найдена.")
            return

        df = create_df(table_name)
        if df is None or df.empty:
            print("Не удалось загрузить таблицу или таблица пуста.")
            return

        print("Доступные столбцы для редактирования:")
        for i, col in enumerate(df.columns):
            print(f"{i + 1}. {col}")

        while True:
            try:
                column_index = int(input("Введите номер столбца для редактирования: ")) - 1
                if 0 <= column_index < len(df.columns):
                    column_name = df.columns[column_index]
                    break
                else:
                    print("Некорректный номер столбца.")
            except ValueError:
                print("Некорректный ввод. Пожалуйста, введите число.")

        while True:
            try:
                row_index = int(input(
                    f"Введите индекс строки (код_региона если есть, если нет введите номер строки) для редактирования: "))
                if df.index.name == 'код_региона':
                    if row_index in df.index:
                        break
                    else:
                        print("Некорректный код региона.")
                else:
                    if 0 <= row_index < len(df):
                        break
                    else:
                        print("Некорректный номер строки.")
            except ValueError:
                print("Некорректный ввод. Пожалуйста, введите число.")

        if df.index.name == 'код_региона':
            where_condition = f"{df.index.name} = {row_index}"
        else:
            where_condition = f"1=1 LIMIT 1 OFFSET {row_index}"

        new_value = input(f"Введите новое значение для столбца '{column_name}': ").strip()
        if new_value == "":
            new_value = None  # Установка в NULL

        try:
            new_value = numeric_convert(new_value)
        except ValueError:
            print("Некорректный формат числа. Для числовых столбцов введите число.")
            return
        sql = f"UPDATE {table_name} SET `{column_name}` = %s WHERE {where_condition}"
        mycursor.execute("USE reg_finance")
        mycursor.execute(sql, (new_value,))  # Ensure the new value is passed as a tuple
        db.commit()
        print("Данные успешно обновлены.")

    except Error as e:
        print(f"Ошибка при обновлении данных: {e}")
        db.rollback()
    finally:
        if db:
            db.close()


def view_data():
    """Выводит всю указанную таблицу."""
    mycursor, db = connect_to_data()
    if not mycursor:
        return

    try:
        table_name = input("Введите имя таблицы, которую хотите просмотреть: ").strip()
        if table_name not in table_mapping:
            print(f"Таблица '{table_name}' не найдена.")
            return

        df = create_df(table_name)
        if df is None or df.empty:
            print("Не удалось загрузить таблицу или таблица пуста.")
            return

        print(df)

    except Error as e:
        print(f"Ошибка при просмотре данных: {e}")
    finally:
        if db:
            db.close()



def basic_stats():
    print("Общая статистика данных")

    # Вызываем choose_table для выбора одной таблицы
    df = choose_table("1")  # Используем метод "1" для анализа статистики одной таблицы

    if df is None:
        print("Выбор таблицы завершился неудачей. Прерывание анализа.")
        return

    # Проверяем, не пустой ли DataFrame
    if df.empty:
        print("Выбранная таблица не содержит данных.")
        return

    # Если в DataFrame есть столбец 'код_региона', заменяем его значение на название региона
    if 'код_региона' in df.columns:
        reg_name_df = create_df('коды_регионов')  # Создаём DataFrame со справочными данными о регионах
        df['название_региона'] = df['код_региона'].apply(
            lambda reg: find_reg_name(reg, reg_name_df)  # Меняем код региона на название
        )
        df.drop(columns=['код_региона'], inplace=True)  # Удаляем колонку кода региона, если необходимо

    # Выводим базовую статистику
    print(f"\nБазовая статистика для выбранной таблицы:")

    try:
        print("\nПервая часть данных (первые 5 строк):")
        print(df.head(), "\n")

        print("Базовая статистическая информация (числовые столбцы):")
        print(df.describe(include=[np.number]), "\n")

        # Готовим данные для построения графика сводной статистики
        numerical_columns = df.select_dtypes(include=[np.number]).columns  # Находим числовые столбцы

        if len(numerical_columns) > 0:
            # Создаём словарь для сводных значений
            stats = {
                "Среднее значение": df[numerical_columns].mean(),
                "Медиана": df[numerical_columns].median(),
                "Минимум": df[numerical_columns].min(),
                "Максимум": df[numerical_columns].max(),
                "Ст. отклонение": df[numerical_columns].std(),
            }

            # Преобразуем словарь в новый DataFrame для удобства построения графика
            stats_df = pd.DataFrame(stats)

            # Построение графика
            plt.figure(figsize=(10, 6))

            # Параметры: данные по осям и стекаем линии
            for stat_name in stats.keys():
                plt.plot(stats_df.index, stats_df[stat_name], marker='o', label=stat_name)

            # Настраиваем подписи и заголовки
            plt.title("Сводная статистика для числовых столбцов", fontsize=16)
            plt.xlabel("Столбцы", fontsize=14)
            plt.ylabel("Значения", fontsize=14)
            plt.xticks(rotation=45, ha='right')  # Поворот подписей столбцов

            # Добавляем легенду
            plt.legend(title="Показатели", fontsize=12)

            # Улучшенная компоновка
            plt.tight_layout()

            # Показываем график
            plt.show()
        else:
            print("В выбранной таблице отсутствуют числовые данные для построения графика.")

    except Exception as e:
        print(f"Произошла ошибка при вычислении статистики или построении графика: {e}")



def correlation_analysis():
    """
    Корреляционный анализ данных по таблицам за выбранный год с интерпретацией результатов.
    """
    print("Корреляционный анализ по таблицам со средними значениями за выбранный год.")
    selected_year = input("Введите год для анализа (например, 2021): ").strip()

    try:
        selected_year = int(selected_year)
    except ValueError:
        print("Ошибка: Введен некорректный год. Пожалуйста, введите числовое значение.")
        return

    # Фильтрация таблиц по году
    yearly_dataframes = {}
    for table_name, df in table_mapping.items():
        if df is None:
            print(f"Предупреждение: таблица '{table_name}' не загружена, пропуск.")
            continue

        year_column = str(selected_year)
        if year_column not in df.columns:
            print(f"Предупреждение: в таблице '{table_name}' нет данных за {selected_year} год, пропуск.")
            continue

        if pd.api.types.is_numeric_dtype(df[year_column]):
            # Создаем копию DataFrame с нужным годом
            yearly_dataframes[table_name] = df[[year_column]].copy()
            yearly_dataframes[table_name].rename(columns={year_column: table_name},
                                                 inplace=True)  # Переименовываем колонку для уникальности
        else:
            print(f"Предупреждение: столбец '{year_column}' в таблице '{table_name}' не является числовым, пропуск.")

    # Объединение таблиц
    if not yearly_dataframes:
        print(f"Нет таблиц с данными за {selected_year} год. Корреляционный анализ невозможен.")
        return

    # Берем первую таблицу как основу для объединения
    merged_df = list(yearly_dataframes.values())[0]

    # Объединяем остальные таблицы
    for table_name, df in list(yearly_dataframes.items())[1:]:
        merged_df = merged_df.join(df, how='outer')

    # Проверяем, что после объединения остались данные
    if merged_df.empty:
        print("После объединения таблиц не осталось данных. Корреляционный анализ невозможен.")
        return

    # Вычисление корреляционной матрицы
    try:
        correlation_matrix = merged_df.corr(method='pearson')
    except Exception as e:
        print(f"Ошибка при вычислении корреляционной матрицы: {e}")
        return

    # Выводим корреляционную матрицу
    print(f"\nКорреляционная матрица по таблицам за {selected_year} год:")
    print(correlation_matrix)

    # Визуализация корреляционной матрицы
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, annot_kws={"fontsize": 10})
    plt.title(f"Корреляция данных за {selected_year} год", fontsize=16)
    table_labels = [table_fullname.get(table, table) for table in correlation_matrix.columns]
    plt.yticks(ticks=np.arange(len(table_labels)), labels=table_labels, rotation=0, fontsize=8)
    plt.xticks(ticks=np.arange(len(table_labels)), labels=table_labels, rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.show()

    # Интерпретация корреляций (пример)
    print("\nИнтерпретация корреляционной матрицы:")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            correlation = correlation_matrix.iloc[i, j]

            print(
                f"Корреляция между '{table_fullname.get(col1, col1)}' и '{table_fullname.get(col2, col2)}': {correlation:.2f}")

            if abs(correlation) > 0.7:
                print(f"  => Обнаружена сильная {'положительная' if correlation > 0 else 'отрицательная'} корреляция.")
                # Здесь можно добавить более детальный анализ или предупреждения
            elif abs(correlation) > 0.5:
                print(
                    f"  => Обнаружена умеренная {'положительная' if correlation > 0 else 'отрицательная'} корреляция.")
            else:
                print(f"  => Корреляция слабая или отсутствует.")

            print("")  # Добавляем пустую строку для разделения выводов



def data_distribution_analysis():
    """
    Анализ распределения данных за указанный год.
    """
    print("Анализ распределения данных")

    # Запрашиваем год у пользователя
    selected_year = input("Введите год для анализа (например, 2021): ").strip()

    try:
        selected_year = int(selected_year)
    except ValueError:
        print("Ошибка: Введен некорректный год. Пожалуйста, введите числовое значение.")
        return

    df = choose_table("3")  # Выбор таблицы

    if df is None or df.empty:
        print("Не удалось выбрать таблицу или таблица пуста. Прерывание анализа.")
        return

    year_column = str(selected_year)
    if year_column not in df.columns:
        print(f"Предупреждение: в таблице нет данных за {selected_year} год.")
        return

    if not pd.api.types.is_numeric_dtype(df[year_column]):
        print(f"Предупреждение: столбец '{year_column}' не является числовым.")
        return

    plt.figure(figsize=(12, 6))
    sns.histplot(df[year_column].dropna(), kde=True)  # Строим гистограмму с оценкой плотности ядра (KDE)
    plt.title(f"Распределение '{year_column}' (Гистограмма)")
    plt.xlabel("Значение")
    plt.ylabel("Частота")

    # Q-Q plot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    stats.probplot(df[year_column].dropna(), dist="norm", plot=plt)  # Строим Q-Q plot
    plt.title(f"Q-Q Plot для '{year_column}'")

    plt.tight_layout()  # Улучшаем размещение графиков
    plt.show()

    # Вывод основных статистик
    print(f"\nСтатистики для столбца '{year_column}':")
    print(df[year_column].describe())
    print(f"Смещенность: {df[year_column].skew()}")  # Коэффициент асимметрии
    print(f"Эксцесс: {df[year_column].kurtosis()}")  # Коэффициент эксцесса
    # Интерпретация этих значений поможет вам понять форму распределения (например, нормальное, скошенное вправо/влево, островершинное или плосковершинное).



def anomaly_detection():
    """
    Детекция аномалий за указанный год и вывод названий регионов с аномалиями.
    """
    print("Детекция аномалий")

    # Запрашиваем год у пользователя
    selected_year = input("Введите год для анализа (например, 2021): ").strip()

    try:
        selected_year = int(selected_year)
    except ValueError:
        print("Ошибка: Введен некорректный год. Пожалуйста, введите числовое значение.")
        return

    df = choose_table("4")  # Выбор таблицы

    if df is None or df.empty:
        print("Не удалось выбрать таблицу или таблица пуста. Прерывание анализа.")
        return

    year_column = str(selected_year)
    if year_column not in df.columns:
        print(f"Предупреждение: в таблице нет данных за {selected_year} год.")
        return

    if not pd.api.types.is_numeric_dtype(df[year_column]):
        print(f"Предупреждение: столбец '{year_column}' не является числовым.")
        return

    # Сохраняем индекс (код региона) во временный столбец, если он есть
    if df.index.name == 'код_региона':
        df['temp_region_code'] = df.index

    # Z-score method
    z_scores = np.abs(stats.zscore(df[year_column].dropna()))  # Считаем Z-score для каждого значения
    threshold = 3  # Определяем порог (обычно 2.5 - 3)
    anomalies = df[year_column].dropna()[z_scores > threshold]  # Находим аномалии

    print(f"\nАномалии в столбце '{year_column}' (Z-score > {threshold}):")

    # Если есть индексы (коды регионов), выводим названия регионов
    if df.index.name == 'код_региона':
        reg_name_df = create_df('коды_регионов')  # Загружаем таблицу с названиями регионов
        for region_code, value in anomalies.items():
            region_name = find_reg_name(region_code, reg_name_df)  # Находим название региона
            print(f"Регион: {region_name} (Код: {region_code}), Значение: {value}")

    else:
        print(anomalies)

    # Boxplot visualization
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[year_column])
    plt.title(f"Boxplot для '{year_column}' (с аномалиями)")
    plt.xlabel("Значение")
    plt.show()

    # IQR method (alternative) - можно добавить по желанию
    Q1 = df[year_column].quantile(0.25)
    Q3 = df[year_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies_iqr = df[(df[year_column] < lower_bound) | (df[year_column] > upper_bound)][year_column]

    print(f"\nАномалии в столбце '{year_column}' (IQR method):")
    if df.index.name == 'код_региона':
        reg_name_df = create_df('коды_регионов')  # Загружаем таблицу с названиями регионов
        for region_code, value in anomalies_iqr.items():
            region_name = find_reg_name(region_code, reg_name_df)  # Находим название региона
            print(f"Регион: {region_name} (Код: {region_code}), Значение: {value}")
    else:
        print(anomalies_iqr)



def data_forecasting():
    print("Прогнозирование данных")

    df = choose_table("6")

    if df is None or df.empty:
        print("Не удалось выбрать таблицу или таблица пуста. Прерывание прогнозирования.")
        return

    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    year_columns = [col for col in numerical_columns if col.isdigit()]

    if not year_columns or len(year_columns) < 2:
        print("В выбранной таблице недостаточно данных по годам для прогнозирования изменений.")
        return

    while True:
        try:
            forecast_years_input = input(
                "Введите годы *для прогнозирования* через пробел (например, 2025 2026): ").strip()
            forecast_years = [int(year) for year in forecast_years_input.split()]
            if any(str(year) in year_columns for year in forecast_years):
                print("Один или несколько введенных годов уже есть в данных. Введите годы, которых нет в данных.")
            else:
                break
        except ValueError:
            print("Некорректный ввод. Пожалуйста, введите числа через пробел.")

    # Создаем признаки на основе изменений между годами
    changes = {}
    for index, row in df.iterrows():
        year_data = row[year_columns].astype(float)  # Ensure data is numeric
        if year_data.isnull().any():  # Skip rows with missing data
            print(f"Skipping row {index} due to missing data")
            continue
        row_changes = {}
        for i in range(1, len(year_columns)):
            # Access by label (column name)
            change = year_data.iloc[i] - year_data.iloc[i - 1]
            row_changes[year_columns[i]] = change
        changes[index] = row_changes

    if not changes:
        print("Недостаточно данных для расчета изменений между годами. Прогнозирование невозможно.")
        return

    changes_df = pd.DataFrame.from_dict(changes, orient='index')
    changes_df = changes_df.dropna(axis=1, how='all')  # Drop columns with all NaN values

    # Автоматически выбираем изменение до 2021, если оно доступно
    target_column = '2021'
    if target_column not in changes_df.columns:
        # Если 2021 недоступно, выбираем последний доступный год
        target_column = changes_df.columns[-1]
        print(f"Изменение до 2021 недоступно, автоматически выбран: {target_column}")
    else:
        print("Автоматически выбрано: Изменение до 2021")

    feature_columns = [col for col in changes_df.columns if col != target_column]

    # Подготовка данных для модели
    X = changes_df[feature_columns]
    y = changes_df[target_column]

    # Обработка пропущенных значений
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except ValueError as e:
        print(f"Ошибка при разделении данных: {e}")
        print("Убедитесь, что у вас достаточно данных для разделения на обучающую и тестовую выборки.")
        return

    # Обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Прогнозирование изменения на основе имеющихся данных
    y_pred = model.predict(X_test)

    # Оценка производительности модели
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nСреднеквадратичная ошибка (MSE) на тестовых данных: {mse:.2f}")

    predicted_values = {}
    last_year = int(year_columns[-1])
    last_year_data = df[year_columns[-1]].mean()  # Average value from the last year
    current_value = last_year_data

    for forecast_year in forecast_years:
        # Прогнозирование изменения на forecast_year
        forecast_data = pd.DataFrame([X.mean()], columns=X.columns)  # mean of features
        predicted_change = model.predict(forecast_data)[0]

        current_value += predicted_change
        predicted_values[forecast_year] = current_value
        print(f"\nПрогнозируемое значение для {forecast_year}: {current_value:.2f}")

    # Visualization
    years = [int(col) for col in year_columns]
    data = df[year_columns].mean().tolist()  # Use mean for each year

    # Добавляем спрогнозированные значения
    for forecast_year in forecast_years:
        years.append(forecast_year)
        data.append(predicted_values[forecast_year])

    plt.figure(figsize=(12, 6))
    plt.plot(years, data, marker='o', linestyle='-', color='blue', label='Исторические данные')

    # Выделяем спрогнозированные значения
    for forecast_year, predicted_value in predicted_values.items():
        plt.scatter(forecast_year, predicted_value, color='red', s=100,
                    label='Прогноз' if forecast_year == forecast_years[0] else "")

    plt.title(f"Прогнозирование данных")
    plt.xlabel("Год")
    plt.ylabel("Значение")
    plt.xticks(years)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


reg_name = create_df('коды_регионов')
okr_name = create_df('коды_округов')
vrp = create_df('врп')
dkb = create_df('дкб')
rkb = create_df('ркб')
fkp = create_df('фкп')
pkb_r = create_df('пкб_р')
pkb_f = create_df('пкб_фед')
zd_ul = create_df('задолженность_юл')
zd_fl = create_df('задолженность_фл')
vk_ul = create_df('вклады_юл')
vk_fl = create_df('вклады_фл')


table_fullname = {
    'врп': " ВАЛОВОЙ РЕГИОНАЛЬНЫЙ ПРОДУКТ",
    'дкб': " ДОХОДЫ КОНСОЛИДИРОВАННЫХ БЮДЖЕТОВ",
    'ркб': " РАСХОДЫ КОНСОЛИДИРОВАННЫХ БЮДЖЕТОВ",
    'пкб_р': " ПОСТУПЛЕНИЕ ОБЯЗАТЕЛЬНЫХ ПЛАТЕЖЕЙ В КОНСОЛИДИРОВАННЫЙ БЮДЖЕТ СУБЪЕКТА",
    'пкб_фед': " ПОСТУПЛЕНИЕ ОБЯЗАТЕЛЬНЫХ ПЛАТЕЖЕЙ В ФЕДЕРАЛЬНЫЙ БЮДЖЕТ",
    'вклады_фл': " ВКЛАДЫ В РУБЛЯХ ФИЗИЧЕСКИХ ЛИЦ",
    'вклады_юл': " ВКЛАДЫ В РУБЛЯХ ЮРИДИЧЕСКИХ ЛИЦ",
    'задолженность_юл': " ЗАДОЛЖЕННОСТЬ В РУБЛЯХ ЮРИДИЧЕСКИХ ЛИЦ",
    'задолженность_фл': " ЗАДОЛЖЕННОСТЬ В РУБЛЯХ ФИЗИЧЕСКИХ ЛИЦ"
}
table_mapping = {
    'врп': vrp,
    'дкб': dkb,
    'ркб': rkb,
    'пкб_р': pkb_r,
    'пкб_фед': pkb_f,
    'вклады_фл': vk_fl,
    'вклады_юл': vk_ul,
    'задолженность_юл': zd_ul,
    'задолженность_фл': zd_fl
}



def choose_table(selected_method):
    """
    Функция для выбора таблицы/таблиц пользователем.
    В зависимости от метода анализа поддерживает выбор одной или нескольких таблиц.
    """
    if selected_method == "5":  # Анализ взаимосвязей между таблицами
        print('Выберите две или более таблиц для анализа взаимосвязей.')
        selected_tables = input('Введите сокращения таблиц через пробел: ').strip().split()

        # Проверяем выбранные таблицы
        dfs = []
        for table in selected_tables:
            if table in table_mapping:
                dfs.append(table_mapping[table])
            else:
                print(f"Таблица '{table}' не найдена.")

        if len(dfs) < 2:
            print("Необходимо выбрать как минимум две таблицы для анализа взаимосвязей.")
            return None

        # Печатаем полные названия выбранных таблиц
        full_names = [table_fullname.get(tbl, "Неизвестное название") for tbl in selected_tables]
        print("\nВы выбрали следующие таблицы:")
        print(", ".join(full_names))
        return dfs  # Возвращаем список DataFrame'ов

    else:  # Для остальных методов анализа (выбор одной таблицы)
        print('Выберите одну таблицу для анализа.')
        selected_table = input('Введите сокращение таблицы: ').strip()

        if selected_table in table_mapping:
            df = table_mapping[selected_table]
            full_name = table_fullname.get(selected_table, "Неизвестное название")
            print(f"\nВы выбрали таблицу: {full_name}")
            return df  # Возвращаем выбранный DataFrame
        else:
            print(f"Таблица '{selected_table}' не найдена.")
            return None



def menu():
    print('Вы вошли в главное меню')
    while True:
        print('Выбор нужного функционала')
        print('1 - Изменение данных')
        print('2 - Анализ данных')
        selected_number = input('Введите номер нужного функционала: ').strip()
        if selected_number == "1":
            print('Выберите какие изменения вы хотите внести')
            print('1 - Удалить данные')
            print('2 - Добавить данные')
            print('3 - Изменить данные')
            print('4 - Просмотреть данные')
            print('0 - выход из меню')

            selected_method = input('Введите номер метода изменения данных: ')
            if selected_method == "1":
                delete_data()
            elif selected_method == "2":
                add_data()
            elif selected_method == "3":
                edit_data()
            elif selected_method == "4":
                view_data()
            elif selected_method == "0":
                print('Выход из меню')
                break

        elif selected_number == "2":
            print('Выбор метода анализа')
            print('1 - Общая статистика данных')
            print('2 - Корреляционный анализ')
            print('3 - Анализ распределения данных')
            print('4 - Детекция аномалий')
            print('5 - Прогнозирование данных')
            print('0 - выход из меню')

            selected_method = input('Введите номер метода анализа данных: ').strip()

            if selected_method in ["0", "1", "2", "3", "4", "5"]:  # Убедимся, что введена корректная опция

                if selected_method == "1":  # Общая статистика
                    basic_stats()
                elif selected_method == "2":  # Корреляционный анализ
                    correlation_analysis()
                elif selected_method == "3":  # Анализ распределения
                    print("Вы выбрали анализ распределения данных.")
                    data_distribution_analysis()
                elif selected_method == "4":  # Детекция аномалий
                    print("Вы выбрали детекцию аномалий.")
                    anomaly_detection()
                elif selected_method == "5":  # Прогнозирование данных
                    print(
                        "Вы выбрали прогнозирование данных. (Реализация в разработке прогнозирование на основе данных из таблицы .)")
                    data_forecasting()
                elif selected_method == "0":
                    print('Выход из меню')
                break
            else:
                print("Некорректный ввод. Повторите попытку.")
                break


a = menu()