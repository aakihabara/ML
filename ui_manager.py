# Modules connections
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import os
import model_evaluation
import model_train
import model_preprocessing


class ClassificationPrediction:
    # Инициализация настроек окна и переменных, которые используются в разных функциях
    def __init__(self, root):

        # Создание переменных
        self.root = root
        self.root.geometry('1200x600+100+100')  # Базовый размер окна
        self.root.minsize(1200, 600)  # Минимальный размер окна
        self.root.attributes('-fullscreen', False)  # Запуск полноэкранного режима
        self.root.title('Prediction Program')

        self.mainframe = ttk.Frame(self.root, padding=6)
        self.mainframe.grid(column=0, row=0, sticky='nsew')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.classificator_button = None
        self.regressor_button = None
        self.data_table_view = None
        self.Error_text = None
        self.Error_label = None
        self.graphic_show_button = None
        self.choose_file_button = None

        self.data_column_name = StringVar(value='Имя столбца')
        self.predictor_text = StringVar()
        self.iterator = 0
        self.variables_list = []
        self.confirm_button = None
        self.result_value = StringVar()
        self.train_x, self.train_y, self.test_x, self.test_y = None, None, None, None
        self.label_encoders = None
        self.model_method = StringVar()
        self.accuracy_score = StringVar()
        self.abs_err = StringVar()
        self.model_type = StringVar()
        self.datas = None
        self.model_main = None

        self.result_value.set('Результат')

        self.create_widgets()

    def make_none_fullscreen(self, event):
        self.root.attributes('-fullscreen', False)

    def make_fullscreen(self, event):
        self.root.attributes('-fullscreen', True)

    def press_next_step(self, event):
        if self.confirm_button['state'] == 'enabled':
            self.confirm_button.invoke()

    def create_widgets(self):

        # Создание стилей для виджетов
        style = ttk.Style(self.root)
        style.configure('TFrame', background='gray')
        style.configure('TButton', width=20, font=("Arial", 14),
                        foreground="black", background="gray", relief="flat",
                        padding=10)
        style.configure('TRadiobutton', font=("Arial", 14), background='gray', foreground="white")
        style.configure('TLabel', width=20, font=("Arial", 14), background='gray', foreground="white")

        # Инициализация виджетов
        reset_button = ttk.Button(self.mainframe, text="\u27F3", command=self.reset, style='TButton', width=2)
        quit_button = ttk.Button(self.mainframe, text="\u2715", command=self.root.quit, style='TButton', width=2)

        types_label = ttk.Label(self.mainframe, text="Тип", style='TLabel',
                                font=("Arial", 12, 'bold'), anchor=CENTER)

        self.classificator_button = ttk.Radiobutton(self.mainframe, text='Классификация', variable=self.model_type,
                                                    value='classificator', style='TRadiobutton',
                                                    command=self.model_pressed)

        self.regressor_button = ttk.Radiobutton(self.mainframe, text='Регрессия', variable=self.model_type,
                                                value='predictor', style='TRadiobutton',
                                                command=self.model_pressed)

        methods_label = ttk.Label(self.mainframe, text="Метод", style='TLabel',
                                  font=("Arial", 12, 'bold'), anchor=CENTER)

        decision_trees_button = ttk.Radiobutton(self.mainframe, text='Дерево решений', variable=self.model_method,
                                                value='decision_trees', style='TRadiobutton',
                                                command=self.model_pressed)

        random_forest_button = ttk.Radiobutton(self.mainframe, text='Случайный лес', variable=self.model_method,
                                               value='random_forest', style='TRadiobutton',
                                               command=self.model_pressed)

        self.choose_file_button = ttk.Button(self.mainframe, text='Выбрать файл', command=self.choose_file, style='TButton')

        run_script_button = ttk.Button(self.mainframe, text='Предсказать', command=self.run_script, style='TButton')

        self.graphic_show_button = ttk.Button(self.mainframe, text='График', state='disabled',
                                              command=self.init_graphic, style='TButton')

        ttk.Label(self.mainframe, text='Точность:',
                  style='TLabel', anchor=E).grid(column=3, row=2, sticky='ew')
        self.Error_label = ttk.Label(self.mainframe, text='Ошибка:',
                                     style='TLabel', anchor=E)

        ttk.Label(self.mainframe, textvariable=self.accuracy_score,
                  style='TLabel', width=8, anchor=W, relief='groove').grid(column=4, row=2, sticky='w')
        self.Error_text = ttk.Label(self.mainframe, textvariable=self.abs_err,
                                    style='TLabel', width=8, anchor=W, relief='groove')

        self.data_table_view = ttk.Treeview(self.mainframe, show='headings')

        vsb = ttk.Scrollbar(self.mainframe, orient="vertical", command=self.data_table_view.yview)
        hsb = ttk.Scrollbar(self.mainframe, orient="horizontal", command=self.data_table_view.xview)

        column_label = ttk.Label(self.mainframe, style='TLabel', text='Имя столбца', anchor=CENTER,
                                 textvariable=self.data_column_name)
        var_entry = ttk.Entry(self.mainframe, textvariable=self.predictor_text)
        self.confirm_button = ttk.Button(self.mainframe, text='Далее', style='TButton', state='disabled',
                                         command=self.move_to_next_step)
        result_label = ttk.Label(self.mainframe, textvariable=self.result_value, style='TLabel',
                                 relief='groove', anchor=CENTER)

        # Расположение виджетов на сетке окна
        vsb.grid(column=6, row=4, rowspan=4, sticky="nse")
        self.data_table_view.configure(yscrollcommand=vsb.set)

        hsb.grid(column=0, row=8, columnspan=6, sticky="sew")
        self.data_table_view.configure(xscrollcommand=hsb.set)

        reset_button.grid(column=8, row=0, sticky='ew')
        quit_button.grid(column=9, row=0, sticky='ew')
        methods_label.grid(column=1, row=1)
        types_label.grid(column=0, row=1)
        self.classificator_button.grid(column=0, row=2)
        self.regressor_button.grid(column=0, row=3)
        decision_trees_button.grid(column=1, row=2)
        random_forest_button.grid(column=1, row=3)
        self.choose_file_button.grid(column=7, row=1)
        run_script_button.grid(column=7, row=2)
        self.graphic_show_button.grid(column=7, row=3)

        self.Error_label.grid(column=3, row=3, sticky='ew')
        self.Error_text.grid(column=4, row=3, sticky='w')

        self.data_table_view.grid(column=0, row=4, columnspan=6, rowspan=4, sticky='nsew', pady=6, padx=6)

        column_label.grid(column=7, row=4, sticky='sew')
        var_entry.grid(column=7, row=5, sticky='ew')
        self.confirm_button.grid(column=7, row=6, sticky='new')
        result_label.grid(column=7, row=7, sticky='new')

        # Конфигурация столбцов и строк для масштабирования
        for i in range(0, 6):
            self.mainframe.columnconfigure(i, weight=1)

        for i in range(0, 8):
            self.mainframe.rowconfigure(i, weight=1)

        self.mainframe.rowconfigure(7, weight=0)

        for child in self.mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)

        decision_trees_button.invoke()
        self.classificator_button.invoke()
        self.model_pressed()

        # Создание горячих клавиш
        self.root.bind("<Escape>", self.make_none_fullscreen)
        self.root.bind("<F>", self.make_fullscreen)
        self.root.bind("<f>", self.make_fullscreen)
        self.root.bind("<Return>", self.press_next_step)

    def reset(self):
        self.root.destroy()
        new_root = Tk()
        ClassificationPrediction(new_root)
        new_root.mainloop()

    # Выбор файла данных
    def choose_file(self):
        try:
            self.data_table_view.delete(*self.data_table_view.get_children())  # Clear previous treeview
            selected_file = filedialog.askopenfilename(filetypes=(("Excel files", "*.xlsx"),
                                                                  ("CSV files", "*.csv")), title="Select a file")
            file_extension = os.path.splitext(selected_file)[1]  # Get file extension
            match file_extension:
                case '.xlsx':
                    self.datas = pd.read_excel(selected_file)
                case '.csv':
                    self.datas = pd.read_csv(selected_file)
                case _:
                    pass

            self.data_table_view["columns"] = list(self.datas.columns)  # Set treeview columns
            self.data_table_view.heading("#0", text="Index")
            for col in self.datas:
                self.data_table_view.heading(col, text=col)
            for column in self.data_table_view["columns"]:
                self.data_table_view.column(column, stretch=False)  # Configuring columns to disable stretching
            for index, row in self.datas.iterrows():
                self.data_table_view.insert("", END, text=str(index), values=row.tolist())  # Set values for each row

            # Enable and disable buttons depending on result type
            if self.datas.iloc[:, -1].dtype == 'int64' or self.datas.iloc[:, -1].dtype == 'float64':
                self.classificator_button['state'] = 'disabled'
                self.regressor_button['state'] = 'enabled'
                self.regressor_button.invoke()
            else:
                self.classificator_button['state'] = 'enabled'
                self.regressor_button['state'] = 'disabled'
                self.classificator_button.invoke()

            self.model_pressed()
            self.graphic_show_button['state'] = 'enabled'
        except ValueError:
            messagebox.showinfo("Ошибка в файле",
                                "Вы должны выбрать файл форматов .csv или .xlsx с правильными данными")

    def init_graphic(self):
        model_evaluation.print_graphic(self.test_x, self.test_y, self.model_type.get(), self.model_main)

    # Создание и обучение модели
    def model_pressed(self):
        if self.datas is None:
            return
        else:
            self.model_main = None
            preproc_list, self.label_encoders = model_preprocessing.preprocessing(self.datas.copy())
            self.train_x, self.test_x, self.train_y, self.test_y = preproc_list
            self.model_main = model_train.training(self.train_x, self.train_y,
                                                   self.model_type.get(), self.model_method.get())
            model_train.dump_model_files(self.model_main, self.label_encoders)
            metrics_list = model_evaluation.get_metrics(self.test_x, self.test_y,
                                                        self.model_type.get(), self.model_main)
            if self.model_type.get() == 'classificator':
                self.abs_err.set('')
                self.Error_label['state'] = 'disabled'
                self.Error_text['state'] = 'disabled'
            else:
                self.abs_err.set(f'{metrics_list[1]:.2f}')
                self.Error_label['state'] = 'enabled'
                self.Error_text['state'] = 'enabled'
            self.accuracy_score.set(f'{metrics_list[0]:.2f}')

    # Добавление значение в список для предсказания
    def move_to_next_step(self):
        if self.predictor_text.get() == '':
            messagebox.showerror('Числовая ошибка', 'Вы должны ввести данные в поле ввода')
            return None
        else:
            self.variables_list.append(self.predictor_text.get())
            self.predictor_text.set('')
        if self.iterator == len(self.datas.columns) - 2:
            self.reset_to_default()
            return None
        self.iterator += 1
        self.data_column_name.set(f'{self.datas.columns[self.iterator]}')
        if self.iterator == len(self.datas.columns) - 2:
            self.confirm_button.configure(text='Конец')

    # Получение результата предсказания и установка стандартных значений
    def reset_to_default(self):
        try:
            res_value = model_train.predict_value(self.variables_list, self.datas.copy(), self.model_main,
                                                  self.label_encoders, self.model_type.get())
            self.result_value.set(f'{res_value}')
        except ValueError:
            messagebox.showinfo("Числовая ошибка", "Попробуйте использовать другие значения или измените метод модели")
            self.result_value.set('Неизвестно')
        finally:
            self.variables_list = []
            self.iterator = 0
            self.confirm_button['state'] = 'disabled'
            self.confirm_button.configure(text='Далее')
            self.data_column_name.set('Имя столбца')
            self.choose_file_button['state'] = 'enabled'

    # Начало процесса предсказания
    def run_script(self):

        if self.datas is None:
            messagebox.showerror('Отсутствуют данные', 'Для начала предсказания, вы должны выбрать файл данных')
            return

        self.result_value.set('Результат')
        self.confirm_button['state'] = 'enabled'
        self.choose_file_button['state'] = 'disabled'
        self.data_column_name.set(f'{self.datas.columns[self.iterator]}')


# Создание экземпляра окна и класса программы
main_root = Tk()
ClassificationPrediction(main_root)
main_root.mainloop()
