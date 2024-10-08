import pandas as pd
import random


class BlackBoxModel:
    def __init__(self, data_range, model, feature_list):
        self.no_attr = len(data_range)
        self.data_range = data_range  # e.g., [[1, 2], [3, 4]]
        self.model = model
        self.feature_list = feature_list

    @classmethod
    def create_data_range_from_csv(cls, file_name):
        data_range = []
        df = pd.read_csv(file_name)
        number_attr = df.shape[1] - 1
        for i in range(0, number_attr):
            min_ = df.iloc[:, i].min()
            max_ = df.iloc[:, i].max()
            data_range += [[min_, max_]]
        return data_range, df

    @classmethod
    def create_data_unique_list_from_csv(cls, file_name):
        data_u = []
        df = pd.read_csv(file_name)
        number_attr = df.shape[1] - 1
        for i in range(0, number_attr):
            data_u += [[sorted(df.iloc[:, i].unique())]]
        return data_u

    def predict(self, inputs):
        inputs = [[int(item) for item in row] for row in inputs]
        outputs = self.model.predict(inputs)
        return outputs

    def predict_proba(self, inputs):
        inputs = [[int(item) for item in row] for row in inputs]
        outputs = self.model.predict_proba(inputs)
        return outputs

    def generate_random_inputs(self, num):
        train_data = list()
        for _ in range(num):
            temp = list()
            for i in range(self.no_attr):
                temp.append(random.randint(self.data_range[i][0], self.data_range[i][1]))
            temp.append(int(self.predict([temp])))
            train_data.append(temp)
        return train_data

    def generate_random_inputs_with_random_outputs(self, num):
        train_data = list()
        for _ in range(num):
            temp = list()
            for i in range(self.no_attr):
                temp.append(random.randint(self.data_range[i][0], self.data_range[i][1]))
            temp.append(int(0 if random.random() < 0.5 else 1))
            train_data.append(temp)
        return train_data
