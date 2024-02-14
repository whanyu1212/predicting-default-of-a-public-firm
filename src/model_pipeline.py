class ModelPipeline:
    def __init__(self, df, splitting_date, target_column):
        self.df = df
        self.splitting_date = splitting_date
        self.target_column = target_column

    def split_data(self):
        train = self.df[self.df["Date"] < self.splitting_date]
        test = self.df[self.df["Date"] >= self.splitting_date]
        return train, test
