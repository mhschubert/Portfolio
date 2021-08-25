import torch


class PandasDataset(torch.utils.data.Dataset):
    """Dataset class for pandas dataframes."""
    def __init__(self, df_X, df_y, df_label=None):
        self.df_X = torch.tensor(df_X.values.astype(float), dtype=torch.float)
        self.df_y = torch.tensor(df_y.values.astype(float), dtype=torch.float)
        self.df_label = None
        if df_label is not None:
            self.df_label = torch.tensor(df_label.values.astype(float), dtype=torch.float)
    def __len__(self):
        return len(self.df_X)

    def __getitem__(self, idx):
        if self.df_label is not None:
            return self.df_X[idx:idx + 1], self.df_y[idx:idx + 1], self.df_label[idx:idx + 1]
        else:
            return self.df_X[idx:idx + 1], self.df_y[idx:idx + 1]