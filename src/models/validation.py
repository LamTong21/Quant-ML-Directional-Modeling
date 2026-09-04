from sklearn.model_selection import TimeSeriesSplit


class StandardPurgedCV:
    """TimeSeriesSplit cơ bản có purge gap để loại trừ horizon leakage."""

    def __init__(self, n_splits=5, purge_gap=5, max_train_size=None):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits, max_train_size=self.max_train_size
        )
        for tr_idx, te_idx in tscv.split(X):
            tr_purged = tr_idx[tr_idx < (te_idx[0] - self.purge_gap)]
            if len(tr_purged) > 0 and len(te_idx) > 0:
                yield tr_purged, te_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits