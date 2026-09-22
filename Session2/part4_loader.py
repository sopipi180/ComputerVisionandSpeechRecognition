import math
import numpy as np
from common.config import TARGET
from common.milk10k import available_subset
from part3_processing import process_batch


class MilkDataLoader:
    """
    loader = MilkDataLoader(df, label_col="diagnosis_1", batch_size=32,
                            shuffle=True, size=(224, 224), color="rgb", norm="minmax")

    for images, labels, ids in loader:
        images: (batch, H, W, 3) float array
        labels: class numbers - loader.classes[n] gives the name
        ids:    isic_id of each image

    Only images found on disk are used. Images are loaded one batch at a time.
    """

    def __init__(self, df, label_col=TARGET, batch_size=32, shuffle=True,
                 seed=0, **options):
        self.label_col = label_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.options = options  # passed on to preprocess_image
        self.rng = np.random.default_rng(seed)

        self.df = available_subset(df.dropna(subset=[label_col]))
        self.classes = sorted(self.df[label_col].unique())
        self.skipped = []

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __iter__(self):
        order = self.rng.permutation(len(self.df)) if self.shuffle else range(len(self.df))
        order = list(order)
        for start in range(0, len(order), self.batch_size):
            rows = self.df.iloc[order[start:start + self.batch_size]]
            images, ids, skipped = process_batch(rows, **self.options)
            self.skipped += skipped
            if not ids:
                continue
            names = rows.set_index("isic_id").loc[ids, self.label_col]
            labels = np.array([self.classes.index(n) for n in names])
            yield images, labels, ids