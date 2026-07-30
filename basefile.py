# ============================================================
# HGD EDF EEGNET PIPELINE
# WORKING VERSION USING EDF FILES
# ============================================================

# ============================================================
# INSTALL
# ============================================================

# pip install mne
# pip install torch torchvision torchaudio
# pip install numpy scipy scikit-learn

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import mne

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader
)

# ============================================================
# PARAMETERS
# ============================================================

TRAIN_FILE = './hgd/train1/1.edf'
TEST_FILE  = './hgd/test1/1.edf'

FS = 250

N_CLASSES = 4

# ------------------------------------------------------------
# MI INTERVAL
# ------------------------------------------------------------

TMIN = 0.5
TMAX = 3.5

# ------------------------------------------------------------
# FILTER
# ------------------------------------------------------------

LOWCUT  = 4
HIGHCUT = 38

# ------------------------------------------------------------
# CROPPED TRAINING
# ------------------------------------------------------------

WINDOW_SIZE = 250
STEP_SIZE   = 50

# ------------------------------------------------------------

BATCH_SIZE = 32

EPOCHS = 30

LR = 0.0001

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)

print("DEVICE:", DEVICE)

# ============================================================
# LOAD EDF
# ============================================================

def load_edf(file_path):

    print(f"\nLOADING: {file_path}")

    raw = mne.io.read_raw_edf(
        file_path,
        preload=True,
        verbose=False
    )

    print(raw)

    # --------------------------------------------------------
    # RESAMPLE
    # --------------------------------------------------------

    raw.resample(FS)

    # --------------------------------------------------------
    # BANDPASS FILTER
    # --------------------------------------------------------

    raw.filter(
        LOWCUT,
        HIGHCUT,
        fir_design='firwin',
        verbose=False
    )

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    events, event_dict = mne.events_from_annotations(
        raw,
        verbose=False
    )

    print("\nEVENT DICT:")
    print(event_dict)

    # --------------------------------------------------------
    # CREATE LABEL MAP
    # --------------------------------------------------------

    event_keys = list(
        event_dict.keys()
    )

    event_keys = sorted(event_keys)

    label_map = {}

    for idx, key in enumerate(event_keys):

        label_map[
            event_dict[key]
        ] = idx

    print("\nLABEL MAP:")
    print(label_map)

    # --------------------------------------------------------
    # EPOCHS
    # --------------------------------------------------------

    epochs = mne.Epochs(

        raw,

        events,

        event_id=event_dict,

        tmin=TMIN,

        tmax=TMAX,

        baseline=None,

        preload=True,

        verbose=False
    )

    X = epochs.get_data()

    y_raw = epochs.events[:, -1]

    y = np.array([
        label_map[x]
        for x in y_raw
    ])

    print("\nDATA:", X.shape)
    print("LABELS:", y.shape)

    return X, y

# ============================================================
# CROPPED WINDOWS
# ============================================================

def create_windows(
    X,
    y
):

    X_out = []
    y_out = []

    trial_ids = []

    for trial_idx in range(
        len(X)
    ):

        trial = X[trial_idx]

        # ----------------------------------------------------
        # Z-SCORE
        # ----------------------------------------------------

        mean = np.mean(
            trial,
            axis=1,
            keepdims=True
        )

        std = np.std(
            trial,
            axis=1,
            keepdims=True
        )

        trial = (
            trial - mean
        ) / (std + 1e-6)

        # ----------------------------------------------------
        # SLIDING WINDOWS
        # ----------------------------------------------------

        for ws in range(
            0,
            trial.shape[1] - WINDOW_SIZE,
            STEP_SIZE
        ):

            we = ws + WINDOW_SIZE

            window = trial[
                :,
                ws:we
            ]

            X_out.append(window)

            y_out.append(y[trial_idx])

            trial_ids.append(trial_idx)

    return (

        np.array(X_out),

        np.array(y_out),

        np.array(trial_ids)
    )

# ============================================================
# LOAD TRAIN
# ============================================================

X_train_raw, y_train_raw = load_edf(
    TRAIN_FILE
)

X_train, y_train, train_trial_ids = create_windows(
    X_train_raw,
    y_train_raw
)

print("\nTRAIN WINDOWS:", X_train.shape)

# ============================================================
# LOAD TEST
# ============================================================

X_test_raw, y_test_raw = load_edf(
    TEST_FILE
)

X_test, y_test, test_trial_ids = create_windows(
    X_test_raw,
    y_test_raw
)

print("\nTEST WINDOWS:", X_test.shape)

# ============================================================
# PARAMETERS
# ============================================================

N_CHANNELS = X_train.shape[1]

print("\nCHANNELS:", N_CHANNELS)

# ============================================================
# DATASET
# ============================================================

class EEGDataset(Dataset):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx]
        )

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    EEGDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    EEGDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# EEGNET
# ============================================================

class EEGNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # TEMPORAL
        # ----------------------------------------------------

        self.temporal = nn.Sequential(

            nn.Conv1d(
                N_CHANNELS,
                32,
                kernel_size=32,
                padding=16
            ),

            nn.BatchNorm1d(32),

            nn.ELU(),

            nn.AvgPool1d(2),

            nn.Dropout(0.5)
        )

        # ----------------------------------------------------
        # SPATIAL
        # ----------------------------------------------------

        self.spatial = nn.Sequential(

            nn.Conv1d(
                32,
                64,
                kernel_size=16,
                padding=8
            ),

            nn.BatchNorm1d(64),

            nn.ELU(),

            nn.AvgPool1d(4),

            nn.Dropout(0.5)
        )

        # ----------------------------------------------------
        # GAP
        # ----------------------------------------------------

        self.gap = nn.AdaptiveAvgPool1d(1)

        # ----------------------------------------------------
        # FC
        # ----------------------------------------------------

        self.fc = nn.Linear(
            64,
            N_CLASSES
        )

    def forward(self, x):

        x = self.temporal(x)

        x = self.spatial(x)

        x = self.gap(x)

        x = x.squeeze(-1)

        x = self.fc(x)

        return x

# ============================================================
# MODEL
# ============================================================

model = EEGNet().to(
    DEVICE
)

print("\nMODEL:")
print(model)

# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# ============================================================
# TRAINING
# ============================================================

print("\n==============================")
print("TRAINING")
print("==============================")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            X_batch
        )

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, pred = torch.max(
            outputs,
            1
        )

        total += y_batch.size(0)

        correct += (
            pred == y_batch
        ).sum().item()

    train_acc = (
        100 * correct / total
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {running_loss:.4f} "
        f"Train Acc: {train_acc:.2f}%"
    )

# ============================================================
# WINDOW TESTING
# ============================================================

print("\n==============================")
print("WINDOW TESTING")
print("==============================")

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        outputs = model(
            X_batch
        )

        _, pred = torch.max(
            outputs,
            1
        )

        all_preds.extend(
            pred.cpu().numpy()
        )

        all_labels.extend(
            y_batch.numpy()
        )

window_acc = accuracy_score(
    all_labels,
    all_preds
)

print(
    f"\nWINDOW ACCURACY: "
    f"{window_acc*100:.2f}%"
)

# ============================================================
# TRIAL VOTING
# ============================================================

from scipy.stats import mode

trial_preds = []
trial_true  = []

unique_trials = np.unique(
    test_trial_ids
)

for tid in unique_trials:

    idx = np.where(
        test_trial_ids == tid
    )[0]

    preds = np.array(all_preds)[idx]

    true_label = np.array(all_labels)[idx][0]

    voted = mode(
        preds,
        keepdims=False
    ).mode

    trial_preds.append(voted)

    trial_true.append(true_label)

trial_acc = accuracy_score(
    trial_true,
    trial_preds
)

print(
    f"\nTRIAL ACCURACY: "
    f"{trial_acc*100:.2f}%"
)

print("\nTRIAL CLASSIFICATION REPORT:\n")

print(
    classification_report(
        trial_true,
        trial_preds
    )
)