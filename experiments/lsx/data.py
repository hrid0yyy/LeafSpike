"""Dataset discovery, duplicate-aware split (saved once, reused by every job), augmentation, datasets.

Dataset rules (see README.md): tomato only, duplicate-aware stratified 70/15/15 split, macro-F1 reported.
"""
import hashlib
import io
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedGroupKFold
try:
    from torchvision import transforms as T
except ImportError:
    T = None

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

try:
    import cv2
    cv2.setNumThreads(0)
    HAS_CV2 = True
except Exception:  # pragma: no cover
    HAS_CV2 = False

CLASSES = ['bacterial_spot', 'early_blight', 'late_blight', 'leaf_mold', 'septoria_leaf_spot',
           'spider_mites', 'target_spot', 'yellow_leaf_curl_virus', 'mosaic_virus', 'healthy']
C2I = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
SKIP_KEYS = ['grayscale', 'segmented', '__macosx']
CLASS_RULES = [('healthy', ['healthy']), ('bacterial_spot', ['bacterial']), ('early_blight', ['early']),
               ('late_blight', ['late']), ('leaf_mold', ['mold', 'mould']),
               ('septoria_leaf_spot', ['septoria']), ('spider_mites', ['spider', 'mite']),
               ('target_spot', ['target']), ('yellow_leaf_curl_virus', ['yellow', 'curl']),
               ('mosaic_virus', ['mosaic'])]


def norm_name(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


def map_tomato_class(dirname, path_l, kind):
    n = norm_name(dirname)
    if 'tomato' not in n and not (kind == 'field' and 'tomato' in path_l):
        return None
    for cls, keys in CLASS_RULES:
        if any(k in n for k in keys):
            return cls
    if n in ('tomato leaf', 'tomato leaves', 'tomato'):
        return 'healthy'   # PlantDoc names its healthy class "Tomato leaf"
    return None


def split_hint(rel_dir):
    parts = [p.lower() for p in rel_dir.split(os.sep)]
    if any(p in ('test', 'testing') for p in parts):
        return 'test'
    if any(p in ('val', 'valid', 'validation') for p in parts):
        return 'val'
    return 'train'


def discover(root, kind):
    """All tomato class folders under `root` (kind = 'lab' or 'field')."""
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f'{kind} data folder not found: {root}')
    rows, ignored = [], Counter()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        imgs = sorted(f for f in filenames if os.path.splitext(f)[1].lower() in IMG_EXTS)
        if not imgs:
            continue
        path_l = dirpath.lower().replace('\\', '/')
        if any(k in path_l for k in SKIP_KEYS):
            continue
        cls = map_tomato_class(os.path.basename(dirpath), path_l, kind)
        if cls is None:
            if 'tomato' in path_l:
                ignored[dirpath] += len(imgs)
            continue
        hint = split_hint(os.path.relpath(dirpath, root))
        for f in imgs:
            rows.append(dict(path=os.path.join(dirpath, f), label=C2I[cls], cls=cls, split_hint=hint))
    for d, n in ignored.items():
        print(f'   [info] tomato-looking folder not used ({n} images): {d}')
    return pd.DataFrame(rows, columns=['path', 'label', 'cls', 'split_hint'])


def _readable(p):
    try:
        with Image.open(p) as im:
            im.size
        return True
    except Exception:
        return False


def _md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _parallel(fn, items):
    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(fn, items))


def build_lab_split(lab_root, out_csv, test_frac=0.15, val_frac=0.15, seed=42, max_per_class=None):
    """Create the duplicate-aware stratified split once and save it; later calls just load it."""
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        df['path'] = [os.path.join(lab_root, p) for p in df.relpath]
        missing = (~df.path.map(os.path.exists)).sum()
        if missing:
            raise FileNotFoundError(f'{missing} files listed in {out_csv} are missing under {lab_root}. '
                                    'Restore the data, or delete the split file to rebuild it '
                                    '(this changes the split for every model).')
        return df
    df = discover(lab_root, 'lab')
    if len(df) == 0:
        raise RuntimeError(f'No PlantVillage tomato folders found under {lab_root}')
    ok = np.array(_parallel(_readable, df.path.tolist()))
    if (~ok).sum():
        print(f'   dropping {(~ok).sum()} unreadable files')
    df = df[ok].reset_index(drop=True)
    print(f'   found {len(df)} tomato images; hashing for duplicates...')
    df['md5'] = _parallel(_md5, df.path.tolist())
    df['fname'] = df.path.map(os.path.basename)
    n0 = len(df)
    df = df.drop_duplicates('md5').drop_duplicates(['cls', 'fname']).reset_index(drop=True)
    print(f'   removed {n0 - len(df)} duplicates -> {len(df)} images')
    if max_per_class:
        df = df.sample(frac=1.0, random_state=seed).groupby('label').head(max_per_class)
        df = df.sort_values('path').reset_index(drop=True)
    groups = df.md5.values   # unique after de-duplication; kept as a group key for clarity

    def gsplit(d, g, frac, s):
        k = max(2, int(round(1.0 / frac)))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=s)
        a, b = next(sgkf.split(np.zeros(len(d)), d.label.values, g))
        return a, b

    tv, te = gsplit(df, groups, test_frac, seed)
    df['split'] = 'test'
    tv_df = df.iloc[tv]
    tr, va = gsplit(tv_df, groups[tv], val_frac / (1 - test_frac), seed + 1)
    df.loc[tv_df.index[tr], 'split'] = 'train'
    df.loc[tv_df.index[va], 'split'] = 'val'
    df['relpath'] = [os.path.relpath(p, lab_root) for p in df.path]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df[['relpath', 'label', 'cls', 'md5', 'split']].to_csv(out_csv + '.tmp', index=False)
    os.replace(out_csv + '.tmp', out_csv)
    return df


def load_field(field_root):
    if not field_root:
        return None
    df = discover(field_root, 'field')
    if len(df) == 0:
        print(f'   [warn] no tomato folders found in field root {field_root}')
        return None
    ok = np.array(_parallel(_readable, df.path.tolist()))
    return df[ok].reset_index(drop=True)


# ---------------------------------------------------------------- augmentation

def leaf_mask(rgb):
    """Leaf mask for a PlantVillage-style photo (uniform backdrop); None if segmentation fails."""
    if not HAS_CV2:
        return None
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = cv2.GaussianBlur(hsv[..., 1], (5, 5), 0)
    _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = np.where(lab == idx, 255, 0).astype(np.uint8)
    h, w = m.shape
    for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if m[y, x] == 0:
            ff = m.copy()
            cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (x, y), 255)
            m = cv2.bitwise_or(m, cv2.bitwise_not(ff))
            break
    frac = float((m > 0).mean())
    if frac < 0.05 or frac > 0.9:
        return None
    return m > 0


BG_PALETTE = np.array([[60, 90, 40], [110, 85, 60], [140, 140, 135], [90, 120, 70], [170, 160, 130],
                       [40, 60, 35], [120, 150, 190], [200, 200, 195], [80, 70, 55], [150, 175, 110]], np.float32)


def synth_background(h, w, rng):
    g = int(rng.integers(2, 9))
    cols = BG_PALETTE[rng.integers(0, len(BG_PALETTE), size=(g, g))] + rng.normal(0, 12, size=(g, g, 3))
    bg = cv2.resize(cols.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.normal(0, 1, size=(max(2, h // 8), max(2, w // 8))).astype(np.float32), (w, h),
                      interpolation=cv2.INTER_LINEAR)[..., None]
    bg = bg + fine * rng.uniform(0, 25) + rng.normal(0, rng.uniform(0, 6), size=(h, w, 1))
    if rng.random() < 0.5:
        bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=float(rng.uniform(0.5, 3)))
    return np.clip(bg, 0, 255).astype(np.uint8)


def paste_on_background(rgb, mask, rng):
    h, w = mask.shape
    bg = synth_background(h, w, rng)
    s = float(rng.uniform(0.7, 1.0))
    nh, nw = max(8, int(h * s)), max(8, int(w * s))
    fg = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize(mask.astype(np.uint8) * 255, (nw, nh), interpolation=cv2.INTER_NEAREST)
    a = (cv2.GaussianBlur(m, (5, 5), 0).astype(np.float32) / 255.0)[..., None]
    y0, x0 = int(rng.integers(0, h - nh + 1)), int(rng.integers(0, w - nw + 1))
    region = bg[y0:y0 + nh, x0:x0 + nw].astype(np.float32)
    bg[y0:y0 + nh, x0:x0 + nw] = np.clip(a * fg + (1 - a) * region, 0, 255).astype(np.uint8)
    return bg


class RandomJPEG:
    def __init__(self, p=0.2, q=(30, 95)):
        self.p, self.q = p, q

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=random.randint(*self.q))
        buf.seek(0)
        return Image.open(buf).convert('RGB')


def make_train_tf(size, mean, std):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.35, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.1, hue=0.02),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
        RandomJPEG(p=0.2),
        T.ToTensor(), T.Normalize(mean, std),
        T.RandomErasing(p=0.25, value='random'),
    ])


def make_eval_tf(size, mean, std):
    return T.Compose([T.Resize(int(round(size * 232 / 224)), interpolation=T.InterpolationMode.BICUBIC),
                      T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])


class LeafDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform, bg_paste_prob=0.0):
        self.paths = df.path.tolist()
        self.labels = df.label.astype(int).tolist()
        self.tf = transform
        self.p = bg_paste_prob if HAS_CV2 else 0.0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                img = im.convert('RGB')
        except Exception:
            img = Image.new('RGB', (256, 256))
        if self.p > 0 and random.random() < self.p:
            rgb = np.asarray(img)
            m = leaf_mask(rgb)
            if m is not None:
                rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
                img = Image.fromarray(paste_on_background(rgb, m, rng))
        return self.tf(img), self.labels[i]
