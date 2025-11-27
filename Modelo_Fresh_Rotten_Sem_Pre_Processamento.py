import os
import shutil
import uuid
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import cv2
import kagglehub
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import time
import seaborn as sns
import hashlib

# ===============================================================================
# RODAR NO GOOGLE COLAB
# https://colab.research.google.com/drive/1lYgMpwM4NhicY_YXIexAMVBpbei96rYx?usp=sharing
# ===============================================================================

# ===============================================================================
# 0. PACOTES OBRIGATÓRIOS
#   pip install numpy
#   pip install matplotlib
#   pip install scikit-learn
#   pip install opencv-python
#   pip install seaborn
#   pip install kagglehub
#   
#   pip install numpy matplotlib scikit-learn opencv-python seaborn kagglehub
# 
#   Python 3.13.2:
#       python --version
# ===============================================================================

# ============================================================== 
# 1. BAIXAR DATASET
# ==============================================================

print("📥 Baixando dataset via kagglehub...")

dataset_path = kagglehub.dataset_download(
    "narayanibokde/augmented-dataset-for-fruits-rottenfresh"
)

print("📦 Dataset baixado em:", dataset_path)


# -----------------------------
# LIMPEZA DE VERSÕES ANTIGAS (MANTIDO)
# -----------------------------

def _find_kagglehub_version_parent(path):
    cur = os.path.dirname(path)
    while True:
        try:
            entries = os.listdir(cur)
        except Exception:
            return None
        versions = [e for e in entries if e.startswith("v")]
        if len(versions) > 1:
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent

def limpar_kagglehub_versions(path_dataset):
    base_dir = _find_kagglehub_version_parent(path_dataset)
    if base_dir is None:
        print("Nenhuma pasta de versões do KaggleHub encontrada.")
        return

    versões = sorted([v for v in os.listdir(base_dir) if v.startswith("v")],
                     reverse=True)

    if len(versões) <= 1:
        print("Nenhuma versão extra do KaggleHub para limpar.")
        return

    print("\n🧹 Limpando versões antigas do KaggleHub...\n")

    versões_para_apagar = versões[1:]
    for v in versões_para_apagar:
        caminho = os.path.join(base_dir, v)
        try:
            shutil.rmtree(caminho, ignore_errors=True)
            print(f"   🔥 Removido: {caminho}")
        except Exception as e:
            print(f"   ⚠ Falha ao remover {caminho}: {e}")

    print("\n✔ Versões antigas removidas!\n")


try:
    limpar_kagglehub_versions(dataset_path)
except Exception as e:
    print("⚠ Erro na limpeza automática:", e)


# ============================================================== 
# 2. ENCONTRAR PASTAS ORIGINAIS
# ==============================================================

def localizar_pastas_brutas(path):
    candidatos = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            nome = d.lower()
            if nome.startswith("fresh") or nome.startswith("rotten"):
                candidatos.append(os.path.join(root, d))
    return candidatos

originais = localizar_pastas_brutas(dataset_path)

print("\n📂 Pastas detectadas:")
for p in originais:
    print(" -", p)


# ============================================================== 
# 3. ORGANIZAR EM fresh / rotten (com deduplicação por hash)
# ==============================================================

BASE_ORGANIZADA = os.path.join(dataset_path, "ORGANIZADO")
fresh_dir  = os.path.join(BASE_ORGANIZADA, "fresh")
rotten_dir = os.path.join(BASE_ORGANIZADA, "rotten")

shutil.rmtree(BASE_ORGANIZADA, ignore_errors=True)
os.makedirs(fresh_dir, exist_ok=True)
os.makedirs(rotten_dir, exist_ok=True)

print("\n🧹 Reorganizando imagens...")

# --- Função MD5 ---
def file_md5(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def build_hash_set(folder):
    hashes = set()
    if not os.path.exists(folder):
        return hashes
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if os.path.isfile(p) and f.lower().endswith((".jpg",".jpeg",".png")):
            try:
                hashes.add(file_md5(p))
            except Exception:
                pass
    return hashes

hashes_fresh  = build_hash_set(fresh_dir)
hashes_rotten = build_hash_set(rotten_dir)

def copiar_sem_duplicar(origem, destino_dir, hashes_set):
    try:
        md5 = file_md5(origem)
    except Exception:
        return False

    if md5 in hashes_set:
        return False

    novo_nome = f"{uuid.uuid4().hex}.jpg"
    destino = os.path.join(destino_dir, novo_nome)
    shutil.copy2(origem, destino)
    hashes_set.add(md5)
    return True


total_fresh = 0
total_rotten = 0
origem_log = {}


def mover_imagens(pasta):
    global total_fresh, total_rotten

    nome = os.path.basename(pasta).lower()
    destino = fresh_dir if nome.startswith("fresh") else rotten_dir
    hashes = hashes_fresh if nome.startswith("fresh") else hashes_rotten

    arquivos = [f for f in os.listdir(pasta)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    copiados = 0
    for f in arquivos:
        origem = os.path.join(pasta, f)
        if copiar_sem_duplicar(origem, destino, hashes):
            copiados += 1

    origem_log[pasta] = copiados

    if nome.startswith("fresh"):
        total_fresh += copiados
    else:
        total_rotten += copiados


for pasta in originais:
    mover_imagens(pasta)

print("\n✔ Organização concluída!")
print(f"➡ fresh novas:  {total_fresh}")
print(f"➡ rotten novas: {total_rotten}")

print("\n📋 Detalhamento:")
for pasta, qtd in origem_log.items():
    print(f" - {os.path.basename(pasta)}: {qtd} novas")


# ============================================================== 
# 4. CRIAR train / val / test (limpeza mantida)
# ==============================================================

base_out = os.path.join(dataset_path, "DIVIDIDO_RAW")

train_path = os.path.join(base_out, "train")
val_path   = os.path.join(base_out, "val")
test_path  = os.path.join(base_out, "test")

shutil.rmtree(base_out, ignore_errors=True)

for f in [train_path, val_path, test_path]:
    os.makedirs(os.path.join(f, "fresh"), exist_ok=True)
    os.makedirs(os.path.join(f, "rotten"), exist_ok=True)


def split_dataset(src, train_dir, val_dir, test_dir, split=(0.7, 0.15, 0.15)):
    files = [f for f in os.listdir(src)
             if f.lower().endswith(("jpg","png","jpeg"))]

    random.shuffle(files)
    total = len(files)
    n_train = int(split[0] * total)
    n_val   = int(split[1] * total)

    train_files = files[:n_train]
    val_files   = files[n_train:n_train+n_val]
    test_files  = files[n_train+n_val:]

    for f in train_files:
        shutil.copy2(os.path.join(src, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy2(os.path.join(src, f), os.path.join(val_dir, f))
    for f in test_files:
        shutil.copy2(os.path.join(src, f), os.path.join(test_dir, f))


split_dataset(fresh_dir,
              os.path.join(train_path, "fresh"),
              os.path.join(val_path,   "fresh"),
              os.path.join(test_path,  "fresh"))

split_dataset(rotten_dir,
              os.path.join(train_path, "rotten"),
              os.path.join(val_path,   "rotten"),
              os.path.join(test_path,  "rotten"))


# ============================================================== 
# 5. CARREGAMENTO RAW (SEM PRÉ-PROCESSAMENTO)
# ==============================================================

IMG_SIZE = (128, 128)

def load_raw(path_label):
    path, label = path_label
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    return img.reshape(-1), label


def carregar_pasta(base, label):
    files = [(os.path.join(base, f), label)
             for f in os.listdir(base)
             if f.lower().endswith(("jpg","jpeg","png"))]

    X, y = [], []
    with ThreadPoolExecutor(max_workers=8) as exe:
        futures = [exe.submit(load_raw, fl) for fl in files]
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                img, lbl = r
                X.append(img)
                y.append(lbl)

    return np.array(X), np.array(y)


def carregar_dataset(base):
    xf, yf = carregar_pasta(os.path.join(base, "fresh"), 0)
    xr, yr = carregar_pasta(os.path.join(base, "rotten"), 1)
    return np.concatenate([xf, xr]), np.concatenate([yf, yr])


print("\n⚡ Carregando imagens (RAW)...")

X_train, y_train = carregar_dataset(train_path)
X_val,   y_val   = carregar_dataset(val_path)
X_test,  y_test  = carregar_dataset(test_path)


# ============================================================== 
# 6. PCA + SVM (para comparar diretamente)
# ==============================================================

pipeline = Pipeline([
    ("pca", PCA(n_components=150, whiten=True, random_state=42)),
    ("svm", LinearSVC(C=1.0, max_iter=6000, dual=False))
])

print("\n⏳ Treinando PCA+SVM (RAW)...")
start = time.time()
pipeline.fit(X_train, y_train)
end = time.time()

print(f"\n⏳ Tempo total: {end - start:.2f}s")
print(f"🎯 Acurácia RAW: {pipeline.score(X_test, y_test):.4f}")


# ============================================================== 
# 7. MÉTRICAS
# ==============================================================

scores = pipeline.decision_function(X_test)
y_pred = (scores > 0).astype(int)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred, target_names=["Fresh", "Rotten"]))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão (RAW)")
plt.show()

fpr, tpr, _ = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.title("Curva ROC (RAW)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()

#=========================================================================================
# O dataset "Fruits Fresh & Rotten" tem:
#   Fundo muito homogêneo
#   Iluminação bem controlada
#   Frutas bem centralizadas
#   Pouca sujeira visual
#   Poucas variações severas de escala/rotação
#   Classes com padrões de textura muito distintos
# Isso significa que mesmo imagens pouco tratadas já são separáveis.
# #=========================================================================================