





# # ===========================================================================================
###############################################################################################
# VERSÃO PARA RODAR NO GOOGLE COLAB !!!!!!!!!!!!!!!!!!!!!!!!
# https://colab.research.google.com/drive/1qGWvekc4GR4pXq_vz3xFTGpCMO8AU3Rj?usp=sharing
###############################################################################################
# =============================================================================================










# import os
# import cv2
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from sklearn.decomposition import PCA
# from sklearn.svm import LinearSVC
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, recall_score, precision_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# from skimage.feature import local_binary_pattern
# import shutil
# import random
# import hashlib
# import uuid
# import kagglehub
# # ======================
# # CONFIGURAÇÕES
# # ======================
# IMG_SIZE = (128, 128)
# LBP_P, LBP_R = 8, 1

# print("📥 Baixando dataset via kagglehub...")

# dataset_path = kagglehub.dataset_download(
#     "narayanibokde/augmented-dataset-for-fruits-rottenfresh"
# )

# print("📦 Dataset baixado em:", dataset_path)

# # ======================
# # FUNÇÕES DE PRÉ-PROCESSAMENTO
# # ======================
# # -----------------------------
# # Limpeza automática de versões antigas do kagglehub
# # -----------------------------
# def _find_kagglehub_version_parent(path):
#     cur = os.path.dirname(path)
#     # sobe até raiz, procurando pasta que contenha entradas 'v*'
#     while True:
#         try:
#             entries = os.listdir(cur)
#         except Exception:
#             return None
#         versions = [e for e in entries if e.startswith("v")]
#         if len(versions) > 1:
#             return cur
#         parent = os.path.dirname(cur)
#         if parent == cur:
#             return None
#         cur = parent

# def limpar_kagglehub_versions(path_dataset):
#     base_dir = _find_kagglehub_version_parent(path_dataset)
#     if base_dir is None:
#         print("Nenhuma pasta de versões do KaggleHub encontrada para limpeza.")
#         return

#     versões = sorted(
#         [v for v in os.listdir(base_dir) if v.startswith("v")],
#         reverse=True
#     )

#     if len(versões) <= 1:
#         print("Nenhuma versão extra do KaggleHub para limpar.")
#         return

#     print("\n🧹 Limpando versões antigas do KaggleHub...\n")

#     # Mantém só a mais recente
#     versões_para_apagar = versões[1:]

#     for v in versões_para_apagar:
#         caminho = os.path.join(base_dir, v)
#         try:
#             shutil.rmtree(caminho, ignore_errors=True)
#             print(f"   🔥 Removido: {caminho}")
#         except Exception as e:
#             print(f"   ⚠ Falha ao remover {caminho}: {e}")

#     print("\n✔ Versões antigas removidas com sucesso!\n")

# try:
#     limpar_kagglehub_versions(dataset_path)
# except Exception as e:
#     print("⚠ Erro ao tentar limpar versões do KaggleHub:", e)


# # ==============================================================
# # 2. ENCONTRAR PASTAS ORIGINAIS
# # ==============================================================
# def localizar_pastas_brutas(path):
#     candidatos = []
#     for root, dirs, files in os.walk(path):
#         for d in dirs:
#             nome = d.lower()
#             if nome.startswith("fresh") or nome.startswith("rotten"):
#                 candidatos.append(os.path.join(root, d))
#     return candidatos

# originais = localizar_pastas_brutas(dataset_path)

# print("\n📂 Pastas detectadas:")
# for p in originais:
#     print(" -", p)


# # ==============================================================
# # 3. ORGANIZAR EM fresh/ rotten/ (com limpeza antes)
# # ==============================================================
# BASE_ORGANIZADA = "/content/ORGANIZADO"
# fresh_dir  = os.path.join(BASE_ORGANIZADA, "fresh")
# rotten_dir = os.path.join(BASE_ORGANIZADA, "rotten")

# # 🔥 LIMPEZA PARA EVITAR DUPLICAÇÃO
# shutil.rmtree(BASE_ORGANIZADA, ignore_errors=True)
# os.makedirs(fresh_dir, exist_ok=True)
# os.makedirs(rotten_dir, exist_ok=True)

# print("\n🧹 Limpando e reorganizando imagens...")

# # -----------------------------
# # Funções para copiar sem duplicar
# # -----------------------------
# def file_md5(path, chunk_size=8192):
#     h = hashlib.md5()
#     with open(path, "rb") as f:
#         for chunk in iter(lambda: f.read(chunk_size), b""):
#             h.update(chunk)
#     return h.hexdigest()

# def build_hash_set(folder):
#     hashes = set()
#     if not os.path.exists(folder):
#         return hashes
#     for f in os.listdir(folder):
#         p = os.path.join(folder, f)
#         if os.path.isfile(p) and f.lower().endswith((".jpg",".jpeg",".png")):
#             try:
#                 hashes.add(file_md5(p))
#             except Exception:
#                 # ignora arquivos ilegíveis
#                 pass
#     return hashes

# hashes_fresh = build_hash_set(fresh_dir)
# hashes_rotten = build_hash_set(rotten_dir)

# def copiar_sem_duplicar_por_hash(origem, destino_dir, hashes_set):
#     novo_nome = f"{uuid.uuid4().hex}.jpg"
#     destino_final = os.path.join(destino_dir, novo_nome)
#     shutil.copy2(origem, destino_final)
#     return True

# total_copiadas_fresh = 0
# total_copiadas_rotten = 0
# copiados_por_origem = {}

# def mover_imagens(pasta):
#     global total_copiadas_fresh, total_copiadas_rotten
#     nome = os.path.basename(pasta).lower()
#     if nome.startswith("fresh"):
#         destino = fresh_dir
#         hashes_set = hashes_fresh
#     else:
#         destino = rotten_dir
#         hashes_set = hashes_rotten

#     arquivos = [f for f in os.listdir(pasta) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#     copiadas = 0

#     for f in arquivos:
#         origem = os.path.join(pasta, f)
#         if copiar_sem_duplicar_por_hash(origem, destino, hashes_set):
#             copiadas += 1

#     copiados_por_origem[pasta] = copiadas
#     if nome.startswith("fresh"):
#         total_copiadas_fresh += copiadas
#     else:
#         total_copiadas_rotten += copiadas

# for pasta in originais:
#     mover_imagens(pasta)

# print("\n✔ Reorganização concluída!")
# print(f"➡ fresh/ (novas cópias):  {total_copiadas_fresh} imagens")
# print(f"➡ rotten/ (novas cópias): {total_copiadas_rotten} imagens")

# # Exibe detalhamento por pasta de origem
# print("\n📋 Detalhe de cópias por pasta de origem:")
# for origem_pasta, qtd in copiados_por_origem.items():
#     print(f" - {os.path.basename(origem_pasta)}: {qtd} imagens copiadas")


# # ==============================================================
# # 4. CRIAR train/ val/ test (com limpeza antes)
# # ==============================================================
# base_out = "/content/dataset"

# train_path = os.path.join(base_out, "train")
# val_path   = os.path.join(base_out, "val")
# test_path  = os.path.join(base_out, "test")

# # 🔥 LIMPA TUDO ANTES DE CRIAR
# shutil.rmtree(base_out, ignore_errors=True)

# for folder in [train_path, val_path, test_path]:
#     os.makedirs(os.path.join(folder, "fresh"), exist_ok=True)
#     os.makedirs(os.path.join(folder, "rotten"), exist_ok=True)


# def split_dataset(src, train_dir, val_dir, test_dir, split=(0.7, 0.15, 0.15)):
#     files = [f for f in os.listdir(src) if f.lower().endswith(("jpg","png","jpeg"))]
#     random.shuffle(files)

#     total = len(files)
#     n_train = int(split[0] * total)
#     n_val   = int(split[1] * total)

#     train_files = files[:n_train]
#     val_files   = files[n_train:n_train+n_val]
#     test_files  = files[n_train+n_val:]

#     for f in train_files:
#         shutil.copy2(os.path.join(src, f), os.path.join(train_dir, f))

#     for f in val_files:
#         shutil.copy2(os.path.join(src, f), os.path.join(val_dir, f))

#     for f in test_files:
#         shutil.copy2(os.path.join(src, f), os.path.join(test_dir, f))


# split_dataset(fresh_dir,  os.path.join(train_path,"fresh"),
#                            os.path.join(val_path,"fresh"),
#                            os.path.join(test_path,"fresh"))

# split_dataset(rotten_dir, os.path.join(train_path,"rotten"),
#                            os.path.join(val_path,"rotten"),
#                            os.path.join(test_path,"rotten"))

# def augment_image(img):
#     img = (img * 255).astype(np.uint8)
#     rows, cols, _ = img.shape

#     angle = np.random.uniform(-20, 20)
#     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
#     img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

#     if np.random.rand() > 0.5:
#         img = cv2.flip(img, 1)  # horizontal
#     if np.random.rand() > 0.5:
#         img = cv2.flip(img, 0)  # vertical

#     beta = np.random.randint(-30, 30)
#     img = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)

#     img = img.astype("float32") / 255.0
#     return img

# def segment_fruit(img):
#     hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
#     h, s, v = cv2.split(hsv)

#     # Máscara baseada em saturação e valor
#     mask = cv2.inRange(hsv, (0, 30, 30), (179, 255, 255))

#     # Morfologia para limpar ruído
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#     # Aplicar máscara
#     result = cv2.bitwise_and((img*255).astype(np.uint8), (img*255).astype(np.uint8), mask=mask)
#     result = cv2.resize(result, IMG_SIZE)
#     result = result.astype("float32") / 255.0
#     return result

# def apply_clahe(img, augment=False):
#     lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     l_eq = clahe.apply(l)

#     lab_eq = cv2.merge([l_eq, a, b])
#     img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
#     return img_eq.astype("float32") / 255.0

# def preprocess_image(path, augment=False):
#   img = cv2.imread(path)
#   if img is None:
#     return None

#   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#   img = cv2.resize(img, IMG_SIZE)
#   img = img.astype("float32") / 255.0
#   img = segment_fruit(img)
#   if augment:
#         img = augment_image(img)
#   # img = apply_clahe(img)
#   return img

# # ======================
# # EXTRAÇÃO DE FEATURES
# # ======================
# def extract_features(img):
#     # --- Histograma HSV ---
#     hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
#     h, s, v = cv2.split(hsv)
#     hist_h = cv2.calcHist([h], [0], None, [32], [0, 180])
#     hist_s = cv2.calcHist([s], [0], None, [32], [0, 256])
#     hist_v = cv2.calcHist([v], [0], None, [32], [0, 256])
#     hist_hsv = np.concatenate([hist_h, hist_s, hist_v]).flatten()
#     hist_hsv = hist_hsv / (hist_hsv.sum() + 1e-6)

#     # --- LBP de textura ---
#     gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
#     lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
#     n_bins = int(lbp.max() + 1)
#     lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
#     lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)

#     feature_vec = np.concatenate([hist_hsv, lbp_hist])
#     return feature_vec

# def mostrar_exemplos_imagens(base_path, n=5):
#     for classe, label in [("Fresh", "fresh"), ("Rotten", "rotten")]:
#         folder = os.path.join(base_path, classe.lower())
#         files = [os.path.join(folder, f) for f in os.listdir(folder)
#                  if f.lower().endswith((".jpg", ".png", ".jpeg"))][:n]

#         plt.figure(figsize=(15, 5))
#         plt.suptitle(f"{classe} - {n} exemplos", fontsize=16)

#         for i, fpath in enumerate(files):
#             img_original = cv2.imread(fpath)
#             img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
#             img_original = cv2.resize(img_original, IMG_SIZE)

#             img_segmented = segment_fruit(img_original.astype("float32") / 255.0)
#             img_clahe    = apply_clahe(img_segmented)

#             # Mostra original
#             plt.subplot(3, n, i+1)
#             plt.imshow(img_original)
#             plt.axis("off")
#             if i == 0: plt.ylabel("Original", fontsize=12)

#             # Mostra segmentada
#             plt.subplot(3, n, i+1+n)
#             plt.imshow(img_segmented)
#             plt.axis("off")
#             if i == 0: plt.ylabel("Segmentada", fontsize=12)

#             # Mostra CLAHE
#             # plt.subplot(3, n, i+1+2*n)
#             # plt.imshow(img_clahe)
#             # plt.axis("off")
#             # if i == 0: plt.ylabel("CLAHE", fontsize=12)

#         plt.show()

# # ======================
# # CAMINHOS DOS DATASETS
# # ======================
# base_train = "/content/dataset/train"
# base_val   = "/content/dataset/val"
# base_test  = "/content/dataset/test"

# mostrar_exemplos_imagens(base_train, n=5)

# # ======================
# # CARREGAMENTO MULTITHREADED
# # ======================
# def load_folder_features(folder, label, augment=False):
#   files = [os.path.join(folder, f) for f in os.listdir(folder)
#   if f.lower().endswith((".jpg", ".png", ".jpeg"))]
#   X, y = [], []

#   def process_file(fpath):
#       img = preprocess_image(fpath, augment=augment)
#       if img is None:
#           return None
#       feat = extract_features(img)
#       return feat

#   with ThreadPoolExecutor(max_workers=8) as executor:
#       futures = {executor.submit(process_file, f): f for f in files}
#       for future in as_completed(futures):
#           feat = future.result()
#           if feat is not None:
#               X.append(feat)
#               y.append(label)

#   return np.array(X), np.array(y)

# def load_dataset(base_path, augment=False):
#   Xf, yf = load_folder_features(os.path.join(base_path, "fresh"), 0, augment=augment)
#   Xr, yr = load_folder_features(os.path.join(base_path, "rotten"), 1, augment=augment)
#   X = np.concatenate([Xf, Xr])
#   y = np.concatenate([yf, yr])
#   return X, y

# # ======================
# # CAMINHOS DOS DATASETS
# # ======================
# base_train = "/content/dataset/train"
# base_val   = "/content/dataset/val"
# base_test  = "/content/dataset/test"

# print("⚡ Carregando dataset com extração de features...")
# X_train, y_train = load_dataset(base_train, augment=True)
# X_val,   y_val   = load_dataset(base_val, augment=False)
# X_test,  y_test  = load_dataset(base_test, augment=False)

# print("✅ Features carregadas:")
# print("Train:", X_train.shape)
# print("Val:  ", X_val.shape)
# print("Test: ", X_test.shape)

# # ======================
# # TREINAMENTO PCA + SVM
# # ======================
# pipeline = Pipeline([
# ("pca", PCA(n_components=80, whiten=True, random_state=42)),
# ("svm", LinearSVC(C=1.0, max_iter=5000))
# ])

# print("\n⏳ Treinando modelo PCA + SVM...")
# pipeline.fit(X_train, y_train)
# print("✅ Treinamento concluído.")

# # ======================
# # AVALIAÇÃO: TREINO, VALIDAÇÃO E TESTE
# # ======================
# def avaliar_modelo(pipeline, X, y, fase="Treino"):
#     y_pred = pipeline.predict(X)
#     print(f"\n===== Classification Report ({fase}) =====")
#     print(classification_report(y, y_pred, target_names=["Fresh", "Rotten"]))

#     cm = confusion_matrix(y, y_pred)
#     plt.figure(figsize=(5,4))
#     ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#     ax.set_xlabel("Predito")
#     ax.set_ylabel("Real")
#     ax.set_xticklabels(["Fresh", "Rotten"])
#     ax.set_yticklabels(["Fresh", "Rotten"])
#     plt.title(f"Matriz de Confusão ({fase})")
#     plt.show()

# # Avaliar no treino
# avaliar_modelo(pipeline, X_train, y_train, fase="Treinamento")
# # Avaliar na validação
# avaliar_modelo(pipeline, X_val, y_val, fase="Validação")
# # Avaliar no teste
# avaliar_modelo(pipeline, X_test, y_test, fase="Teste")
# # Predição
# # y_pred = pipeline.predict(X_test) --> mudando para treshold customizado
# threshold = -0.34 # --> treshold customizado | -0.4431 fez com que FP aumentasse muito, então diminuímos para -0.34, equilibrando FP e FN

# # ======================
# # CURVA ROC
# # ======================
# y_score = pipeline.decision_function(X_test)
# y_pred_thresh = (y_score >= threshold).astype(int)

# print("\n=== Classification Report (Threshold Customizado) ===")
# print(classification_report(y_test, y_pred_thresh, target_names=["Fresh", "Rotten"]))

# cm = confusion_matrix(y_test, y_pred_thresh)
# plt.figure(figsize=(5,4))
# ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
# ax.set_xlabel("Predito")
# ax.set_ylabel("Real")
# ax.set_xticklabels(["Fresh", "Rotten"])
# ax.set_yticklabels(["Fresh", "Rotten"])
# plt.title(f"Matriz de Confusão (Threshold={threshold})")
# plt.show()

# fpr, tpr, thresholds = roc_curve(y_test, y_score)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(6,5))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.legend(loc="lower right")
# plt.show()








# # # ==========================================================
# # # VARREDURA DE TODOS OS THRESHOLDS
# # # ==========================================================

# # thresholds_test = np.linspace(y_score.min(), y_score.max(), 200)

# # best_recall = 0
# # best_t_recall = None

# # best_precision = 0
# # best_t_recall90 = None
# # desired_recall = 0.90

# # for t in thresholds_test:

# #     y_pred_t = (y_score >= t).astype(int)

# #     r = recall_score(y_test, y_pred_t, pos_label=1)
# #     p = precision_score(y_test, y_pred_t, pos_label=1)

# #     # ---- A) Melhor recall absoluto (threshold paranoico) ----
# #     if r > best_recall:
# #         best_recall = r
# #         best_t_recall = t

# #     # ---- B) Melhor precision garantindo recall >= 0.90 ----
# #     if r >= desired_recall and p > best_precision:
# #         best_precision = p
# #         best_t_recall90 = t

# # # ==========================================================
# # # RESULTADOS
# # # ==========================================================

# # print("=== Threshold que maximiza RECALL (Rotten) ===")
# # print("Recall máximo:", best_recall)
# # print("Threshold:", best_t_recall)

# # print("\n=== Threshold com Recall >= 0.90 e MELHOR Precision ===")
# # print("Recall alvo:", desired_recall)
# # print("Threshold:", best_t_recall90)
# # print("Precision obtida:", best_precision)

# # # ==========================================================
# # # REPORTS
# # # ==========================================================

# # # ---- A) Aplicar threshold paranoico (máx recall) ----
# # y_pred_max_recall = (y_score >= best_t_recall).astype(int)

# # print("\n=== Classification Report (Max Recall) ===")
# # print(classification_report(y_test, y_pred_max_recall, target_names=["Fresh", "Rotten"]))


# # # ---- B) Aplicar threshold com recall≥0.90 e melhor precision ----
# # if best_t_recall90 is not None:
# #     y_pred_recall90 = (y_score >= best_t_recall90).astype(int)

# #     print("\n=== Classification Report (Recall>=0.90 + Best Precision) ===")
# #     print(classification_report(y_test, y_pred_recall90, target_names=["Fresh", "Rotten"]))
# # else:
# #     print("\n⚠ Nenhum threshold atingiu recall ≥ 0.90. Tente diminuir para 0.85 ou 0.80.")
# # MELHOR TRESHOLD ENCONTRADO = -0.44315309233917866






# # ===========================================================================================
###############################################################################################
# SEPARAR EM UMA CÉLULA A PARTE!!!!!!!
###############################################################################################
# =============================================================================================

# import urllib.request
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # ======================
# # FUNÇÃO PARA BAIXAR IMAGEM
# # ======================

# def baixar_imagem_url(url, save_path="imagem_externa.jpg"):
#   print("📥 Baixando imagem externa...")
#   urllib.request.urlretrieve(url, save_path)
#   return save_path

# # ======================
# # FUNÇÃO PARA EXTRAÇÃO DE FEATURES (HSV + LBP)
# # ======================

# def extract_features_for_url(img):
#   # --- Histograma HSV ---
#   hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
#   h, s, v = cv2.split(hsv)
#   hist_h = cv2.calcHist([h], [0], None, [32], [0, 180])
#   hist_s = cv2.calcHist([s], [0], None, [32], [0, 256])
#   hist_v = cv2.calcHist([v], [0], None, [32], [0, 256])
#   hist_hsv = np.concatenate([hist_h, hist_s, hist_v]).flatten()
#   hist_hsv = hist_hsv / (hist_hsv.sum() + 1e-6)

#   # --- LBP de textura ---
#   gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
#   lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
#   n_bins = int(lbp.max() + 1)
#   lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
#   lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)

#   feature_vec = np.concatenate([hist_hsv, lbp_hist])
#   return feature_vec.reshape(1, -1)

# # ======================
# # FUNÇÃO PARA TESTAR IMAGEM EXTERNA
# # ======================

# def testar_imagem_url(url):
#     caminho = baixar_imagem_url(url, save_path="imagem_externa.jpg")

#     img = cv2.imread(caminho)
#     if img is None:
#         print("❌ Erro: imagem não pôde ser carregada.")
#         return

#     # Converte para RGB
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_rgb = cv2.resize(img_rgb, IMG_SIZE)
#     img_rgb = img_rgb.astype("float32") / 255.0

#     # --- Aplicar mesmas normalizações do pipeline ---
#     img_rgb = segment_fruit(img_rgb)  # segmenta a fruta
#     img_rgb = apply_clahe(img_rgb)    # equalização local (CLAHE)

#     # --- Extrai features compatíveis com o pipeline ---
#     X_ext = extract_features_for_url(img_rgb)

#     # Predição
#     pred_label = pipeline.predict(X_ext)[0]
#     score = pipeline.decision_function(X_ext)[0] if hasattr(pipeline, "decision_function") else None
#     classe_str = "🍏 Fresh (0)" if pred_label == 0 else "🍎 Rotten (1)"

#     print("\n🔍 RESULTADO DA CLASSIFICAÇÃO:")
#     print("--------------------------------")
#     print(f"🎯 Classe prevista: {classe_str}")
#     if score is not None:
#         print(f"📊 Score da SVM (decision_function): {score:.4f} (positivo = Rotten)")

#     # Mostrar imagem
#     plt.figure(figsize=(4,4))
#     plt.imshow(img_rgb)
#     plt.title(f"Predição: {classe_str}")
#     plt.axis("off")
#     plt.show()

# # ======================
# # EXEMPLO DE USO
# # ======================

# testar_imagem_url("https://thumbs.dreamstime.com/z/morango-com-apodrecimento-fruta-podre-doen%C3%A7as-f%C3%BAngicas-dos-frutos-e-outros-produtos-isolado-sobre-fundo-branco-237464921.jpg")
# testar_imagem_url("https://storage.googleapis.com/kagglesdsdata/datasets/6076655/9893737/Augmented%20Image/FreshApple/FreshOrange%20%28105%29.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20251203%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251203T232224Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=4ffe85091b3918a31ae71a326cacccd383e9488f26be7a6cab40514051beb6db27a5840fb54f8e8b0657db0a599c17d2c4c450f516bd47eb7363843b4560daa57f0f79c925819dfce692c61a58942ad68db8773116a5c8f9defd0e650f6ab069c4e443f1b4f80cad0e4ed1f3ab1becd9ecdcb168a46d13b3519e72b5778c49ffeee6b016e6322b88c992cc6ae4515afae06341677b363a1025d00e9e17a94a9035122ba6b387f6d158132ad2ca83586ddee35e7aac6210e178dfd59036cc534972fcb164896a58c963b50ffb180a322ade6f537901c4429d54227e5515f514687888cc17d9aeb582a1031227135cab01f47374d8f3e680f0e1e0eb65bee8c9d9")
# testar_imagem_url("https://storage.googleapis.com/kagglesdsdata/datasets/6076655/9893737/Augmented%20Image/FreshStrawberry/FreshStrawberry%20%28105%29.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20251206%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251206T200330Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=64185d869fc434facbfa5c1acad9b66b7c597123dfd13970c6589d5a8175cec865a3659ab8375198d0c2b2c22766812e3b9137eaeb6e3b26ce80568763d64c5862054ffd4bbf44a57424a71d30c74d606bb81dc0339304b4e1807ff610ee95a5b760c69e76528be404eafc39864c272ac0186ba356edbd5b3291011461154df56c706c53cec50cc279d0b9f84b21844c4c7d64c6edced656eb18bfe134351f605217a23afdc4d52a9e56142d6d582f2f1c98aa359bdfe1147307ef1521385d3170f4c78c8cc75d081a3412d8234fa9b0654c546f6f9f9c5ab45e979ec36bfb21b31b8473878aa7e20f730bccd08e33669b6c9fc1d0c3eab9461b3889ce75fb38")
# testar_imagem_url("https://storage.googleapis.com/kagglesdsdata/datasets/6076655/9893737/Augmented%20Image/RottenStrawberry/RottenStrawberry%20%28110%29.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20251206%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251206T200349Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=595b7e661014c3e86e60bb9ba35ae0f54d5a1daed409acd5e2809ade71637eb1f8bfd242a6dee3dc2dd5204c3b55b5dfe5748a79f6069ded15a61c7a8a368e5f2d49658eabb1d09074588d67ef9106a520ed15f38b8248976e75b58ab83481712d9cf0e25c5b7a0560be7da71c53ea1540259ec78bc83df6c116d54f651ad2d56016091c6df37e14ded4b8f72f4fe83a7e287c333557f1c993fbd525de75de6dbdfad461500b4847aadae5a319189b829ad0f8024a934246afa2abf90589d93894641e5d889039ed957c79e540ee702a7bcd5a297c98a918c87cdb8410c230d9390cedb52edee3aaad6e798055e7433bcfb84f3d23ce3344085ba8c10dd5c7d7")
# testar_imagem_url("https://m.media-amazon.com/images/I/71XdMMf9AaL._SL1500_.jpg")
# testar_imagem_url("https://thumbs.dreamstime.com/z/maracuj%C3%A1-sobreamadurecido-maduro-sobre-fundo-branco-208858448.jpg")#
# testar_imagem_url("https://media.istockphoto.com/id/147058486/pt/foto/uma-ma%C3%A7%C3%A3-estragada.jpg?s=612x612&w=0&k=20&c=jw9_8CiyM6lbEGqqxqDZXDO1q_F1IvnOQDPFotu6nMw=")
# testar_imagem_url("https://thumbs.dreamstime.com/b/ma%C3%A7%C3%A3-podre-ilustra-uma-alimenta%C3%A7%C3%A3o-pouco-saud%C3%A1vel-vis%C3%A3o-mais-pr%C3%B3xima-do-decaimento-e-da-comida-n%C3%A3o-geradora-de-ai-301349633.jpg")
# testar_imagem_url("https://p.turbosquid.com/ts-thumb/NP/Cw8r0m/89kEYlSZ/02/jpg/1422727639/600x600/fit_q87/71c88f46793bc3ffb8d26b64357cf8aaca3749e3/02.jpg")
# testar_imagem_url("https://images.freeimages.com/images/premium/previews/9308/9308927-rotting-banana.jpg")
# testar_imagem_url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRThvypl90QMXvJYvsobeAl2dchu-NBzs0n8A&s")
# testar_imagem_url("https://thumbs.dreamstime.com/b/morango-podre-com-grande-molde-imagem-disparada-macro-do-close-up-da-o-isolado-no-fundo-branco-123389527.jpg")
# testar_imagem_url("https://mondiniplantas.cdn.magazord.com.br/img/2025/04/produto/6534/pera-d-agua.jpg?ims=800x800")
# testar_imagem_url("https://thumbs.dreamstime.com/b/uma-pera-podre-isolada-sobre-fundo-branco-207418869.jpg")
# testar_imagem_url("https://previews.123rf.com/images/freerlaw/freerlaw1708/freerlaw170800155/84405333-half-of-a-cut-out-rotten-pear-on-white-background.jpg")
# testar_imagem_url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSP7NndjZmKlNrl5uIdUs5uwpE_9BjQ66YVkQ&s")