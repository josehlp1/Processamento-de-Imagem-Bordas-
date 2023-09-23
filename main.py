import cv2
import numpy as np
from sklearn.cluster import KMeans
import pywt


def exibir_imagem(titulo, img):
    altura, largura = img.shape[:2]
    max_dimensao = 400

    # Redimensiona a imagem caso ela seja maior que a dimensão máxima permitida
    if altura > largura:
        if altura > max_dimensao:
            nova_altura = max_dimensao
            nova_largura = int((nova_altura / altura) * largura)
            img = cv2.resize(img, (nova_largura, nova_altura))
    else:
        if largura > max_dimensao:
            nova_largura = max_dimensao
            nova_altura = int((nova_largura / largura) * altura)
            img = cv2.resize(img, (nova_largura, nova_altura))

    # Exibe a imagem
    cv2.imshow(titulo, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Carrega a imagem
imagem = cv2.imread('tomato.jpg')

# Verifica se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao carregar tomato.jpg")
    exit()

# Exibe a imagem original
exibir_imagem('Imagem Original', imagem)

# Redimensiona a imagem
imagem = cv2.resize(imagem, (3456, 4608))
imagem_pequena = cv2.resize(imagem, (1728, 2304))

# Exibe a imagem redimensionada
exibir_imagem('Imagem Redimensionada', imagem_pequena)

# Utiliza o K-means para detectar a ROI (Região de Interesse)
pixels = imagem_pequena.reshape((-1, 3))

# Ajusta o valor de n_init para evitar avisos
kmeans = KMeans(n_clusters=2, n_init=10).fit(pixels)

segmentado = kmeans.labels_.reshape(imagem_pequena.shape[:2])
mascara_tomate = np.where(segmentado == np.argmin(kmeans.cluster_centers_.sum(axis=1)), 255, 0).astype(np.uint8)
exibir_imagem('Mascara do Tomate', mascara_tomate)

# Realiza a decomposição wavelet da imagem
coeficientes = pywt.dwt2(imagem_pequena, 'db1')
cA, (cH, cV, cD) = coeficientes

# Reconstrói a imagem a partir dos coeficientes da decomposição wavelet
imagem_equalizada = pywt.idwt2((cA, (cH, cV, cD)), 'db1').astype(np.uint8)
exibir_imagem('Imagem Reconstruida com Wavelet', imagem_equalizada)

# Detecta as bordas da imagem utilizando o algoritmo Canny
bordas = cv2.Canny(imagem_equalizada, 100, 200)
exibir_imagem('Bordas', bordas)

# Preenche os buracos nas bordas detectadas
contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contorno in contornos:
    cv2.drawContours(bordas, [contorno], 0, 255, -1)
imagem_binaria = bordas
exibir_imagem('Bordas Preenchidas', imagem_binaria)

# Remove ruídos da imagem binária
kernel = np.ones((5, 5), np.uint8)
imagem_binaria = cv2.morphologyEx(imagem_binaria, cv2.MORPH_OPEN, kernel)
exibir_imagem('Imagem sem Ruido', imagem_binaria)

# Realiza a dilatação da imagem binária
imagem_binaria = cv2.dilate(imagem_binaria, kernel, iterations=2)
exibir_imagem('Imagem Dilatada', imagem_binaria)
