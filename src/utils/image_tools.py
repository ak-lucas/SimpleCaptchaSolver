import imutils.contours
import scipy.ndimage
import numpy as np
import imutils
import cv2
import os


class PreprocessPipeline:
    """
        Recebe o diretório e o nome da imagem e retorna uma lista com os caracteres extraídos da imagem
    """
    def __init__(self):
        pass

    def imgTransform(self, dir, imgName, version='v4'):
        imtools = ImageTools()

        # carrega a imagem na memória
        img = imtools.load(dir, imgName)
        # binariza e remove ruídos
        img = imtools.preprocess(img, version)
        # armazena os caracteres em uma lista de arrays numpy
        charsArray = imtools.crop(img)

        # se não houverem 7 caracteres houve algum problema no pré-processamento
        if len(charsArray) != 7:
            print('Os caracteres não puderam ser separados corretamente.')
            raise Exception('CropError')

        return charsArray


class ImageTools:
    """
        Implementa métodos úteis para a manipulação dos captchas
    """
    def __init__(self):
        pass

    def load(self, dir, imgName):
        os.chdir(dir)

        # lê imagem do disco e converte para escala de cinza
        img = cv2.imread(imgName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img is None:
            raise Exception('LoadError')

        return img

    def preprocess(self, img, version='v4'):
        if version == 'v1':
            # usa um filtro para diminuir o ruído
            img = scipy.ndimage.median_filter(img, (2, 2))

            # usa um filtro para diminuir o ruído tentando preservar as bordas das letras
            img = cv2.bilateralFilter(img, 11, 17, 17)

            # binariza a imagem e inverte as cores
            img = ~cv2.threshold(img, 215, 255, cv2.THRESH_BINARY)[1]

            # filtra os ruídos novamente
            img = scipy.ndimage.median_filter(img, (2, 2))

            # aplica operação para melhorar a definição
            kernel = np.ones((2, 2), np.uint8)
            img = cv2.erode(img, kernel, iterations=1)

        elif version == 'v2':
            # usa um filtro para diminuir o ruído tentando preservar as bordas das letras
            img = cv2.bilateralFilter(img, 11, 17, 17)

            # binariza a imagem e inverte as cores
            img = ~cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)[1]

            # aplica operação de erosão para remover os ruídos e afinar as letras
            kernel = np.ones((2, 2), np.uint8)
            img = cv2.erode(img, kernel, iterations=1)

            # aplica operação de fechamento para tentar tampar algumas falhas
            kernel = np.ones((3, 2), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        elif version == 'v3':
            # binariza a imagem e inverte as cores
            img = ~cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)[1]

            # aplica operação de erosão para remover os ruídos e afinar as letras
            kernel = np.ones((2, 2), np.uint8)
            img = cv2.erode(img, kernel, iterations=1)

            # aplica operação de fechamento para tentar tampar algumas falhas
            kernel = np.ones((3, 2), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        elif version == 'v4':
            # binariza a imagem e inverte as cores
            img = ~cv2.threshold(img, 235, 255, cv2.THRESH_BINARY)[1]

            # aplica operação de erosão para remover os ruídos e afinar as letras
            kernel = np.ones((2, 2), np.uint8)
            img = cv2.erode(img, kernel, iterations=1)

        else:
            raise Exception("UnknownOption")

        return img

    def crop(self, img):
        # encontra os contornos e ordena para continuar na sequência correta
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts, _ = imutils.contours.sort_contours(cnts, method="left-to-right")

        # recorta os caracteres
        chars = []
        for c in cnts:
            area = cv2.contourArea(c)
            # garante que não vai pegar pequenos ruídos que possam ter restado
            if area > 10:
                x, y, w, h = cv2.boundingRect(c)
                cropped_char = 255 - img[y:y + h, x:x + w]
                chars.append(cv2.resize(~cropped_char, (28, 28)))

        return chars
