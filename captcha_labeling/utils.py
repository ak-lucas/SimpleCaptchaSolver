from IPython.display import clear_output
import matplotlib.pyplot as plt
from skimage import io
import requests
import shutil
import time
import os


class CaptchaLabeling:
    def __init__(self):
        pass
    
    def start(self):
        img = Image()
        label = 'default'
        counter = 0
        while True:
            clear_output(wait=True)
            response = img.downloadImage()

            img.saveTmpImage(response)

            img.showImg()
            
            label = input('Quais são os 7 caracteres apresentados na imagem?')
            
            
            if label.upper() == 'P' or label == 'proxima' or label =='pular':
                print('pulando para próxima imagem')
                continue
                
            elif len(label) != 7:
                print("Você rotulou {} imagens! Obrigado!".format(counter))
                break
                   
            img.saveImageWithLabel(label.upper())
                
            print("Obrigado! Imagem salva.")
            counter += 1
            time.sleep(1)
            
        os.remove(r'tmp.jpeg')
            

class Image:
    def __init__(self):
        pass
    
    def downloadImage(self):
        s = requests.Session()

        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.129 Safari/537.36'}

        response = s.get('http://web.trf3.jus.br/consultas/Captcha/GerarCaptcha', headers=headers, stream=True)
        response.raw.decode_content = True
        
        return response.raw
        
    def saveTmpImage(self, binImg):
        with open('tmp.jpeg', 'wb') as f:
            shutil.copyfileobj(binImg, f)
        
    def showImg(self):
        plt.subplot(1, 1, 1)
        plt.axis('off')
        io.imshow('tmp.jpeg')
        plt.show()
    
    def saveImageWithLabel(self, label):
        if not os.path.isdir('labeled_captchas'):
            os.mkdir('labeled_captchas')
            
        os.rename(r'tmp.jpeg', r'labeled_captchas/{}.jpeg'.format(label))
        