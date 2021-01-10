import cv2
import numpy as np
import matplotlib.pyplot as plt

# 処理前と処理後の比較のPlot
def bef_aft_img_show(img1,img2):
    print(f'average brightness:{img1.mean():.1f}, {img2.mean():.1f} ')
    plt.figure(figsize=(10,10))
    plt.subplot(121),plt.imshow(img1),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img2),plt.title('After')
    plt.xticks([]), plt.yticks([])
    plt.show()

# 処理前と処理後の比較のPlot
def all_img_show(img1,img2,img3):
    print(f'average brightness:{img1.mean():.1f}, {img2.mean():.1f}, {img3.mean():.1f}')
    plt.figure(figsize=(12,12))
    plt.subplot(131),plt.imshow(img1),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(img2),plt.title('Flat')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img3),plt.title('After')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
# ImageとHistgramのPlot
def image_hist_show(img):
    print(f'Shape:{img.shape},type:{img.dtype}')
    print(f'Average Brightness:{img.mean():.1f}')
    hist = cv2.calcHist([img],[0],None,[256],[0,256])

    plt.figure(figsize=(10,5))
    plt.subplot(121),plt.imshow(img),plt.title('Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.plot(hist),plt.title('Histgram')
    plt.show()
    
class FlatField():
    
    def __init__(self, nd_img):
        self.nd_img = nd_img
        
    def apply(self, mean_size=(50,50),method ='ffc'):
        """
        Args:
            mean_size (tuple, optional): region. Defaults to (50,50).
            method (str, optional):ffc = LF/FF, ffcl = LF-FF+ave. Defaults to 'ffc'.

        Returns:
            ndarry: calculated image
        """
        
        self.mean_img = cv2.blur(self.nd_img,mean_size) 
        
        
        avg_hist = self.nd_img.mean()
        if method == 'ffc':
            ffc = (self.nd_img / self.mean_img) * avg_hist
        elif method =='ffcl' :
            ffc = self.nd_img - self.mean_img + avg_hist
        else:
            pass
        
        self.ffc_i8 = ffc.astype('uint8')
        
        return self.ffc_i8
    
    def plot_img(self):  
        all_img_show(self.nd_img,self.mean_img,self.ffc_i8)
        
if __name__ == '__main__':
    file_path = './data/samp_Gradation1.bmp' 
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    ffc_test = FlatField(img)
    ffc_img = ffc_test.apply()
    ffc_test.plot_img()