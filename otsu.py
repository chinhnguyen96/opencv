import numpy as np
from matplotlib import pyplot as plt
import argparse
import mahotas
import cv2
#=================== THỰC HIỆN THUẬT TOÁN OTSU =======================#
img = cv2.imread('C:/Users/image.jpg',0)
blur = cv2.GaussianBlur(img,(5,5),0)

# xây dựng lược đồ xám, và tìm hàm phân phối tích lũy của nó
hist = cv2.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()

bins = np.arange(256)

fn_min = np.inf
t = -1

for i in range(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    b1,b2 = np.hsplit(bins,[i]) # weights

    # tìm trung bình và phương sai
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

    # tính toán lớp phương sai và chọn lớp phương sai nhỏ nhất
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        t = i
print(t) # in kết quả ngưỡng t 

#============TÌM GIÁ TRỊ NGƯỠNG OTSU BẰNG HÀM TRONG OPENCV=========#
# find otsu's threshold value with OpenCV function
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret) #in kết quả ngưỡng t bằng hàm
######### Kết luận: ta thấy chạy ra 2 kết quả xấp xỉ bằng nhau là:
#### THUẬT TOÁN OTSU : 109
#### HÀM TRONG OPENCV: 108.0
#===================================================================
#### VÌ THUẬT TOÁN ĐÃ TÌM RA ĐƯỢC GIÁ TRỊ NGƯỠNG t=109
#### NÊN TA SẼ KIỂM TRA LẠI BẰNG CÁCH DÙNG THUẬT TOÁN OTSU ĐÃ XÂY DỰNG
#### SO VỚI DÙNG HÀM MAHOTAS TRONG OPENCV
#### XEM KẾT QUẢ 2 ẢNH CÓ GIỐNG NHAU KHÔNG

#### CÁCH 1: DÙNG HÀM MAHOTAS TRONG OPENCV 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = False, help = 'C:/Users/image.jpg.jpg')
args = vars(ap.parse_args())
image = cv2.imread('C:/Users/image.jpg.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
T = mahotas.thresholding.otsu(blurred) 
print("Otsu’s threshold: {}".format(T)) # T=108
thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0 
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Otsu", thresh)

#### CÁCH 2: DÙNG THUẬT TOÁN OTSU ĐÃ XÂY DỰNG ĐẦU BÀI

thresh1 = image.copy()
thresh1[thresh1 > t] = 255 #t=109 đã tìm trong thuật toán đầu bài
thresh1[thresh1 < 255] = 0 
thresh1 = cv2.bitwise_not(thresh1)
cv2.imshow("my_Otsu", thresh1)
cv2.waitKey(0)

##### KẾT QUẢ 2 HÌNH GIỐNG NHAU 99,9% - XEM FILE HÌNH ketqua_otsu.png #####
##### THUẬT TOÁN XÂY DỰNG ĐÚNG #####
    
    