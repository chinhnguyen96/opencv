import cv2
import numpy as np
#=================== THỰC HIỆN THUẬT TOÁN CANNY =======================#
def RGB(img):
    weak = np.min(img)
    strong = np.max(img)
    img_1 = (img - weak) / (strong - weak) 
    img_1 *= 255
    return img_1

def thuat_toan_canny(img, weak, strong, size_sobel=3, gradient_2=False):

    # Làm nhiễu, mờ bằng bộ lọc Gaussian
    img_2 = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1, sigmaY=1)
    
    # Tìm độ dốc cường độ của hình ảnh
    x = cv2.Sobel(img_2, cv2.CV_64F, 1, 0, ksize=size_sobel)
    y = cv2.Sobel(img_2, cv2.CV_64F, 0, 1, ksize=size_sobel)
        
    if gradient_2:
        edge_gdt = np.sqrt(x*x + y*y)
    else:
        edge_gdt = np.abs(x) + np.abs(y)
        
    angle = np.arctan2(y, x) * 180 / np.pi
    
    # Làm tròn góc theo 4 hướng
    angle = np.abs(angle)
    angle[angle <= 22.5] = 0
    angle[angle >= 157.5] = 0
    angle[(angle > 22.5) * (angle < 67.5)] = 45
    angle[(angle >= 67.5) * (angle <= 112.5)] = 90
    angle[(angle > 112.5) * (angle <= 157.5)] = 135
    
    # Non-maximum Suppression
    keep_mask = np.zeros(img_2.shape, np.uint8)
    for y in range(1, edge_gdt.shape[0]-1):
        for x in range(1, edge_gdt.shape[1]-1):
            agi = edge_gdt[y-1:y+2, x-1:x+2] # area gradient intensity
            aa = angle[y-1:y+2, x-1:x+2] # area angel
            ca = aa[1,1] # current angle
            cgi = agi[1,1] # current gradient intensity
            
            if ca == 0: 
                if cgi > max(agi[1,0], agi[1,2]):
                    keep_mask[y,x] = 255
                else:
                    edge_gdt[y,x] = 0
            elif ca == 45:
                if cgi > max(agi[2,0], agi[0,2]):
                    keep_mask[y,x] = 255
                else:
                    edge_gdt[y,x] = 0
            elif ca == 90:
                if cgi > max(agi[0,1], agi[2,1]):
                    keep_mask[y,x] = 255
                else:
                    edge_gdt[y,x] = 0
            elif ca == 135:
                if cgi > max(agi[0,0], agi[2,2]):
                    keep_mask[y,x] = 255
                else:
                    edge_gdt[y,x] = 0
    
    # Hysteresis Thresholding    
    canny_mask = np.zeros(img_2.shape, np.uint8)
    canny_mask[(keep_mask>0) * (edge_gdt>weak)] = 255
    
    return RGB(canny_mask)
# load ảnh
img = cv2.imread('C:/Users/image.jpg', 0)
# gọi hàm đã xây dựng ở đầu bài
my_canny = thuat_toan_canny(img, weak=50, strong=200)
### Ở BƯỚC NÀY TA SẼ KIỂM TRA LẠI THUẬT TOÁN CANNY TA ĐÃ XÂY DỰNG
### BẰNG CÁCH CHẠY HÀM CANNY TRONG OPENCV ĐÊ XEM 2 ẢNH CÓ KHÁC BIỆT KHÔNG
canny = cv2.Canny(img, 50, 200)

cv2.imshow('my_canny.jpg', my_canny)
cv2.imshow('canny.jpg', canny)
cv2.waitKey(0)

### KẾT QUẢ 2 HÌNH KHÁ GIỐNG NHAU - XEM FILE HÌNH ketqua_canny.png
### THUẬT TOÁN XÂY DỰNG ĐÚNG
    
    