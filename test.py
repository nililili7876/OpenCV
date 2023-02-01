import cv2
img = cv2.imread('./111.jpg')
print(type(img))
cv2.imshow('test',img)
cv2.waitKey(0)