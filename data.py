import cv2
image_path = "./ds9/A/b/depth_1_0001.png"
img = cv2.imread(image_path)  # For Reading The Image
cv2.imshow('image', img)      # For Showing The Image in a window with first parameter as it's title
cv2.waitKey(0)   #waits for a key to be pressed on a window 
cv2.destroyAllWindows() 