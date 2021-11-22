import cv2

def add_Lights(img, colors):

    cur_light = cv2.imread('./utils/images/light_' + colors + '.png', -1)


    ## Image Resize
    rp = cv2.resize(cur_light, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)


    ## Image Addtion with Alpha
    x_offset = y_offset = 10
    for c in range(0,3):
        img[y_offset:y_offset+rp.shape[0], x_offset:x_offset+rp.shape[1], c] = rp[:,:,c] * (rp[:,:,3]/255.0) + img[y_offset:y_offset+rp.shape[0], x_offset:x_offset+rp.shape[1], c] * (1.0 - rp[:,:,3]/255.0)
    
    return img