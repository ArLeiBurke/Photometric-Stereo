from photostereo import photometry
import cv2 as cv
import sys


IMAGES = 4

root_fold = "./samples/0107Test/"
# root_fold = "./samples/压伤1/"
# root_fold = "C:/Users/l/Desktop/OpenCV Halcon  C++  PCL/photometric stereo/Test/"

obj_name = "S0001_C01_P01_L."

# 图片格式;格式不对的话 会导致程序跑不起来;
format = ".bmp"
# format = ".png"

light_manual = True

#Load input image array
image_array = []
for id in range(0, IMAGES):
    try:
        filename = root_fold + str(obj_name) + str(id) + format
        # im = cv.imread(root_fold + str(obj_name) + str(id) + format, cv.IMREAD_GRAYSCALE)
        im = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        image_array.append(im)
    except cv.error as err:
        print(err)

myps = photometry(IMAGES, True,root_fold)

if light_manual:


    slants = [60.0, 60.0, 60.0, 60.0]
    tilts = [0, -90, 180, 90]

    myps.setlmfromts(tilts, slants)
    print(myps.settsfromlm())

else:
    # LOADING LIGHTS FROM FILE
    fs = cv.FileStorage(root_fold + "LightMatrix.yml", cv.FILE_STORAGE_READ)
    fn = fs.getNode("Lights")
    light_mat = fn.mat()
    myps.setlightmat(light_mat)
    print(myps.settsfromlm())

normal_map = myps.runphotometry(image_array)

albedo = myps.getalbedo()

albedo = cv.normalize(albedo, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)

image_rgb = cv.cvtColor(normal_map, cv.COLOR_BGR2RGB)

# 一阶导数（梯度）计算 - Sobel
sobel_x = cv.Sobel(image_rgb, cv.CV_64F, 1, 0, ksize=3)  # X方向一阶导数
sobel_y = cv.Sobel(image_rgb, cv.CV_64F, 0, 1, ksize=3)  # Y方向一阶导数

# 计算梯度幅值和方向
gradient_magnitude = cv.magnitude(sobel_x, sobel_y)  # 梯度幅值
gradient_direction = cv.phase(sobel_x, sobel_y, angleInDegrees=True)  # 梯度方向


laplacian = cv.Laplacian(image_rgb, cv.CV_64F)  # 对RGB图像计算Laplacian

#
# cv.imshow("laplacian",laplacian)
# cv.imwrite("laplacian.bmp",laplacian)
#
# edge = np.uint8(np.absolute(gradient_magnitude))
# cv.imshow("edge",edge)
# cv.imwrite("edge.bmp",edge)
#
# cv.imshow("normal_map",normal_map)
# cv.imwrite("normal_map.bmp",normal_map)
#


# cv.imshow("soble_x",sobel_x)
# cv.imshow("sobel_y",sobel_y)
#
# cv.imshow("gradient_magnitude",gradient_magnitude)
# cv.imshow("gradient_direction",gradient_direction)




# cv.imwrite("albedo_after.bmp",albedo)

#gauss = myps.computegaussian()
#med = myps.computemedian()

#cv.imwrite('normal_map.png',normal_map)
#cv.imwrite('albedo.png',albedo)
#cv.imwrite('gauss.png',gauss)
#cv.imwrite('med.png',med)

toc = time.process_time()
print("Process duration: " + str(toc - tic))

# TEST: 3d reconstruction
# myps.computedepthmap()
# myps.computedepth2()
# myps.display3dobj()
# cv.imshow("normal", normal_map)
#cv.imshow("mean", med)
#cv.imshow("gauss", gauss)
cv.waitKey(0)
cv.destroyAllWindows()