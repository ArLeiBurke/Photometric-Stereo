# Photometric Stereo

Photometric Stereo 是将多个光源从不同方向采集到的图片通过一定的算法融合成Normal Map等;

# 碎碎念
其实Photometric Stereo 这个算法还是有一定局限性的;我感觉只适用于产品表面有凹坑或者凸起的缺陷。<br>

Photometric Stereo 还比较吃打光，如果打光没有把产品表面的缺陷打出来的话，那合成后的图片大概率看不到缺陷! <br>

这个项目里面的所有图片都是我用一个 四分区环光 来采集的。实际应用场景下,单个四分区环光可能有点吃力，真正用到的可能不止四个光源，如果想要达到好的成像效果的话，可能需要八个或者更多光源来对产品进行打光。从不同的角度和高度来进行打光<br>

这个算法一开始是比较耗时的，因为有非常多Numpy操作！后来我通过将Numpy相关的操作全都替换成了Tensor。算法的执行速度快了很多，我拿公司的电脑测试了一下(i7-14700kf + RTX 4060D),算法耗时大概在0.3s左右<br>

如果想要把这个应用到实际项目里面的话，可以用C++重写一下！用libtorch优化一下！这个我觉得问题不大<br>


# 测试用图片

链接:https://pan.baidu.com/s/1k8Xg0Jf4gNq9dxFoxSH29w?pwd=7wf6 <br>
提取码:7wf6 复制这段内容后打开百度网盘手机App，操作更方便哦



# 效果
下面是四张原图！光源从产品的哪个方向开始拍照搭眼一看就知道！！！<br>

![image](pic/S0001_C01_P01_L.0.bmp) <br>
![image](pic/S0001_C01_P01_L.1.bmp) <br>
![image](pic/S0001_C01_P01_L.2.bmp) <br>
![image](pic/S0001_C01_P01_L.3.bmp) <br>

*下面这几张图片是融合上面四张原图而得到的！！！分别是 albedo , Normal Map , P Grade ,Q Grade*<br>

![image](pic/albedo.bmp) <br>
![image](pic/normal_normalized_GPU.bmp) <br>
![image](pic/pgrads.bmp) <br>
![image](pic/qgrads.bmp) <br>





# Reference <br>
[Introduction to Photometric Stereo: 1 – The Basics](http://www.ian-hales.com/2019/06/26/introduction-to-photometric-stereo-1-the-basics/) <br>
[Photometric Stereo and Process](https://documentation.euresys.com/Products/OPEN_EVISION/OPEN_EVISION/en-us/Content/03_Using/6_3D_Processing/1_Easy3D/4_Photometric_Stereo/Photometric_Stereo_and_Process.htm?TocPath=Photometric%20Stereo%20and%20Process%7C_____0) <br>


