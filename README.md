# 更新日志

## 2024-03-16
---
黄家俊 
- 1.串口：原来的串口接收线程非常不稳定，接收数据帧增加了帧尾标识符，并改为暴力方法，从两帧长度的数据中找到合法的一帧，经过测试，频率高达两百赫兹，完全满足要求
- 2.识别：装甲板检测功能包替换为川大开源的四点模型，通过openvino推理加速，经测试每帧推理耗时在15-20ms之间。
- 3.相机：使用官方默认的qos文件，相机发布节点的队列为10，相机发布赫兹为120hz左右，远远大于推理的频率，会导致队列阻塞，造成每次估计出的状态信息都是100ms甚至更久之前的状态，修改相机发布队列和图像订阅队列为1，保证每次推理的实时性。
- 4.标定：采纳其他学校的意见，放弃matlab相机标定工具箱，使用ros标定工具，适配rv。
- 5.跟踪器：修改了r的范围，将r的估计范围抑制到0.2m到0.3m(大多数步兵的r都在此范围之间)
---
## 2024-03-21
---
黄家俊
- 1.PnP结算：添加角点均值滤波PnP结算，暂未测试效果
---
## 2024-04-13
---
黄家俊
- 1.模型：替换模型文件为沈航开源的最新网络，与之前的网络模型相比，增加了大小装甲板的分类,如示：
```c++
enum class ArmorColor
{
  BLUE_SMALL = 0,
  BLUE_BIG,
  RED_SMALL,
  RED_BIG,
  GRAY_SMALL,
  GRAY_BIG,
  PURPLE_SMALL,
  PURPLE_BIG
};
```
- 2.函数：修改了相应函数实现以适配该模型。
- 3.推理：经测试发现推理一帧真正耗时在6-9ms之间，优化了OpenVINODetector类成员函数，优化了检测图像话题订阅者回调函数的实现。经测试目前仅打开识别节点，关闭debug模式，帧率能达到150帧左右。
- 3.关于图像话题：当图像话题的订阅方处理速度高于发布方的时候，订阅者能够在图像被发布的第一时间获取到图像，当订阅方处理速度较低的时候，因为队列有长度，订阅方并不能在图像被发布的第一时间获取到图像，换句话来收，一张图片被发布，后需要等待队列前的图像都被订阅方获取并处理（或者被挤出队列）后才会被订阅者获取到，将队列长度设置为1，可以降低这种延时，保证订阅方拿到的图像总是最新的那一帧，但因为发布者和订阅者的频率不可能达到完全同步，仍然会存在1-4ms的延时。
---