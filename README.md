# 验证码识别

Ref: [使用 Keras 来破解 captcha 验证码](https://ypwhs.github.io/captcha/)

4位验证码，最终结果**accuracy: 0.9706, loss: 0.5285** = 0.1078 + 0.1180 + 0.1762 + 0.1265

-----

## 神经网络模型

<img src='./assets/model.png'>

核心就是神经网络的设计，这里基于作者的基础做了很多尝试，总结经验如下：

- 神经网络模型太简单（比如只有2层卷积网络），会有strong bias，最终难以达到一个好的预测结果
- 模型太复杂（比如10层），收敛会变得非常缓慢。当然我相信数据量充足（这里几乎是无限）的情况下，可以达到更好的效果，不过我等不及了。这是一个trade off
- 2层卷积+1层Pooling的组合多次感觉效果最佳。我尝试过“2层卷积+1层Pooling”和"3层卷积+1层Pooling"多种情况，都不如之前。
- strides=1, padding='same'是蛮不错的选择，虽然默认是valid
- 个人觉得，我们有必要不断做卷积操作提起特征，然后Pooling，直到图形的size缩小到一定经验值以内（比如10），这样才算是最充分“榨取”了图像的特征信息。再多卷积可能得不到更多有价值信息了，因为已经只有10 size，而Conv的kernel_size和Stride结合起来会马上超过。
- Adadelta, Adagrad, RMSprop是不错的Optimizor选择。Ref:[深度学习最全优化方法总结比较（SGD，Adagrad，Adadelta，Adam，Adamax，Nadam）](https://zhuanlan.zhihu.com/p/22252270)


其它方面需要注意的点：

- `captcha.generate_image`生成的验证码是(height, width, channel)的格式，不是(width, height, channel)
- 使用generator而不是fixed size datasets，这里X和y的格式需要注意

```
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    # not like this
	# y = np.zeros((batch_size, n_len, n_class))
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
```

因为我们最终预测的结果是4个完全独立无关的字母/数字，它们的组合是无意义的，因此预测结果最好是4个，而不是1个'字符串'。神经网络肯定会自动感知到预测的第一个和图像最左边的相关，最后一个和最右边相关。
