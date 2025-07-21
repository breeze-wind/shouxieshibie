本模型采用cnn-lstm流程，对于单通道电信号的手写识别进行预测。
当前版本只针对保康的硬笔字母书写过程
目前模型欠拟合，训练集和验证集效果均较差
模型结构：cnn捕捉图像特征，lstm选择性保留与遗忘，构建时间与电压关系
show.py脚本用于手动看一个csv的波形（目前基本不用了）
process_samples.py用于划分数据集（有一份config是宝康的参数），结果在processed_data（需要一个标签一个标签弄，并手动配置输入输出路径），可视化结果在visualizatiions文件夹、
*****补充：process_samples新增功能：路径合成（方便路径配置），自动删除废数据集
train_model.py用于模型训练，读取processed_data中的数据，其中子目录名称即为标签名。最终标签名与模型存储与models文件夹
predict.py用于预测（模型人工检验），读取一个csv，或npy（process_samples处理后的数据）,预测结果在prediction_results里
*****predict.py读取csv功能目前不可用，请手动调用process_samples.py处理后传入npy
requirements里是版本需求，创建虚拟环境通常ide自动读取

*****fjq的奇妙目录命名逻辑，本次需要调试的数据集在各目录的test子目录中。dataset里存储的是csv原始数据，processed_data里是npy（处理，划分后的样本），predict_samples里是预测集，标xunlian的是训练集本身（用来监测是否欠拟合），标yuce的是专门录的预测集，内容是a b c d e a b c d e a b c d e。预测结果在predict_results里。