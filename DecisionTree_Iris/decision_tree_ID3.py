#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/16 11:29
# @Author  : Jack
import numpy as np

def load_data(data_path):
    """
    按要求加载样本
    :param data_path: 样本数据的存储路径
    :return: 规范化存储的样本数据
    """
    # 1. 读取数据
    with open(data_path) as f:
        data = f.readlines()
    data_size = len(data)
    # 2. 定义空列表，存放临时数据集
    data_set = []
    # 3. 对数据进行预处理
    for sample in data:
        # 3.1 将每行最后的换行符替换掉并按空格分割字符串成列表
        sample = sample.replace("\n", '').split("\t")
        if int(sample[0]) <= data_size // 3:
            label = '0'   # 0 为 setosa
        elif int(sample[0]) > data_size // 3 * 2:
            label = '2'   # 2 为 virginaca
        else:
            label = '1'   # 1 为 versicolor
        # 3.2 将标记放在每个数据的后边
        sample = sample[1:]
        sample.append(label)
        # 3.3 把处理好的数据，放到训练集当中
        data_set.append(sample)
    data_set = np.array(data_set, dtype=float)
    return np.rint(data_set)


def cal_entropy(data_set):
    """
    计算训练样本集的信息熵
    :param data_set: 训练样本集
    :return: 训练集的信息熵
    """
    labels = list(set(data_set[:, -1]))
    labels_len = len(data_set[:, -1])
    tmp_entropy = 0
    for label in labels:
        # 分别计算每个类别的数量
        tmp = sum([1 for sample in data_set if sample[-1] == label])
        tmp_entropy += (tmp / labels_len) * np.math.log(tmp / labels_len, 2)
    entropy = -tmp_entropy
    return entropy


def get_info_gain(data_set, feature: int):
    """
    计算某一特征的信息增益
    :param data_set: 训练集
    :param feature: 对应特征
    :return: 该特征的信息增益
    """
    # 特征的种类
    feature_tags = list(set(data_set[:, feature]))
    entropy4feature = 0
    for feature_tag in feature_tags:
        # 具有该特征的样本集合
        sub_data_set = [sample for sample in data_set if sample[feature]==feature_tag]
        # 列表转化为数组，使其满足cal_entropy函数
        sub_data_set = np.array(sub_data_set)
        tmp_entropy = cal_entropy(sub_data_set)
        # 累加该特征的熵值
        entropy4feature += (len(sub_data_set)/len(data_set)) * tmp_entropy
    info_gain = cal_entropy(data_set) - entropy4feature
    return info_gain


def select_feature(data_set, features: list) -> int:
    """
    根据信息增益选择切分特征
    :param data_set: 训练集 
    :param features: 样本特征
    :return: 信息增益最大对应的特征
    """
    # 定义空列表，存放各个特征的信息增益
    info_gains = []
    for feature in features:
        feature: int
        info_gain = get_info_gain(data_set, feature)
        info_gains.append(info_gain)
    # 返回信息增益最大对应的特征
    return features[info_gains.index(max(info_gains))]


def major_label(labels):
    """
    获取数量最多的label对应的label
    :param labels: 训练集的所有标记
    :return: 最多数量的标记
    """
    tags = list(set(labels))
    # 计算各个label的数量，并用数组存储
    tag_num = [sum([1 for i in labels if i==label]) for label in tags]
    k = tag_num.index(max(tag_num))
    return tags[k]


def build_tree(data_set, features) -> dict:
    """
    根据训练集生成决策树
    :param data_set: 训练集
    :param features: 样本特征编号
    :return: 决策树
    """
    # 1. 获取训练集所有标记
    labels = data_set[:, -1]
    # 2. 如果只有一种标记，则直接返回
    if len(set(labels)) == 1:
        return {'label': labels[0]}
    # 3. 如果没有特征，则返回最主要的标记
    if not len(features):
        return {'label': major_label(labels)}
    # 4. 如果 2、3 点都没有发生，则从多个特征中选择最好的特征作为根节点
    best_feature = select_feature(data_set, features)
    tree = {'feature': best_feature, 'children': {}}
    # 5. 得到根节点后，需要进一步分类
    # 5.1 获取最好特征的特征值种类
    feature_tags = list(set(data_set[:, best_feature]))
    for feature_tag in feature_tags:
        # 5.2 获取各个特征值种类对应样本
        sub_data_set = [sample for sample in data_set if sample[best_feature]==feature_tag]
        sub_data_set = np.array(sub_data_set)
        if not len(sub_data_set):
            tree['children'][feature_tag] = {'label': major_label(labels)}
        else:
            sub_features = [i for i in features if i != best_feature]
            tree['children'][feature_tag] = build_tree(sub_data_set, sub_features)
    return tree


def classify(tree: dict, sample) -> int:
    """
    根据建好的决策树对输入的单个样本进行判别分类
    :param tree: 决策树
    :param sample: 待分类的样本
    :return: 分类后的结果，标记
    """
    for k, v in tree.items():
        if k != 'feature':
            return tree['label']
        return classify(tree['children'][sample[tree['feature']]], sample)


def classifier(tree: dict, features_data, default):
    """
    根据决策树，对输入的样本集统一进行判别分类
    :param tree: 决策树
    :param features_data: 待决策的样本数据集的特征集
    :param default: 当分类器出错，或者输入的样本没有特征，则返回该默认值
    :return: 判别决策分类后的结果
    """
    predict_ret = []
    for features_sample in features_data:
        try:
            predict_val = classify(tree, features_sample)
        except KeyError:
            predict_val = default
        predict_ret.append(predict_val)
    return predict_ret

if __name__=="__main__":
    train_data = load_data("./train_data.txt")
    # print(train_data)
    test_data = load_data("./test_data.txt")
    # print(test_data)
    tree = build_tree(train_data, list(range(train_data.shape[1]-1)))
    # print(tree)
    test_data_labels = test_data[:, -1]         # 获得测试样本的全部标记
    test_data_features = test_data[:, :-1]      # 获得测试样本的全部特征
    default = major_label(test_data_labels)     # 获得测试样本的主要标记，作为默认初始值
    predict_ret = classifier(tree, test_data_features, default)
    #    print(predict_ret)
    accuracy = np.mean(np.array(predict_ret==test_data_labels))
    print(accuracy)
