# @File: 鸢尾花数据集.py
# @Author: chen_song
# @Time: 2025-04-04 9:22

# 加载鸢尾花数据集
from sklearn.datasets import load_iris
# 数据集划分操作
from sklearn.model_selection import train_test_split
# 分类器KNN K近邻分类器
from sklearn.neighbors import KNeighborsClassifier
# 指标库 准确率分数
from sklearn.metrics import accuracy_score

# 1.加载鸢尾花数据集
iris = load_iris()
X = iris.get('data')
y = iris.get('target')

# 2.划分训练集以及测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
# 3.创建KNN分类器
knn = KNeighborsClassifier(n_neighbors=3)
# 4.训练模型
knn.fit(X_train,y_train)
# 5.预测
y_pred = knn.predict(X_test)
# 6.评估模型 说白了，就是两个分数相减
accuracy = accuracy_score(y_test,y_pred)
print(f"模型准确率：{accuracy}")
