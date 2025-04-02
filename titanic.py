"""
泰坦尼克号生存预测模型
==================================================
该代码实现了完整的机器学习流程：
1. 数据加载与探索
2. 数据清洗与预处理  
3. 特征工程
4. 数据可视化分析
5. 多模型训练与评估
6. 模型优化与集成
7. 预测与结果导出
"""

# ---------- 导入必要的库 ----------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
import tempfile
warnings.filterwarnings('ignore')  # 忽略警告信息

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ---------- 数据加载 ----------
print("========== 1. Data Loading ==========")
# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建到titanic子文件夹中文件的相对路径
train_path = os.path.join(current_dir, "titanic", "train.csv")
test_path = os.path.join(current_dir, "titanic", "test.csv")

# 读取数据
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# 查看数据基本信息
print("\nTraining data (first 5 rows):")
print(train_data.head())
print("\nTraining data info:")
print(train_data.info())
print("\nTraining data description:")
print(train_data.describe())

# 检查缺失值情况
print("\nMissing values in training data:")
print(train_data.isnull().sum())
print("\nMissing values in test data:")
print(test_data.isnull().sum())

# ---------- 数据预处理 ----------
print("\n========== 2. Data Preprocessing ==========")

# ----- 2.1 缺失值处理 -----
print("Handling missing values...")

# 处理Age缺失值：用不同称呼(Title)的人群的年龄中位数填充
# 从Name中提取称呼(Title)
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 将称呼分类合并成更少的类别
title_mapping = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Dr': 'Rare',
    'Rev': 'Rare',
    'Col': 'Rare',
    'Major': 'Rare',
    'Mlle': 'Miss',
    'Countess': 'Rare',
    'Ms': 'Miss',
    'Lady': 'Rare',
    'Jonkheer': 'Rare',
    'Don': 'Rare',
    'Dona': 'Rare',
    'Mme': 'Mrs',
    'Capt': 'Rare',
    'Sir': 'Rare'
}
train_data['Title'] = train_data['Title'].map(title_mapping)
test_data['Title'] = test_data['Title'].map(title_mapping)

# 使用不同Title类别的Age中位数填充缺失的Age
for title in train_data['Title'].unique():
    age_median = train_data[train_data['Title'] == title]['Age'].median()
    train_data.loc[(train_data['Age'].isnull()) & (train_data['Title'] == title), 'Age'] = age_median
    test_data.loc[(test_data['Age'].isnull()) & (test_data['Title'] == title), 'Age'] = age_median

# 如果仍有缺失，使用总体中位数填充
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)

# 处理Embarked缺失值：用众数填充
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# 处理Fare缺失值：用对应Pclass的Fare中位数填充
for pclass in range(1, 4):
    fare_median = train_data[train_data['Pclass'] == pclass]['Fare'].median()
    test_data.loc[(test_data['Fare'].isnull()) & (test_data['Pclass'] == pclass), 'Fare'] = fare_median

# 删除Cabin列（缺失太多）
train_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

# ----- 2.2 特征工程 -----
print("Feature engineering...")

# 性别转换为数值
train_data['Sex'] = train_data['Sex'].map({'female': 1, 'male': 0})
test_data['Sex'] = test_data['Sex'].map({'female': 1, 'male': 0})

# 创建家庭规模特征
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# 创建是否单独旅行的特征
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

# 创建家庭类型特征
def categorize_family_size(family_size):
    if family_size == 1:
        return 0  # 'Alone'
    elif family_size <= 4:
        return 1  # 'Small'
    else:
        return 2  # 'Large'
        
train_data['FamilyType'] = train_data['FamilySize'].apply(categorize_family_size)
test_data['FamilyType'] = test_data['FamilySize'].apply(categorize_family_size)

# 创建年龄组特征
def categorize_age(age):
    if age <= 12:
        return 0  # 'Child'
    elif age <= 18:
        return 1  # 'Teenager'
    elif age <= 65:
        return 2  # 'Adult'
    else:
        return 3  # 'Elderly'
        
train_data['AgeGroup'] = train_data['Age'].apply(categorize_age)
test_data['AgeGroup'] = test_data['Age'].apply(categorize_age)

# 票价组特征
def categorize_fare(fare):
    if fare <= 7.91:
        return 0  # 'Low'
    elif fare <= 14.454:
        return 1  # 'Mid-low'
    elif fare <= 31:
        return 2  # 'Mid-high'
    else:
        return 3  # 'High'
        
train_data['FareGroup'] = train_data['Fare'].apply(categorize_fare)
test_data['FareGroup'] = test_data['Fare'].apply(categorize_fare)

# 转换Embarked（独热编码）
train_data = pd.get_dummies(train_data, columns=['Embarked', 'Title'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Embarked', 'Title'], drop_first=True)

# 确保测试集和训练集有相同的特征列（处理独热编码后可能的列不匹配问题）
for col in train_data.columns:
    if col not in test_data.columns and col != 'Survived':
        test_data[col] = 0
for col in test_data.columns:
    if col not in train_data.columns:
        train_data[col] = 0

# 删除不再需要的列
drop_columns = ['Ticket', 'Name', 'PassengerId']
train_data.drop(drop_columns, axis=1, inplace=True)
test_passenger_ids = test_data['PassengerId']  # 保存测试集ID
test_data.drop(drop_columns, axis=1, inplace=True)

print("Processed features:", train_data.columns.tolist())

# ---------- 数据可视化分析 ----------
print("\n========== 3. Data Visualization ==========")

# 设置图表风格
sns.set(style="whitegrid")
plt.figure(figsize=(15, 10))

# 性别与生存率关系
plt.subplot(2, 3, 1)
sns.barplot(x='Sex', y='Survived', data=train_data, palette='Set2')
plt.title('Impact of Gender on Survival')
plt.xlabel('Gender (0=Male, 1=Female)')
plt.ylabel('Survival Rate')

# 年龄组与生存率关系
plt.subplot(2, 3, 2)
sns.barplot(x='AgeGroup', y='Survived', data=train_data, palette='Set2')
plt.title('Impact of Age Group on Survival')
plt.xlabel('Age Group (0=Child, 1=Teen, 2=Adult, 3=Elderly)')
plt.ylabel('Survival Rate')

# 客舱等级与生存率关系
plt.subplot(2, 3, 3)
sns.barplot(x='Pclass', y='Survived', data=train_data, palette='Set2')
plt.title('Impact of Passenger Class on Survival')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

# 家庭类型与生存率关系
plt.subplot(2, 3, 4)
sns.barplot(x='FamilyType', y='Survived', data=train_data, palette='Set2')
plt.title('Impact of Family Type on Survival')
plt.xlabel('Family Type (0=Alone, 1=Small, 2=Large)')
plt.ylabel('Survival Rate')

# 票价组与生存率关系
plt.subplot(2, 3, 5)
sns.barplot(x='FareGroup', y='Survived', data=train_data, palette='Set2')
plt.title('Impact of Fare Group on Survival')
plt.xlabel('Fare Group (0=Low, 1=Mid-low, 2=Mid-high, 3=High)')
plt.ylabel('Survival Rate')

# 是否单独旅行与生存率关系
plt.subplot(2, 3, 6)
sns.barplot(x='IsAlone', y='Survived', data=train_data, palette='Set2')
plt.title('Impact of Traveling Alone on Survival')
plt.xlabel('Is Alone (0=No, 1=Yes)')
plt.ylabel('Survival Rate')

plt.tight_layout()
plt.savefig('titanic_survival_analysis.png')  # 保存图表
plt.show()

# 特征相关性热图
plt.figure(figsize=(12, 10))
corr = train_data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # 创建上三角掩码
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('titanic_correlation_matrix.png')
plt.show()

# ---------- 模型训练与评估 ----------
print("\n========== 4. Model Training and Evaluation ==========")

# 准备数据
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_data)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义要评估的模型
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# 评估不同模型
print("Evaluating different models...")
model_scores = {}

for name, model in models.items():
    # 使用交叉验证评估模型
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    
    # 在完整训练集上训练，然后在验证集上评估
    model.fit(X_train, y_train)
    val_score = accuracy_score(y_val, model.predict(X_val))
    
    # 保存结果
    model_scores[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'val_score': val_score
    }
    
    print(f"{name}:")
    print(f"  Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Validation accuracy: {val_score:.4f}")
    print("  Classification Report:")
    print(classification_report(y_val, model.predict(X_val)))
    print("")

# ---------- 模型优化 ----------
print("\n========== 5. Model Optimization ==========")

# 避免使用n_jobs=-1和网格搜索，直接手动调整参数
print("Optimizing Random Forest...")

# 手动调整随机森林模型
rf_models = []
rf_scores = []

rf_params = [
    {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 2},
    {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2},
    {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 5}
]

for i, params in enumerate(rf_params):
    print(f"Testing RF parameters set {i+1}: {params}")
    rf = RandomForestClassifier(random_state=42, **params)
    score = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy').mean()
    rf_models.append(rf)
    rf_scores.append(score)
    print(f"  CV accuracy: {score:.4f}")

# 选择最佳随机森林模型
best_rf_idx = np.argmax(rf_scores)
best_rf = rf_models[best_rf_idx]
best_rf.fit(X_scaled, y)
print(f"Best Random Forest parameters: {rf_params[best_rf_idx]}")
print(f"Best Random Forest CV accuracy: {rf_scores[best_rf_idx]:.4f}")

# 梯度提升模型优化
print("\nOptimizing Gradient Boosting...")
gb_models = []
gb_scores = []

gb_params = [
    {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3},
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3}
]

for i, params in enumerate(gb_params):
    print(f"Testing GB parameters set {i+1}: {params}")
    gb = GradientBoostingClassifier(random_state=42, **params)
    score = cross_val_score(gb, X_scaled, y, cv=5, scoring='accuracy').mean()
    gb_models.append(gb)
    gb_scores.append(score)
    print(f"  CV accuracy: {score:.4f}")

# 选择最佳梯度提升模型
best_gb_idx = np.argmax(gb_scores)
best_gb = gb_models[best_gb_idx]
best_gb.fit(X_scaled, y)
print(f"Best Gradient Boosting parameters: {gb_params[best_gb_idx]}")
print(f"Best Gradient Boosting CV accuracy: {gb_scores[best_gb_idx]:.4f}")

# 使用最佳模型进行集成
print("\nCreating ensemble model...")
best_lr = LogisticRegression(max_iter=1000, random_state=42)
best_lr.fit(X_scaled, y)

ensemble = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('gb', best_gb),
        ('lr', best_lr)
    ],
    voting='soft'
)

ensemble.fit(X_scaled, y)
ensemble_cv_score = cross_val_score(ensemble, X_scaled, y, cv=5, scoring='accuracy').mean()
print(f"Ensemble model CV accuracy: {ensemble_cv_score:.4f}")

# 对测试集进行预测
print("\n========== 6. Prediction and Submission ==========")

# 使用集成模型进行预测
final_predictions = ensemble.predict(test_scaled)

# 创建提交文件
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': final_predictions
})

submission_path = os.path.join(current_dir, 'titanic_submission_optimized.csv')
submission.to_csv(submission_path, index=False)
print(f"Submission file saved to: {submission_path}")

# 绘制模型比较图
plt.figure(figsize=(10, 6))
model_names = list(model_scores.keys()) + ['Ensemble']
cv_scores = [model_scores[name]['cv_mean'] for name in model_names[:-1]] + [ensemble_cv_score]

# 按准确率排序
sorted_indices = np.argsort(cv_scores)[::-1]
sorted_names = [model_names[i] for i in sorted_indices]
sorted_scores = [cv_scores[i] for i in sorted_indices]

bars = plt.bar(sorted_names, sorted_scores, color='skyblue')
plt.ylim(0.7, 0.9)  # 设置y轴范围，使差异更明显
plt.xlabel('Model')
plt.ylabel('Cross-validation Accuracy')
plt.title('Model Performance Comparison')

# 在柱状图上显示准确率值
for bar, score in zip(bars, sorted_scores):
    plt.text(bar.get_x() + bar.get_width()/2, 
             bar.get_height() + 0.005, 
             f'{score:.4f}', 
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('titanic_model_comparison.png')
plt.show()

print("\nAnalysis complete! All charts and submission file have been saved.")