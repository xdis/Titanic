"""
泰坦尼克号生存预测模型 - 高级优化版
==================================================
该代码实现了完整的机器学习流程并进行了全面优化：
1. 数据加载与探索
2. 高级数据清洗与预处理  
3. 增强特征工程与家族特征提取
4. 高级特征选择与组合
5. 多模型训练与优化
6. 堆叠与加权集成策略
7. 阈值优化与多模型投票
"""

# ---------- 导入必要的库 ----------
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, RFECV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    has_catboost = True
except ImportError:
    has_catboost = False
    print("CatBoost 未安装，将跳过相关模型")
    
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

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

# 输出原始数据信息
print(f"训练集形状: {train_data.shape}")
print(f"测试集形状: {test_data.shape}")

# 合并数据集以便统一处理
all_data = pd.concat([train_data.assign(is_train=1), 
                      test_data.assign(is_train=0)], 
                     ignore_index=True, sort=False)

# 备份PassengerId和生存标签
train_ids = train_data['PassengerId'].copy()
test_ids = test_data['PassengerId'].copy()
train_labels = train_data['Survived'].copy()

# ---------- 高级数据预处理和特征工程 ----------
print("\n========== 2. Enhanced Data Preprocessing & Feature Engineering ==========")

# ----- 2.0 异常值处理函数 -----
def handle_outliers(df, column, method='clip', lower_quantile=0.01, upper_quantile=0.99):
    """处理异常值"""
    if method == 'clip':
        lower = df[column].quantile(lower_quantile)
        upper = df[column].quantile(upper_quantile)
        df[column] = df[column].clip(lower=lower, upper=upper)
    return df

# ----- 2.1 高级名字特征提取 -----
# 从名字中提取称谓(Title)
def extract_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

all_data['Title'] = all_data['Name'].apply(extract_title)

# 映射相似的称谓
title_mapping = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Dr': 'Officer',
    'Rev': 'Officer',
    'Col': 'Officer',
    'Major': 'Officer',
    'Mlle': 'Miss',
    'Countess': 'Royalty',
    'Ms': 'Miss',
    'Lady': 'Royalty',
    'Jonkheer': 'Royalty',
    'Don': 'Royalty',
    'Dona': 'Royalty',
    'Mme': 'Mrs',
    'Capt': 'Officer',
    'Sir': 'Royalty'
}
all_data['Title'] = all_data['Title'].map(title_mapping)

# 提取姓氏并计算家族规模
all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0])
family_sizes = all_data.groupby('Surname').size()
all_data['FamilySurvival'] = all_data['Surname'].map(family_sizes)

# 先将姓氏添加到训练数据中，再计算生存率
train_data_with_surname = train_data.copy()
train_data_with_surname['Surname'] = train_data_with_surname['Name'].apply(lambda x: x.split(',')[0])

# 新增: 为姓氏添加家族生存率统计
surname_survival = train_data_with_surname.groupby('Surname')['Survived'].mean().to_dict()
# 默认值为训练集平均生存率
mean_survival = train_data['Survived'].mean()
# 应用到所有数据
all_data['FamilySurvivalRate'] = all_data['Surname'].map(surname_survival).fillna(mean_survival)

# ----- 2.2 增强票号和客舱特征提取 -----
# 从船票号提取信息
all_data['TicketPrefix'] = all_data['Ticket'].apply(lambda x: ''.join(x.split(' ')[:-1]).strip() 
                                                 if len(x.split(' ')) > 1 else 'XXXX')
all_data['TicketPrefix'] = all_data['TicketPrefix'].apply(lambda x: x if x else 'XXXX')
all_data['TicketNumber'] = all_data['Ticket'].apply(lambda x: x.split(' ')[-1])

# 新增: 票号数字特征
all_data['TicketNumberInt'] = all_data['TicketNumber'].str.extract('(\d+)').astype(float)
# 对票号数值进行分组
all_data['TicketNumberGroup'] = pd.qcut(all_data['TicketNumberInt'].fillna(0), 5, labels=False)

# 票号组合特征 - 需要修复这里的问题
# 只使用训练集计算生存率
train_ticket_survival = train_data.groupby('Ticket')['Survived'].mean()
# 应用到所有数据，测试集中没有的票号填充为平均值
all_data['TicketSurvivalRate'] = all_data['Ticket'].map(train_ticket_survival).fillna(mean_survival)

# 处理客舱号 - 提取首字母和数量
all_data['CabinFirstLetter'] = all_data['Cabin'].fillna('U').apply(lambda x: x[0])
all_data['CabinCount'] = all_data['Cabin'].fillna('').apply(lambda x: len(x.split(' ')))

# 新增: 客舱等级特征
cabin_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'U': 0}
all_data['CabinLevel'] = all_data['CabinFirstLetter'].map(cabin_mapping)

# ----- 2.3 家庭特征增强 -----
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)

# 创建更细粒度的家庭类型特征
def categorize_family_size(family_size):
    if family_size == 1:
        return 0  # 'Alone'
    elif family_size <= 3:
        return 1  # 'Small'
    elif family_size <= 6:
        return 2  # 'Medium'
    else:
        return 3  # 'Large'
        
all_data['FamilyType'] = all_data['FamilySize'].apply(categorize_family_size)

# SibSp和Parch的非线性组合
all_data['FamilyMultiplied'] = all_data['SibSp'] * all_data['Parch']
all_data['FamilyRatio'] = all_data['SibSp'] / (all_data['Parch'] + 1)

# 先处理FamilyType到训练集中
train_data_with_family = train_data.copy()
train_data_with_family['FamilySize'] = train_data_with_family['SibSp'] + train_data_with_family['Parch'] + 1
train_data_with_family['FamilyType'] = train_data_with_family['FamilySize'].apply(categorize_family_size)

# 新增: 按家庭类型和性别的存活率
family_survival_by_type = train_data_with_family.groupby(['FamilyType', 'Sex'])['Survived'].mean().to_dict()

def get_family_type_sex_survival(row):
    key = (row['FamilyType'], row['Sex'])
    return family_survival_by_type.get(key, mean_survival)

all_data['FamilyTypeSexSurvival'] = all_data.apply(get_family_type_sex_survival, axis=1)

# ----- 2.4 高级年龄特征 -----
# 使用KNN填充年龄
imputer = KNNImputer(n_neighbors=5)
features_for_age = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']

# 性别转换为数值
all_data['Sex'] = all_data['Sex'].map({'female': 1, 'male': 0})

# 准备用于填充年龄的数据
all_data['Age'] = imputer.fit_transform(pd.DataFrame({
    'Age': all_data['Age'],
    'Pclass': all_data['Pclass'],
    'SibSp': all_data['SibSp'], 
    'Parch': all_data['Parch'],
    'Fare': all_data['Fare'].fillna(all_data['Fare'].median()),
    'Sex': all_data['Sex']
}))[:,0]

# 创建多个年龄相关特征
all_data['Age2'] = all_data['Age'] ** 2  # 年龄平方
all_data['Age3'] = all_data['Age'] ** 3  # 年龄立方
all_data['AgePclass'] = all_data['Age'] * all_data['Pclass']  # 年龄与舱位等级交互
all_data['Age*Class'] = all_data['Age'] * all_data['Pclass']

# 更细致的年龄分组
def categorize_age(age):
    if age <= 5:
        return 0  # 'Baby'
    elif age <= 12:
        return 1  # 'Child'
    elif age <= 18:
        return 2  # 'Teenager'
    elif age <= 35:
        return 3  # 'Young Adult'
    elif age <= 60:
        return 4  # 'Adult'
    else:
        return 5  # 'Elderly'
        
all_data['AgeGroup'] = all_data['Age'].apply(categorize_age)

# 先将AgeGroup添加到训练数据中
train_data_with_age = train_data.copy()
train_data_with_age['Age'] = all_data.loc[all_data['is_train'] == 1, 'Age'].values
train_data_with_age['AgeGroup'] = train_data_with_age['Age'].apply(categorize_age)

# 新增: 年龄组与性别的生存率
age_sex_survival = train_data_with_age.groupby(['AgeGroup', 'Sex'])['Survived'].mean().to_dict()

def get_age_sex_survival(row):
    key = (row['AgeGroup'], row['Sex'])
    return age_sex_survival.get(key, mean_survival)

all_data['AgeSexSurvival'] = all_data.apply(get_age_sex_survival, axis=1)

# ----- 2.5 票价特征增强 -----
# 填充缺失的Fare
fare_median = all_data.groupby('Pclass')['Fare'].transform('median')
all_data['Fare'] = all_data['Fare'].fillna(fare_median)

# 处理票价异常值
all_data = handle_outliers(all_data, 'Fare', method='clip', lower_quantile=0.01, upper_quantile=0.99)

# 创建票价衍生特征
all_data['Fare2'] = all_data['Fare'] ** 2  # 票价平方
all_data['Fare3'] = all_data['Fare'] ** 3  # 票价立方
all_data['FarePerPerson'] = all_data['Fare'] / all_data['FamilySize']  # 人均票价
all_data['LogFare'] = np.log1p(all_data['Fare'])  # 票价对数

# 对票价进行分组
def categorize_fare(fare):
    if fare <= 7.75:
        return 0  # 'Very Low'
    elif fare <= 10.5:
        return 1  # 'Low'
    elif fare <= 21.6:
        return 2  # 'Mid-low'
    elif fare <= 41.6:
        return 3  # 'Mid-high'
    else:
        return 4  # 'High'
        
all_data['FareGroup'] = all_data['Fare'].apply(categorize_fare)

# 先将FareGroup添加到训练数据中
train_data_with_fare = train_data.copy()
train_data_with_fare['Fare'] = all_data.loc[all_data['is_train'] == 1, 'Fare'].values
train_data_with_fare['FareGroup'] = train_data_with_fare['Fare'].apply(categorize_fare)
train_data_with_fare['Pclass'] = train_data['Pclass']

# 新增: 票价组与舱位等级的生存率
fare_pclass_survival = train_data_with_fare.groupby(['FareGroup', 'Pclass'])['Survived'].mean().to_dict()

def get_fare_pclass_survival(row):
    key = (row['FareGroup'], row['Pclass'])
    return fare_pclass_survival.get(key, mean_survival)

all_data['FarePclassSurvival'] = all_data.apply(get_fare_pclass_survival, axis=1)

# ----- 2.6 登船港口特征 -----
# 填充缺失的Embarked
all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode()[0])

# 登船港口与其他特征交互
embeddings = {'S': 0, 'C': 1, 'Q': 2}
all_data['EmbarkedCode'] = all_data['Embarked'].map(embeddings)
all_data['Pclass*Embarked'] = all_data['Pclass'] * all_data['EmbarkedCode']

# 修复: 先将Sex数值转换添加到训练数据
train_data_with_embarked = train_data.copy()
train_data_with_embarked['Sex'] = train_data_with_embarked['Sex'].map({'female': 1, 'male': 0})

# 新增: 港口与性别的生存率
embarked_sex_survival = train_data_with_embarked.groupby(['Embarked', 'Sex'])['Survived'].mean().to_dict()

def get_embarked_sex_survival(row):
    key = (row['Embarked'], row['Sex'])
    return embarked_sex_survival.get(key, mean_survival)

all_data['EmbarkedSexSurvival'] = all_data.apply(get_embarked_sex_survival, axis=1)

# ----- 2.7 创建交互特征 -----
# 年龄和性别交互
all_data['Age*Sex'] = all_data['Age'] * all_data['Sex']

# 客舱等级和性别交互
all_data['Pclass*Sex'] = all_data['Pclass'] * all_data['Sex']

# 票价和客舱等级交互
all_data['Fare*Pclass'] = all_data['Fare'] * all_data['Pclass']

# 是否单独旅行与性别交互
all_data['IsAlone*Sex'] = all_data['IsAlone'] * all_data['Sex']

# 家庭规模与票价交互
all_data['FamilySize*Fare'] = all_data['FamilySize'] * all_data['Fare']

# 新增: 三重交互特征
all_data['Sex*Age*Pclass'] = all_data['Sex'] * all_data['Age'] * all_data['Pclass']
all_data['Embarked*Fare'] = all_data['EmbarkedCode'] * all_data['Fare']

# 新增: 多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(all_data[['Sex', 'Pclass', 'Fare']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['Sex', 'Pclass', 'Fare']))

# 添加多项式特征到数据集
all_data['Poly_Sex*Pclass'] = poly_df['Sex Pclass']
all_data['Poly_Sex*Fare'] = poly_df['Sex Fare']
all_data['Poly_Pclass*Fare'] = poly_df['Pclass Fare']

# ----- 2.8 类别特征编码 -----
# 对类别特征进行独热编码
cat_features = ['Embarked', 'Title', 'CabinFirstLetter', 'TicketPrefix']
all_data = pd.get_dummies(all_data, columns=cat_features, drop_first=True)

# ----- 2.9 删除冗余特征 -----
drop_columns = ['Ticket', 'Name', 'Cabin', 'Surname', 'TicketNumber', 'PassengerId']
all_data.drop(drop_columns, axis=1, inplace=True)

# ----- 2.10 特征规范化 -----
# 对数值特征进行标准化
num_features = ['Age', 'Fare', 'FarePerPerson', 'LogFare', 'Fare2', 'Fare3', 'Age2', 'Age3', 'AgePclass']
scaler = StandardScaler()
all_data[num_features] = scaler.fit_transform(all_data[num_features])

# 在特征选择前添加全面的数据检查
print("\n----- 2.11 数据质量检查 -----")
# 检查合并后数据中的NaN值
nan_columns = all_data.columns[all_data.isna().any()].tolist()
if nan_columns:
    print(f"警告：发现含有NaN值的列: {nan_columns}")
    print("列出每列的NaN值数量:")
    print(all_data[nan_columns].isna().sum())
    
    # 使用SimpleImputer填充NaN
    print("使用均值/众数填充NaN值...")
    # 数值特征用均值填充
    num_imputer = SimpleImputer(strategy='mean')
    num_cols = all_data.select_dtypes(include=['float64', 'int64']).columns
    all_data[num_cols] = num_imputer.fit_transform(all_data[num_cols])
    
    # 检查是否还有NaN
    if all_data.isna().sum().sum() > 0:
        print("仍有NaN值，使用0填充...")
        all_data.fillna(0, inplace=True)
else:
    print("数据质量检查通过，没有发现NaN值。")

# 检查是否有无穷大值
inf_check = np.isinf(all_data.select_dtypes(include=['float64']).values).sum()
if inf_check > 0:
    print(f"警告：发现 {inf_check} 个无穷大值")
    # 替换无穷大值
    all_data.replace([np.inf, -np.inf], 0, inplace=True)
    print("已将无穷大值替换为0")
    
# ---------- 特征选择 ----------
print("\n========== 3. Advanced Feature Selection ==========")

# 分离训练数据和测试数据
X_all = all_data[all_data['is_train'] == 1].drop(['Survived', 'is_train'], axis=1)
X_test_final = all_data[all_data['is_train'] == 0].drop(['Survived', 'is_train'], axis=1)
y_all = train_labels

# 新增：使用RFECV进行特征选择
print("使用递归特征消除选择最佳特征...")
rfecv = RFECV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42), 
    step=1, 
    cv=StratifiedKFold(5),
    scoring='accuracy',
    min_features_to_select=10
)
rfecv.fit(X_all, y_all)
print(f"最佳特征数量: {rfecv.n_features_}")

# 选择重要特征
rfecv_support = rfecv.support_
X_rfecv = X_all.loc[:, rfecv_support]
X_test_rfecv = X_test_final.loc[:, rfecv_support]

print(f"RFECV选择的特征数量: {X_rfecv.shape[1]}")
print("RFECV选择的特征: ", X_rfecv.columns.tolist()[:10], "等...")

# 使用互信息评估特征重要性
def select_features(X, y, X_test, threshold=0.005):  # 降低阈值
    # 计算每个特征的互信息分数
    mi_scores = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    
    # 按互信息分数排序
    mi_scores = mi_scores.sort_values(ascending=False)
    
    # 选择分数高于阈值的特征
    important_features = mi_scores[mi_scores > threshold].index.tolist()
    
    print(f"Mutual Info: 选择了 {len(important_features)} 个特征，总共 {X.shape[1]} 个")
    print("Top 10 Mutual Info features:", important_features[:10])
    
    return X[important_features], X_test[important_features], important_features

# 特征选择
X_selected, X_test_selected, selected_features = select_features(X_all, y_all, X_test_final)

# 可视化特征重要性（使用随机森林）
def plot_feature_importance(X, y, feature_names, n=20):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 获取特征重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 绘制前n个重要特征
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.barh(range(min(n, len(indices))), 
             importances[indices[:n]], 
             align='center')
    plt.yticks(range(min(n, len(indices))), 
               [feature_names[i] for i in indices[:n]])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('titanic_feature_importance.png')
    plt.show()

# 绘制特征重要性
plot_feature_importance(X_selected, y_all, X_selected.columns)

# 合并两种特征选择方法的结果
combined_features = list(set(X_rfecv.columns.tolist() + selected_features))
X_combined = X_all[combined_features]
X_test_combined = X_test_final[combined_features]

print(f"合并后的特征数量: {len(combined_features)}")

# ---------- 模型训练与评估 ----------
print("\n========== 4. Advanced Model Training and Evaluation ==========")

# 准备交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 高级模型定义
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42, C=0.1, solver='liblinear'),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=4, 
                                          min_samples_leaf=2, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, 
                                                  max_depth=4, min_samples_split=2, random_state=42),
    "SVC": SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, 
                            colsample_bytree=0.8, subsample=0.8, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=200, max_depth=-1, learning_rate=0.05, 
                              num_leaves=31, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}

# 如果有CatBoost，则添加
if has_catboost:
    models["CatBoost"] = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, 
                                           random_seed=42, verbose=0)

# 训练并评估模型
def train_and_evaluate(models, X, y, cv):
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        # 避免编码问题
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        
        # 记录结果
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 训练完整模型
        model.fit(X, y)
        
    return results

# 训练并评估模型
model_results = train_and_evaluate(models, X_combined, y_all, skf)

# ---------- 模型调优 ----------
print("\n========== 5. Advanced Model Tuning ==========")

# 后期优化：为最佳模型进行参数微调
best_models = {}

# 定义所有模型的参数网格
# XGBoost参数网格
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 2, 3]
}

# LightGBM参数网格
lgbm_param_grid = {
    'n_estimators': [100, 200, 300],
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [-1, 5, 10],
    'subsample': [0.7, 0.8, 0.9]
}

# 随机森林参数网格
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 8, 10, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

# CatBoost参数网格
if has_catboost:
    catboost_param_grid = {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5]
    }

# 优化XGBoost
print("\nFine-tuning XGBoost...")
xgb_grid = GridSearchCV(XGBClassifier(random_state=42), 
                       param_grid=xgb_param_grid, 
                       cv=skf, 
                       scoring='accuracy',
                       verbose=1,
                       n_jobs=1)

xgb_grid.fit(X_combined, y_all)

best_xgb = xgb_grid.best_estimator_
print(f"Best XGBoost parameters: {xgb_grid.best_params_}")
print(f"Best XGBoost CV score: {xgb_grid.best_score_:.4f}")
best_models['XGBoost'] = best_xgb

# 优化LightGBM
print("\nFine-tuning LightGBM...")
lgbm_grid = GridSearchCV(LGBMClassifier(random_state=42), 
                        param_grid=lgbm_param_grid, 
                        cv=skf, 
                        scoring='accuracy',
                        verbose=1,
                        n_jobs=1)

lgbm_grid.fit(X_combined, y_all)

best_lgbm = lgbm_grid.best_estimator_
print(f"Best LightGBM parameters: {lgbm_grid.best_params_}")
print(f"Best LightGBM CV score: {lgbm_grid.best_score_:.4f}")
best_models['LightGBM'] = best_lgbm

# 优化随机森林
print("\nFine-tuning Random Forest...")
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                      param_grid=rf_param_grid, 
                      cv=skf, 
                      scoring='accuracy',
                      verbose=1,
                      n_jobs=1)

rf_grid.fit(X_combined, y_all)

best_rf = rf_grid.best_estimator_
print(f"Best Random Forest parameters: {rf_grid.best_params_}")
print(f"Best Random Forest CV score: {rf_grid.best_score_:.4f}")
best_models['RandomForest'] = best_rf

# 如果有CatBoost，则优化
if has_catboost:
    print("\nFine-tuning CatBoost...")
    catboost_grid = GridSearchCV(CatBoostClassifier(random_seed=42, verbose=0), 
                                param_grid=catboost_param_grid, 
                                cv=skf, 
                                scoring='accuracy',
                                verbose=1,
                                n_jobs=1)

    catboost_grid.fit(X_combined, y_all)

    best_catboost = catboost_grid.best_estimator_
    print(f"Best CatBoost parameters: {catboost_grid.best_params_}")
    print(f"Best CatBoost CV score: {catboost_grid.best_score_:.4f}")
    best_models['CatBoost'] = best_catboost

# ---------- 堆叠和集成 ----------
print("\n========== 6. Stacking & Ensemble ==========")

# 创建一个强大的集成模型
print("Creating advanced ensemble model...")

# 基于交叉验证分数调整权重
cv_scores = {
    'XGBoost': xgb_grid.best_score_,
    'LightGBM': lgbm_grid.best_score_,
    'RandomForest': rf_grid.best_score_,
    'GradientBoosting': model_results['GradientBoosting']['cv_mean']
}

if has_catboost:
    cv_scores['CatBoost'] = catboost_grid.best_score_

# 权重计算基于模型性能
total_score = sum(cv_scores.values())
weights = {model: score/total_score for model, score in cv_scores.items()}
print("计算的模型权重:", weights)

# 将最佳模型添加到集成中
ensemble_models = []
weight_values = []

# 添加最佳模型和权重
best_models_with_weights = [
    ('XGBoost', best_xgb, weights['XGBoost']),
    ('LightGBM', best_lgbm, weights['LightGBM']),
    ('RandomForest', best_rf, weights['RandomForest']),
    ('GradientBoosting', models['GradientBoosting'], weights['GradientBoosting'])
]

if has_catboost:
    best_models_with_weights.append(('CatBoost', best_catboost, weights['CatBoost']))

for name, model, weight in best_models_with_weights:
    ensemble_models.append((name, model))
    weight_values.append(weight)

weighted_ensemble = VotingClassifier(
    estimators=ensemble_models,
    voting='soft',
    weights=weight_values
)

weighted_ensemble.fit(X_combined, y_all)

# 交叉验证评估集成模型
ensemble_cv_scores = cross_val_score(weighted_ensemble, X_combined, y_all, 
                                    cv=skf, scoring='accuracy', n_jobs=1)
print(f"Weighted Ensemble CV Accuracy: {ensemble_cv_scores.mean():.4f} ± {ensemble_cv_scores.std():.4f}")

# 创建堆叠分类器
print("\nCreating stacked model...")
estimators = [
    ('xgb', best_xgb),
    ('lgbm', best_lgbm),
    ('rf', best_rf),
    ('gb', models['GradientBoosting'])
]

if has_catboost:
    estimators.append(('catboost', best_catboost))

# 使用逻辑回归作为元学习器
stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=1
)

# 训练堆叠模型
stacked_model.fit(X_combined, y_all)

# 评估堆叠模型
stacked_cv_scores = cross_val_score(stacked_model, X_combined, y_all, 
                                   cv=skf, scoring='accuracy', n_jobs=1)
print(f"Stacked Model CV Accuracy: {stacked_cv_scores.mean():.4f} ± {stacked_cv_scores.std():.4f}")

# ---------- 预测与结果输出 ----------
print("\n========== 7. Prediction & Output ==========")

# 对测试集进行预测
print("生成最终预测...")

# 获取加权集成的概率预测
ensemble_proba = weighted_ensemble.predict_proba(X_test_combined)
stacked_proba = stacked_model.predict_proba(X_test_combined)

# 标准预测（阈值0.5）
ensemble_predictions = weighted_ensemble.predict(X_test_combined)
stacked_predictions = stacked_model.predict(X_test_combined)

# 尝试不同的阈值
threshold = 0.465  # 稍微倾向于预测0，可微调
ensemble_threshold_predictions = (ensemble_proba[:, 1] > threshold).astype(int)
stacked_threshold_predictions = (stacked_proba[:, 1] > threshold).astype(int)

# 创建提交文件
submissions = {
    'weighted_ensemble': ensemble_predictions,
    'stacked': stacked_predictions,
    'ensemble_threshold': ensemble_threshold_predictions,
    'stacked_threshold': stacked_threshold_predictions
}

# 输出所有预测版本
for name, predictions in submissions.items():
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions
    })
    
    submission_path = os.path.join(current_dir, f'titanic_submission_{name}.csv')
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")

# 模型投票（取多数）
all_model_preds = []
for name, model in best_models.items():
    preds = model.predict(X_test_combined)
    all_model_preds.append(preds)

# 添加集成和堆叠模型的预测
all_model_preds.append(ensemble_predictions)
all_model_preds.append(stacked_predictions)

# 转换为numpy数组并计算多数投票
all_model_preds = np.array(all_model_preds)
final_predictions = np.apply_along_axis(
    lambda x: np.bincount(x).argmax(), 
    axis=0, 
    arr=all_model_preds
)

# 创建最终投票提交文件
final_submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': final_predictions
})

final_submission_path = os.path.join(current_dir, 'titanic_submission_final_vote.csv')
final_submission.to_csv(final_submission_path, index=False)
print(f"Final vote submission saved to: {final_submission_path}")

# 绘制模型比较图
plt.figure(figsize=(14, 8))
all_model_names = list(model_results.keys()) + ['Weighted Ensemble', 'Stacked Model']
all_cv_scores = [model_results[name]['cv_mean'] for name in all_model_names[:-2]] + [ensemble_cv_scores.mean(), stacked_cv_scores.mean()]

# 按准确率排序
sorted_indices = np.argsort(all_cv_scores)[::-1]
sorted_names = [all_model_names[i] for i in sorted_indices]
sorted_scores = [all_cv_scores[i] for i in sorted_indices]
sorted_errors = [model_results[name]['cv_std'] if name in model_results 
                else (ensemble_cv_scores.std() if name == 'Weighted Ensemble' else stacked_cv_scores.std()) 
                for name in sorted_names]

# 使用条形图显示结果
bars = plt.bar(range(len(sorted_names)), sorted_scores, 
              yerr=sorted_errors, align='center', alpha=0.7, 
              color=['#ff9999' if name in ['Weighted Ensemble', 'Stacked Model'] else '#66b3ff' for name in sorted_names], 
              ecolor='black', capsize=5)
              
plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
plt.ylim(0.75, 0.9)  # 设置y轴范围，使差异更明显
plt.xlabel('Model')
plt.ylabel('Cross-validation Accuracy')
plt.title('Advanced Model Performance Comparison')

# 在柱状图上显示准确率值
for bar, score in zip(bars, sorted_scores):
    plt.text(bar.get_x() + bar.get_width()/2, 
             bar.get_height() + 0.005, 
             f'{score:.4f}', 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('titanic_advanced_model_comparison.png', dpi=300)
plt.show()

print("\nAdvanced analysis complete! All charts and submission files have been saved.")
print("建议尝试提交以下文件，按预期性能排序:")
print("1. titanic_submission_final_vote.csv (投票集成)")
print("2. titanic_submission_stacked_threshold.csv (调整阈值的堆叠模型)")
print("3. titanic_submission_ensemble_threshold.csv (调整阈值的加权集成)")
print("4. titanic_submission_stacked.csv (标准堆叠模型)")
print("5. titanic_submission_weighted_ensemble.csv (标准加权集成)")