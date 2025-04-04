# Titanic
# 泰坦尼克号生存预测模型

![泰坦尼克号](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/300px-RMS_Titanic_3.jpg)

## Kaggle 分数: 0.77272

[Kaggle竞赛排行榜](https://www.kaggle.com/competitions/titanic/leaderboard)

使用随机森林、梯度提升和逻辑回归的集成方法进行增强型泰坦尼克号生存预测。特征包括：家庭指标、年龄分组、票价类别和称谓提取。交叉验证准确率约为82%。

## 项目背景

泰坦尼克号的沉没是历史上最著名的海难之一。1912年4月15日，在首航期间，泰坦尼克号与冰山相撞后沉没，2,224名乘客和船员中有1,502人遇难。这一惨剧震惊了国际社会，并促使船舶安全法规得到改善。

本项目利用机器学习技术，基于乘客的性别、年龄、舱位等级、票价等特征来预测哪些乘客在这场灾难中幸存下来。这一挑战特别有意义，因为生存并非随机发生——某些特定群体比其他群体更有可能生存。

## 项目特点

本解决方案实现了完整的机器学习工作流程：

1. **数据加载与探索** - 全面分析可用数据
2. **数据清洗与预处理** - 采用适当策略处理缺失值
3. **特征工程** - 从现有数据创建新特征：
   - 从姓名中提取称谓
   - 家庭规模指标构建
   - 年龄组分类
   - 票价金额分组
4. **数据可视化分析** - 生存模式的广泛视觉分析
5. **多模型训练与评估** - 比较各种算法的性能
6. **模型优化与集成** - 参数微调和模型组合
7. **预测与结果导出** - 生成提交文件

## 结果分析

最终模型在Kaggle测试集上达到了**0.77272**的准确率，采用了集成方法，结合了：
- 随机森林
- 梯度提升
- 逻辑回归

分析中的关键发现：
- 性别是生存预测的最强指标
- 乘客舱位等级有显著影响
- 儿童有更高的生存率
- 家庭规模影响生存概率

## 使用方法

### 环境要求

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### 运行代码

1. 克隆此仓库：
```bash
git clone https://github.com/xdis/Titanic.git
cd Titanic
```

2. 确保数据集位于正确位置：
```
titanic/
├── train.csv
└── test.csv
```

3. 运行预测脚本：
```bash
python titanic.py
```

4. 脚本将生成：
- 用于Kaggle提交的文件：`titanic_submission_optimized.csv`
- 可视化图表：
  - titanic_survival_analysis.png - 各特征对生存的影响分析
  - titanic_correlation_matrix.png - 特征相关性矩阵
  - titanic_model_comparison.png - 模型性能比较

## 代码结构

代码按照以下逻辑组织：
1. 导入必要的库
2. 数据加载
3. 数据预处理与缺失值处理
4. 特征工程（创建家庭规模、年龄组等特征）
5. 数据可视化分析
6. 多模型训练与评估
7. 模型优化
8. 集成学习与预测

## 可以通过以下命令安装依赖：
```bash
# 安装基本依赖
pip install -r requirements.txt

# 或安装精确版本依赖（如果提供）
pip install -r requirements_exact.txt
```

## 许可证

本项目采用MIT许可证开源。

## 贡献

欢迎通过Issue或Pull Request提出改进建议和贡献代码。