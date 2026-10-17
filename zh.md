---
layout: default
title: Yuchen Li | 中文简介
lang: zh-CN
---

# Yuchen Li

南方科技大学本科生，专业为 **Data Science and Big Data Technology**，攻读理学学士学位。

目前准备申请香港、新加坡的硕士项目，同时寻找实习机会。项目涵盖数据分析、机器学习与数据计算系统，注重实验复现、结果验证，以及个人与团队贡献的区分。

[English](./) · [GitHub](https://github.com/0zz0ruby) ·
[Gmail](mailto:rubyli2525@gmail.com) · [邮箱](mailto:lyclyc555@126.com)

## 项目

### 1. DataAgent-SFT

筛选数据分析任务轨迹，对 Qwen3.5-0.8B 的文本骨干进行监督微调，并将微调后的 Data Agent 用于 CrossMetric AI 电商分析原型。

从 11,707 条输入中选择 2,500 条轨迹，按分组互斥原则划分为 2,000 条训练数据与 500 条验证数据；验证 token 损失从 0.731607 降至 0.488760。这一指标不等同于任务成功率。

[项目详情（英文）](projects/data-agent-sft.html) ·
[代码与复现](https://github.com/0zz0ruby/data-agent-sft)

### 2. Meteorite Vision

研究训练集与测试集分布偏移下的陨石分类，并独立开展 DCGAN / WGAN-GP 图像生成实验。个人分类工作包括 ResNet-18 基线、DINOv2 训练与消融、外部图像处理，以及投票集成方案的提出。

个人模型最佳线上 F1 为 0.73；保留的团队集成配置得分为 0.78260，另有团队最终成绩记录 0.78284；最终保存的生成模型 FID 为 134.20。

[项目详情（英文）](projects/meteorite-vision.html) ·
[代码与复现](https://github.com/0zz0ruby/meteorite-vision-portfolio) ·
[Kaggle Stage 2 比赛页面](https://www.kaggle.com/competitions/meteorite-identification-stage-2)

### 3. Hadoop MapReduce 限价订单簿因子计算

先实现并验证双作业计算流程，再针对运行耗时进行瓶颈分析，将流水线重构为单个 MapReduce 作业，减少任务调度与中间数据读写开销。

约 1,500 个股票快照文件覆盖五个交易日。优化后，20 个因子在五个交易日均通过平均相对误差不超过 1% 的验证，单作业平均处理、导出与核验耗时约 16.8 秒。

[项目详情（英文）](projects/lob-factor-mapreduce.html) ·
[代码与复现](https://github.com/0zz0ruby/lob-factor-mapreduce)

### 4. CartPole 强化学习与鲁棒性

个人工作包括 A2C / PPO 超参数优化，以及将方向与 11 级力度分离的 Force-Split Independent PPO 控制器设计与实现。

后续验证性实验使用 10 个训练种子和固定交互预算。A2C、PPO 均达到约 500 步上限，PPO 学习曲线 AUC 相对提升 15.2%；加入惩罚后，力平方控制成本代理指标降低 93.58%，但未能证明控制性能保持不变。

[项目详情（英文）](projects/cartpole-energy-aware-marl.html) ·
[代码与复现](https://github.com/0zz0ruby/cartpole-energy-aware-marl)

### 5. 信用卡客户分群

对 8,950 名客户进行数据清洗、PCA 降维、K-means 聚类、统计验证与蒙特卡洛模拟。

五个主成分保留 70.13% 的方差，综合内部指标与业务可解释性选择四类客户，使用 240 次模拟检验不同样本量和噪声水平下的表现。

[项目详情（英文）](projects/credit-card-customer-segmentation.html) ·
[代码与复现](https://github.com/0zz0ruby/credit-card-customer-segmentation)

## 联系

Gmail：[rubyli2525@gmail.com](mailto:rubyli2525@gmail.com)

邮箱：[lyclyc555@126.com](mailto:lyclyc555@126.com)
