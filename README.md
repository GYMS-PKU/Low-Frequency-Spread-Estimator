# Low-Frequancy-Spread-Estimator

这是一个通过低频数据估计价差的估计器。

该项目的结构主要包括文献总结、数据处理、估计器构建和模型训练。

## 项目架构

### SpreadEstimator

SpreadEstimator是组织所有结构的类，在其中定义的方法包括：

- 给定算符计算所有股票的流动性估计值，并给出所有的效果评估统计量；
- 对于指定的算符集合，调用模型进行组合，并给出效果的评估统计量；

### dataloader

dataloader定义了所有的数据获取，包括从高频数据中加载真实的spread，以及生成结构化的股票低频数据；结构参考QBG

### estimator

estimator定义了所有的估计方法，包括AutoFormula中的自定义算符，以及文献中主要方法的实现，为此需要新增一些复杂操作符，例如Gibbs抽样器

### model

model定义了所有的组合方法，包括常见统计学习模型。