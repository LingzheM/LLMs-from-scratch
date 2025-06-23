# Embedding: 离散符号如何”变成“稠密向量

> 结构: 终点 ➡️ 公理 ➡️ 连线 ➡️ 填补 ➡️ 验证

1. 终点

Embedding到底干什么?

把词表V中的每个离散token w映射成一个低维稠密向量(维度d), 使

1. 空间可度量: 语义相近 ➡️ 向量距离近
2. 维度可控: d <<|V|
3. 端到端可学:  

2. 公理

|编号|原理|说明|
|---|---|---|
|P1|离散 ➡️ 连续: 梯度法只能优化实值权重 => 需要把one-hot(离散)变成连续向量| 否则无法反向传播 |
|P2|分布假设: 上下文相似 => 向量相似|word2vec, 用这一假设学嵌入|
|P3|线性映射: 一组可学习矩阵||

3. 连线(缺口问题)

1. Q1, 如何从原始语料得到corpus_size和vocab_size ?
2. Q2, one-hot => Embedding时, 矩阵乘. 查表的维度怎么精确对应 ?
3. Q3, 训练时梯度怎么从Loss传到嵌入矩阵W ?

4. 填补: 手把手算到矩阵尺寸

示例mini语料
```ini
S1 = "I like deep learning"
S2 = "I like natural language processing"
```

4.1 统计corpus_size和vocab_size

|步骤|结果|
|---|---|
|拆词得到序列|["I", "like", "deep", "learning", "I", "like", "natural", "language", "processing"]|
|corpus_size| 9(总token数)|
|去重排序得到词表|["I", "deep", "language", "learning", "like", "natural", "processing"]|
|vocab_size|7|

4.2 one-hot表示

- 矩阵X: 形状(N, V) = (9, 7)
    每一行是一个token的独热向量; 第p行里只有一列为1, 其余为0

    1, 0, 0, 0, 0, 0, 0
    0, 0, 0, 0, 1, 0, 0
    0, 1, 0, 0, 0, 0, 0
    0, 0, 0, 1, 0, 0, 0
    1, 0, 0, 0, 0, 0, 0
    0, 0, 0, 0, 1, 0, 0
    0, 0, 0, 0, 0, 1, 0
    0, 0, 1, 0, 0, 0, 0
    0, 0, 0, 0, 0, 0, 1

4.3 Embedding权重矩阵

- 设目标向量维度d = 5
- 矩阵W: V x d = 7 * 5
    第j行就是词的可学习向量

4.4 稠密向量计算

    矩阵onehot(9, 7) * 矩阵W(7, 5) = 矩阵emb(9, 5)

- 每行乘法结果等价于“查表”拿到该token的嵌入.
- 时间复杂度: 理论O(Nd)

4.5 反向传播到W

- 假设后续层与Loss产生梯度
