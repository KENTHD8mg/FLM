# 大语言模型技术基础之破晓

## 算法基础

### 机器学习算法

- 神经网络

	- CNN

		- 下采样、上采样

		- 卷积、反卷积

		- 池化、反池化

		- 感受野

		- 卷积计算

			-  

			- 输入图片大小为200×200，依次经过
一层卷积（kernel size 5×5，padding 1，stride 2），
pooling（kernel size 3×3，padding 0，stride 1），
又一层卷积（kernel size 3×3，padding 1，stride 1）之后，
输出特征图大小为( 97 )

			- 卷积神经网络的输入图像尺寸为28x28x3，卷积核大小为3x3x3，卷积核的个数为8，卷积步幅Stride=1，填充值padding=1，则输出的特征图尺寸为(28 宽、28 高、8 深)

	- RNN

		- LSTM

	- Unet

	- GAN

	- ResNet

	- 反向传播公式**

		-  

- SVM

- 决策树

- 随机森林

- Adaboost

- 朴素贝叶斯

### 超参寻找方式

- 网格搜索（Grid Search）：网格搜索是一种简单直观的方法，它通过指定超参数的候选值组成的网格来遍历所有可能的组合。对于每个组合，使用交叉验证来评估模型性能，并选择具有最佳性能的超参数组合。

- 随机搜索（Random Search）：与网格搜索相比，随机搜索不是遍历所有可能的组合，而是在给定的超参数空间中随机选择一组超参数进行评估。通过随机选择，可以更高效地探索超参数空间，并找到良好的超参数组合。

- 贝叶斯优化（Bayesian Optimization）：贝叶斯优化是一种基于贝叶斯推断的序列模型优化方法。它通过建立一个代理模型来估计目标函数（例如模型的性能），并使用贝叶斯推断来指导下一次选择哪个超参数组合进行评估。贝叶斯优化可以有效地在较少的迭代次数内找到最优的超参数组合。

- 交叉验证（Cross-Validation）：交叉验证是一种评估模型性能的技术，也可以用于超参数调整。通过将训练数据分成多个子集，然后在每个子集上轮流进行模型训练和验证，可以获得更稳健的性能评估结果。在超参数调整过程中，使用交叉验证来评估每个超参数组合的性能，并选择具有最佳性能的组合。

- 学习曲线（Learning Curve）：学习曲线是一种可视化工具，用于分析模型在不同超参数设置下的性能变化。通过绘制训练集大小和模型性能之间的关系，可以帮助判断模型是否过拟合或欠拟合，并指导超参数的选择。

- 自动化调参工具：还有一些自动化调参工具可用于帮助寻找最优的超参数组合，例如Hyperopt、Optuna和scikit-optimize等。这些工具结合了上述方法，并提供了更高级的优化算法和搜索策略，以加速超参数调整的过程。

### 损失函数

- 损失函数的可视化

	- 损失函数值的3D示意图

		- 横坐标为W，纵坐标为b，针对每一个w和一个b的组合计算出一个损失函数值，用三维图的高度来表示这个损失函数值。右图中的底部并非一个平面，而是一个有些下凹的曲面，只不过曲率较小

			-  

	- 损失函数值的2D示意图

		- 在平面地图中，我们经常会看到用等高线的方式来表示海拔高度值，右图就是上图在平面上的投影，即损失函数值的等高线图

			-  

### 结果评价

- 偏差(通常是训练集loss)和方差(通常是验证集方差)

	- 高偏差（欠拟合）

		- 1.更大的神经网络
2.更多的神经网络层数

	- 高方差（过拟合）

		- 1.更多训练数据
2.正则化

			- 正则化

				- 范数

					-  

						-  

						-  

						-  

						-  

					- 向量Lp范数

						- X表示n维向量

							- L1 范数：也称为曼哈顿范数（Manhattan Norm）。L1 范数在一些特定场景下具有稀疏性，能够产生稀疏解

							- L2 范数：也称为欧几里得范数（Euclidean Norm）。L2 范数在许多优化问题中都有很好的性质，例如它可以用于正则化、最小二乘问题等。

							- L∞ 范数：也称为切比雪夫范数（Chebyshev Norm），定义为向量中各个元素绝对值的最大值。L∞ 范数可以用于约束向量各个元素的最大值。

					- 矩阵范数

						- Lp元范数

							- A表示n行m列的矩阵

								- L2元范数称为弗罗贝尼乌斯范数(Frobenius norm)，或者希尔伯特-施密特范数(Hilbert-Schmidt norm)，常用于希尔伯特空间

								- L∞ 范数：

						- 诱导范数(由向量范数引导出来的范数)

				- L2正则化

					-  

						- tanh曲线

							-  

					- adam的weight_decay就是L2正则参数，adam把Loss加了L2惩罚后

			- dropout（随机失活）

				-  

			- early stopping

				- 提前终止可能是最简单的正则化方式，他适用于模型的表达能力很强的时候。这种情况下，一般训练误差会随着训练次数的增多逐渐下降，而测试误差则会先下降而后再次上升。我们需要做的就是在测试误差最低的点停止训练即可。

### 归一化输入

- 归一化类型

	- batch normalization

		- CV用BN比较多

			- CV使用BN是认为不同卷积核feature map（channel维）之间的差异性很重要，LN会损失channel的差异性，对于batch内的不同样本，同一卷积核提取特征的目的性是一致的，所以使用BN仅是为了进一步保证同一个卷积核在不同样本上提取特征的稳定性。

	- layer normalization

		- RMS normalization

		- NLP用LN比较多

			- NLP使用LN是认为batch内不同样本同一位置token之间的差异性更重要，而embedding维，网络对于不同token提取的特征目的性是一致的，使用LN是为了进一步保证在不同token上提取的稳定性。

	- instance normalization

	- group normalization

	- weight normalization

- 归一化能优化梯度算法加快网络收敛且有一定的防止过拟合的能力

	- 优化梯度算法

		-  

			-  

### 梯度消失和梯度爆炸

- 网络参数初始化

	- pytorch

		- nn.init.kaiming_normal_ kaiming初始化

		- nn.init.xavier_uniform_ xavier初始化

		- pytorch初始化方法分类

			- 针对饱和激活函数（sigmoid， tanh）：Xavier均匀分布， Xavier正态分布

			- 针对非饱和激活函数（relu及变种）：Kaiming均匀分布， Kaiming正态分布

			- 三个常用的分布初始化方法：均匀分布，正态分布，常数分布

			- 三个特殊的矩阵初始化方法：正交矩阵初始化，单位矩阵初始化，稀疏矩阵初始化

	- 好的初始化参数前向传播不会落在饱和区、方向传播不会梯度爆炸梯度消失

		- 若第 l 层使用relu激活函数，建议使用：

		- 若第 l 层使用tanh激活函数，建议使用：

### 梯度校验

### 国际顶会

- ACL
ACL成立于1962年，它是自然语言处理与计算语言学领域最高级别的学术会议，由计算语言学协会主办，每年一届。这个学会主办了NLP/CL领域最权威的国际会议，即ACL年会，ACL学会还会在北美和欧洲召开分年会，分别称为NAACL和EACL。
ACL是世界上影响力最大、最具活力的国际学术组织之一，它每年夏天都会召开大会，供学者发布论文，分享最新成果，它的会员来自全球60多个国家和地区，是NLP领域最高级别的国际学术组织，代表了国际计算语言学的最高水平。
在2018年第56届国际计算语言学协会（ACL）年会开幕式上，ACL的主席Marti Hearst宣布创建亚太区域分会（AACL，The Asia-Pacific Chapter of Association for Computational Linguistics），计划在2020年举行首次会议，此后每两年举行一次会议，会议地点将设置在亚太地区。

- EMNLP
EMNLP是由国际语言学会（ACL）下属的SIGDAT小组主办的自然语言处理领域的顶级国际会议，聚焦于自然语言算法在各个领域解决方案的学术探讨，EMNLP每年举办一次。
ACL的历史悠久，定位主要偏向于计算语言学理论相关研究，而EMNLP相对灵活一些，更加偏向于自然语言算法在不同领域解决方案的学术探讨。

- NAACL
NACAL是ACL的北美分会，每年举办一次。就目前而言，大家对它的认可度比EACL（欧洲分会）高。
ACL、EMNLP、NAACL 均为每年举办一次，因为是同一学术组织举办，所以会有些有意思的潜规则。例如ACL、EMNLP会在各大洲轮流举办，而每当ACL在北美举办时，当年NAACL就停办一次（同理，当ACL在欧洲举办时，当年EACL就停办一次）。

- COLING
COLING是由老牌NLP/CL学术组织ICCL（The International Committee on Computational Linguistics)）组织的，从1965年开始，每两年举办一次。

## 模型基础

### transformer

- transformer模型结构配置

	- name_or_path: 模型的名称或路径。

	- architectures: 模型所使用的架构。

	- attention_probs_dropout_prob: 注意力矩阵中的dropout概率。

	- bos_token_id: 开始标记的ID。

	- decoder_start_token_id: 解码器的起始标记ID。

	- dropout: Dropout的概率。

	- eos_token_id: 终止标记的ID。

	- hidden_act: 激活函数的类型，如gelu、relu等等。

	- hidden_dropout_prob: 隐藏层中的dropout概率。

	- hidden_size: 模型隐藏层的大小。

	- initializer_range: 参数初始化的范围。

	- intermediate_size: 每个隐藏层中间层的大小。

	- is_decoder: 是否为解码器。

	- is_encoder_decoder: 是否是编码器-解码器架构。

	- layer_norm_eps: LayerNorm层中epsilon的值。

	- n_head: 头部的数量。

	- n_layers: 模型中的总层数。

	- num_attention_heads: 每个隐藏层中的自注意头的数量。

	- num_hidden_layers: 模型的隐藏层数量。

	- pad_token_id: 填充标记的ID。

	- tie_word_embeddings: 是否将编码器和解码器的词嵌入层绑定。

	- tokenizer_class: 使用的分词器的类。

	- transformer_type: Transformer模型的类型。

	- transformers_version: Transformers库的版本号。

	- type_vocab_size: 类型词汇表的大小。

	- use_cache: 是否使用缓存。

	- vocab_size: 词汇表的大小。

- attention

- 位置编码

	- RoPE

	- AliBi

### chatgpt

### Llama

- ## 模型架构的详细解释
# LlamaForCausalLM类，这是一个用于因果语言建模的Llama模型
LlamaForCausalLM(
  # Llama模型的主体部分
  (model): LlamaModel(
    # 词嵌入层，将输入的单词（或词汇表中的标记）转换为固定大小的向量
    # 这里的嵌入维度是4096，词汇表大小是32000，padding_idx=0表示用于填充的标记的索引是0
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)

    # 模型层，由多个LlamaDecoderLayer层组成的列表
    # 这里使用了32个LlamaDecoderLayer层
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        # 自注意力机制模块，用于计算输入序列中各元素间的相互关系
        (self_attn): LlamaAttention(
          # 查询（Q）投影层，用于自注意力机制中的查询向量的线性变换
          (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          # 键（K）投影层，用于自注意力机制中的键向量的线性变换
          (k_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          # 值（V）投影层，用于自注意力机制中的值向量的线性变换
          (v_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          # 输出（O）投影层，用于自注意力机制计算完后的输出向量的线性变换
          (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          # 旋转位置编码，用于在自注意力机制中引入序列的位置信息
          (rotary_emb): LlamaRotaryEmbedding()
        )

        # 多层感知机模块，用于对自注意力机制的输出进行进一步的处理
        (mlp): LlamaMLP(
          # 门控投影层，用于MLP中的门控机制
          (gate_proj): Linear4bit(in_features=4096, out_features=11008, bias=False)
          # 上升投影层，用于MLP中的升维操作
          (up_proj): Linear4bit(in_features=4096, out_features=11008, bias=False)
          # 下降投影层，用于MLP中的降维操作
          (down_proj): Linear4bit(in_features=11008, out_features=4096, bias=False)
          # 激活函数，Swish-like activation（SiLU）
          (act_fn): SiLUActivation()
        )
        # 输入层归一化，对输入进行标准化处理
        (input_layernorm): LlamaRMSNorm()
        # 注意力后层归一化，对自注意力机制的输出进行标准化处理
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    # 最后的层归一化
    (norm): LlamaRMSNorm()
  )
  # 语言模型头部，将模型的输出转换回词汇表大小的向量，用于生成预测结果
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)

	- LlamaML层

		- 门控投影层gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

		- 上升投影层up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

		- 下降投影层down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

		- 激活函数act_fn默认是silu

		- 输出=down_proj(act_fn(gate_proj(x)) * up_proj(x))

- 分词器

	- tiktoken

	- sentencepiece

### 模型格式

- pytorch

- huggingface

- onnx

## 训练

### 数据

- 数据集

	- 开源数据集

		- Common Crawl

			- 400T

		- Colossal Clean Crawled Corpus (C4)

			- 14T

		- The Pile

			- 800G

		- 其他开源预训练语料库

			- openwebtext

			- oscar

			- refinedweb

		- llama2预训练数据集

			-  

- 数据清洗

	- 集群

		- 预处理

			- hadoop

		- 机器学习处理

			- Alink

			- spark

				- pyspark

					- spark.ml包

				- 为什么选择spark

					- spark跟hadoop完美结合

					- 原生机器学习包

					- 高稳定高吞吐高可用

	- 流程示例

		- 差数据剔除

			- HashingTF

				- 将一个序列转成n维特征向量

					- 将一个序列的每个token MurmurHash成一个32bit或128bit的哈希值，然后mod n获取余数k，在n维向量的第k维+1，这样获取一个n维向量

			- LogisticRegression

		- 相似剔除

			- HashingTF

			- MinHashLSH

				- Jaccard距离

					-  

				- 最小Hash

					- 将一个n维特征向量看成n个元素的集合，给这个向量计算一个签名

						- 函数 h对于集合 S的定义为：

						- 如果我们有 k个这样的哈希函数，我们可以得到一个签名 sig(S)，它是一个包含 k个元素的向量，每个元素都是通过一个不同的哈希函数得到的最小哈希值：

				- 局部敏感哈希

					- 简化最小Hash签名的Jaccard距离计算量

						- 如果有2个k维签名两两计算Jaccard距离，时间复杂度是k的平方，如果把k分成r等份，每一份为一个块，时间复杂的就是r方乘以(k/r)方，时间复杂的大大降低，在计算两个块是否相似时还可以只计算块内只要有一个元素相似就认为相似，进一步降低计算量

- 数据技术栈

	-  

### nvidia技术栈

- 训练知识

	- pytroch

		- Dataset

			- 负责数据加载和按照规定格式输出

				- map类型数据集需要实现__getitem__()和__len__()方法获取数据和调整格式写在__getitem__()里

				- Iterable数据集需要实现__iter__()方法，获取数据和调整格式写在__iter__()里

			- 在训练过程中数据集被划分成train训练集、eval评估集和predict预测集

		- DataLoader

			- 负责输出数据集
data_loader_train = DataLoader(data_train, batch_size=5, shuffle=False)
其中data_train是一个dataset，batch_size表示一批5条数据，shuffle是否将数据打乱，这个dataload表示enumerate循环中一次获取5条不用打乱的数据

		- nn.Module

			- 模型是pytorch的核心

				- nn.Module.train() 启动训练模式，比如开启dropout之类的训练特性
nn.Module.eval() 启动推理模式，比如关闭dropout之类的训练特性

		- nn.Dropout

			- 往往是Module里的一层，用于随机失活

		- 损失函数

			- 欧氏距离损失函数nn.MSELoss

				-  

			- 交叉熵损失函数nn.CrossEntropyLoss（内含logsoftmax操作），用于评估预测分类概率到目标分类的距离，Y是一个[0,0,....,1,....0]的分类标签

				-  

				- 交叉熵损失函数定义

					-  

		- 优化器

			- Adam、AdamW自适应梯度下降优化器

				- beta1、beta2、基础学习率lr、学习率调度器类型lr_scheduler_type、warmup_ratio预热率、warmup_steps预热步数、weight_decay 权值衰减系数

				- 这个结论来自于经典的AdamW论文Decoupled Weight Decay Regularization，原因不难从下图的看出（第6行的紫色部分，L2正则的导师就是λθ_{t-1}）：L2正则化作为一个辅助loss反映在梯度中，这个梯度不像SGD那样直接取负作为参数更新量，而是还要加上一阶动量 (第7行)，并且除以二阶动量 （第12行），即历史上梯度二范数的滑动平均值的平方根。二阶动量的目的就是实现自适应学习率，让历史上更新多的参数在后期学习率小一些，但当它和L2正则化结合在一起反而弄巧成拙，导致历史上更新多的参数受到的正则化惩罚更弱，L2正则化的作用被削弱，导致训出来的模型泛化性不如SGD+momentum训出的模型。
解决方案也很简单，就是把第6行的紫色部分去掉，也就是把L2正则化项从损失函数中去掉，直接在参数更新的时候用weight decay（第12行绿色部分），保证weight decay对所有参数“一视同仁”，不受Adam中二阶动量的影响。这种把优化器的更新量计算和weight decay解耦开的实现称为AdamW，已经成为各种Transformer模型训练的标配，例如Llama 2和Baichuan-2都使用了λ=0.1的AdamW。

			- SGD随机梯度下降优化器

		- 典型的训练示例看Lab项目的python的demo的train_model_FashionMNIST.py

	- transformers

		- Trainer

			- 分装了DataLoader、优化器，只要传入训练参数、Module、dataset、tokenizer，transformers的Module的forward里含有loss的计算和损失函数的调用

			- Trainer的dataset的数据格式需要跟模型的入参对应，因为在训练代码里直接用 model(一批dataset数据)做前向传播

	- trl

		- SFTTrainer

			- SFTTrainer是Trainer的一个集成了peft的轻量级分装

- 因果模型训练

	- pretrain

		- 因果模型的forward返回CausalLMOutputWithPast类包含    
loss: Optional[torch.FloatTensor] = None
logits: torch.FloatTensor = None
past_key_values:Optional[Tuple[Tuple[torch.FloatTensor]]] = None
hidden_states:Optional[Tuple[torch.FloatTensor, ...]] = None
attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

	- SFT

	- 高效微调

		- 库

			- PEFT

				- LoRA

					- 矩阵分解

						- 在华佗模型训练
## LoRA注意力维度
lora_r = 64  # LoRA的秩，这里设置为64

## LoRA缩放的Alpha参数
lora_alpha = 16  # LoRA的缩放参数，这里设置为16

## LoRA层的dropout概率
lora_dropout = 0.1  # 设置LoRA层的dropout概率为0.1

				- QLoRA

					- 权值量化+矩阵分解

				- Adapter Tuning

					- 在堆叠之间加一个小网络

				- Prefix Tuning

				- Prompt Tuning

				- P-tuning

				- P-tuning v2

					- 在华佗模型训练
#  然后创建peft配置，即创建ptuning微调方法的相关配置
 peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20, encoder_hidden_size=128)

	- RLHF

		- 库

			- trl

			- deepspeed-chat

			- ChatLLaMA

			- ColossalChat

		- 算法

			- PPO

			- DPPO

			- DPO

- 分布式训练

	- 开源社区主流

		- deepspeed

			- 分布式命令分发工具pdsh

			- master端口用于master跟分布式执行进程之间的通信

			- 通讯录可以查看pci-e端口、网卡、nvidialink

		- megatron

		- Pai-Megatron-Patch

	- ColossalAI

### 负载

- 吞吐

	- A100 80G 16卡训练13B模型

		- 500w token 600s 8.3ktoken/s

	- llama训练65B单卡380token/s

		- 在预训练一个包含65B参数的模型时，LLaMa的代码在具有80GB内存的2048个A100 GPU上每秒处理约380个token

- 负载剖析

	- https://zhuanlan.zhihu.com/p/629127302

## 推理

### nvidia技术栈

- TGI

- vllm

	- 单卡A100 80G 最大长度512 32路并行性能是1000token/s 如果回复平均长度是100token 那么但卡qps是10qps

- TensorRT

- Nvidia Triton

- BitsAndBytes

	- 预量化

		- GPTQ

		- GGUF

		- AWQ

## 应用

### agent

- 学习策略

- 配置策略

- 规划策略

- 行动策略

- 记忆策略

-  

### 应用技术栈

- langchain

- 规划

	- pddl

	- 辅助流程设计

		- BPMN、DMN、CMMN、Forms

- 爬虫

- 向量数据库

	- milvus

		- 邻居搜索ANN索引

			- HNSW

		- 量化搜索索引

			- QFLAT

	- faiss

	- 向量化方法

		- 静态向量化

			- token到向量的映射

				- word2vec、fasttext

		- 动态向量化

			- 联系上下文的词、片段、位置到向量的映射

				- ELMo、gpt、bert

- 固定格式输出

	- outline

### 小模型

- 数据孤岛（一汽chatdb、新人定义口径）

- 群体上下文（比如老师的教育、比如3c的苹果）

- 模型个性，比如两个妇产科大夫的对话

- 节能，目前很多论文显示垂直模型可以用很小的参数获得很好的性能

- 安全语境，比如美国和穆斯林的语境中对安全的定义是不同的

- LLM的Agent的广义知识很难在所有领域中都能最好地执行任务，还有一部分专家小模型（训练一个生成sd prompt的模型、训练一个自组织ppt的模型，CO-LLM[认为LLM擅长生成高级计划，但不擅长低级控制。他们使用启发式设计的低级计划器，根据高级计划稳健地执行基本操作）

## 调度

### 软件栈

- kubernetes

	- nvidia技术栈

		- nvidia-container-runtime

		- 驱动、cuda、cuDNN

		- NCCL

		- torch elastic

		- 华为开源火山调度器volcano

- slurm

	- 有很强的调度能力，可以直接用，自带nvidia虚拟能力

- determinedAI

### 硬件

- 卡

	- nvidia

		- Tesla架构

		- Fermi架构

		- Kepler架构

		- Maxwell架构

		- Pascal架构

		- Volta架构

		- Turing架构

		- Ampere架构

			- A100、A800

		- Hopper架构

			- H100、H800

		- Blackwell架构

			- B200

	- 昇腾

- 通信

	- 单机

		- 单机多卡

			- nvidia技术栈

				- GPUDirect P2P

				- NVLink

					- 提供一个端到端的高速单机内链路

				- NVSwitch

					- 提供一个单机内16卡全连接链路

	- 物理层通信

		- 多机多卡

			- nvidia技术栈

				- infiniBand 网卡

				- RoCEv2 网卡

				- NVSwitch

					- NVSwitch跨机直连

	- 链路层通信

		- 多机多卡

			- nvidia技术栈

				- NVlink交换机

					- NVlink交换机是包含NVSwitch芯片的交换机，跟主机NVSwitch接口直连

				- infiniBand交换机

					- 连主机infiniBand网卡，英伟达Quantum-X800、IB/Mellanox products家族

				- 标准以太网交换机、支持RoCEv2的交换机

					- Spectrum-X800

## kubernetes

### kuberFlow

### pod生命周期

- 调度scheduling

- 启动starting

- 运行running

	- 运行的时候有5种状态

		- 运行running

		- 挂起pending

		- 成功success

		- 失败failed

		- 未知unknow

- 成功退出Successful Termination

- 失败退出Failed Termination

	- 如果容器非正常退出（返回了非零退出状态），Pod会进入失败状态。失败的Pod可以被配置为重启，或者如果它是由控制器管理的（比如Deployment、StatefulSet等），控制器可能会尝试重启或替换它。

- 终止Termination

	- 当Pod需要被删除时，它会被发送终止信号，并且有一段时间（默认是30秒）来清理并优雅地退出。

- 删除Deletion

	- 一旦超时期限结束，如果Pod还没有退出，它会被强制终止，然后从Kubernetes集群中删除。

### OCI、CRI、容器运行时

