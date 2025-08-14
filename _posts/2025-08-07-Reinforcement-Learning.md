---
layout:     post
title:      Reinforcement Learning
subtitle:   Reinforcement learning
date:       2025-08-07
author:     Treaseven
header-img: img/bg2.jpg
catalog: true
tags:
    - reinforcement learning
---

强化学习的目标: 在当前状态下找到一个最优策略到达目标状态

# 马尔科夫决策过程
state、Action、State transition、Policy、Reward、Trajectories、returns、episodes <br>
马尔科夫链描述trajectory:<br>
s1 →(a1) s2 →(a2) s3 →(a3) s4 →(a4) ... → s9 <br>
returns: <br>
return = 0 + 0 + 0 + 1 = 1 <br>
即时收益: 当前状态执行某个动作马上会获得的收益 (奖惩值) <br>
未来收益: 当前状态执行某个动作后进入的新状态，新的状态之后的收益总和 <br>
s1 →(a1) s2 →(a2) s3 →(a3) s4 →(a4) ... → s9 s1状态执行动作a1后即时收益为0，未来收益是s2之后的收益 <br>
trajetory可能是无限长或者很长的，这样return会非常大，但未来收益相对于即时收益来说重要性要小一些，因此引入一个超参数γ作为衰减因子，γ∈(0, 1) <br>
引入衰减因子的return叫做disconted return <br>
discounted return = 0 + γ0 + γ^2*0 + γ^3*0 + γ^4*0 + ..... 当衰减因子越接近于1越关注未来收益，越接近于0越关注当前收益 <br>
State Space: 所有可能的状态，一个集合，可能无穷 <br>
Action Space: 依赖于状态，某个状态下所有的动作，也是一个集合，可能无穷 <br>
动作空间依赖于状态，每个状态下动作空间不一定相同 <br>

### 符号描述
集合
S: 状态空间 <br>
A(s): 状态S下的动作的空间 <br>
R(s, a): 状态S下执行动作a的奖励 <br>
状态转移概率
$$\sum_{s' \in S} p(s'|s, a) = 1 \text{ for any } (s, a)$$
奖励概率
$$\sum_{r \in R(s,a)} p(r|s, a) = 1 \text{ for any } (s, a)$$
策略: 在s状态下执行a的概率
$$\sum_{a \in A(s)} \pi(a|s) = 1 \text{ for any } s \in S$$
未来的状态和收益只依赖于当前状态和动作，与以前的动作和状态无关 <br>

价值函数 <br>
价值函数分为两种: 状态价值以及动作价值 <br>
状态价值: 从当前状态到目标状态的所有trajectory的return的期望 <br>
动作价值: 当前状态及动作之后的所有trajectory的return的期望 <br>
回顾returns: <br>
$G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k}$ <br>
价值函数的数学符号的展开表示:
$\begin{align}
V(s) &= \mathbb{E}[G_t | S_t = s] \\
&= \mathbb{E}[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots | S_t = s] \\
&= \mathbb{E}[R_t + \gamma(R_{t+1} + \gamma R_{t+2} + \cdots) | S_t = s] \\
&= \mathbb{E}[R_t + \gamma G_{t+1} | S_t = s] \\
&= \mathbb{E}[R_t + \gamma V(S_{t+1}) | S_t = s]
\end{align}$
价值函数的转移概率形式:
$V(s) = r(s) + \gamma \sum_{s' \in S} p(s'|s) V(s')$
价值函数的矩阵展开形式:
$$\begin{bmatrix}
V(s_1) \\
V(s_2) \\
\vdots \\
V(s_n)
\end{bmatrix} = \begin{bmatrix}
r(s_1) \\
r(s_2) \\
\vdots \\
r(s_n)
\end{bmatrix} + \gamma \begin{bmatrix}
P(s_1|s_1) & P(s_2|s_1) & \cdots & P(s_n|s_1) \\
P(s_1|s_2) & P(s_2|s_2) & \cdots & P(s_n|s_2) \\
\vdots & \vdots & \ddots & \vdots \\
P(s_1|s_n) & P(s_2|s_n) & \cdots & P(s_n|s_n)
\end{bmatrix} \begin{bmatrix}
V(s_1) \\
V(s_2) \\
\vdots \\
V(s_n)
\end{bmatrix}$$
* 贝尔曼公式不是唯一的，存在非常多种表现形式
* 贝尔曼公式不是针对单独状态的，每个状态都存在一个贝尔曼公式

状态价值函数: 基于某个策略的价值函数
$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$
动作价值函数: 在一个状态下执行动作a的期望价值函数
$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$
状态价值与动作价值之间的关系: 当前状态的状态价值是动作价值的期望 <br>
注意: 状态价值是一个期望，动作价值中当前收益是常数，后面的其他状态的状态价值是一个期望 <br>
$$V^\pi(s) = \sum_{a \in A} \pi(a|s) Q^\pi(s, a)$$
$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$
带策略的贝尔曼方程:
$$\begin{align}
V^\pi(s) &= \mathbb{E}_\pi[R_t + \gamma V^\pi(S_{t+1}) | S_t = s] \\
&= \sum_{a \in A} \pi(a|s) \left( r(s, a) + \gamma \sum_{s' \in S} p(s'|s, a) V^\pi(s') \right) \\
\\
Q^\pi(s, a) &= \mathbb{E}_\pi[R_t + \gamma Q^\pi(S_{t+1}, A_{t+1}) | S_t = s, A_t = a] \\
&= r(s, a) + \gamma \sum_{s' \in S} p(s'|s, a) \sum_{a' \in A} \pi(a'|s') Q^\pi(s', a')
\end{align}$$
注意
* 状态价值函数的作用是对当前策略进行评估，动作价值函数的作用可以看做对于当前状态执行某个动作的评估
* 强化学习的目标是找到一个最优策略，这个一个迭代过程，如何评估策略每次更好了一些，就通过价值函数来体现
* 在当前策略下根据贝尔曼公式求得所有状态的状态价值，根据状态价值来找到策略提升点从而更新一个新的策略，在这个新的策略下再次计算状态价值，不断迭代直到状态价值收敛
* 最终的最优策略是状态价值收敛，注意最优策略不是唯一的

计算贝尔曼公式: 值迭代和策略迭代 <br>
值迭代: 为每个状态初始化一个Value，一般均设置为0，迭代计算出每个状态的State Value，通过当前所有动作的即时收益获取到最优的Action Value并使用贝尔曼最优公式进行更新，循环迭代，直到每个状态的State Value收敛 <br>
策略迭代: 策略评估和策略提升 <br>
初始化一个策略和State Value，计算当前策略下所有策略最优State Value，贪心选择最优的Action Value，得到新的策略，重复执行，不断迭代
策略评估是计算一个策略的状态价值函数 <br>
策略提升是选择每个状态在当前策略下即时收益最大的Action Value <br>
策略提升是找到了策略提升点，即保证更新后的State Value不低于旧策略的State Value


# 蒙特卡洛与时序差分
Model free: 不知道模型转移矩阵的情况，分为蒙特卡洛方法、时序差分方法 <br>
两者思想是一致的: 贝尔曼公式是为了计算State Value，State Value本质上是一个对某一状态开始所有的discounted returns的期望，既然是model free，那么就按照黑盒处理，求解黑盒问题最本质的方法就是采样，只要采样足够，根据大数定理，采样到的数值的平均期望就是期望 <br>
基本思想: 没有环境模型就要有数据，没有数据就要有环境模型，两者都没有是没办法进行强化学习训练的 <br>

### 蒙特卡洛算法(Monte Carlo)
蒙特卡洛: 首先确定的是State Value是从某一个状态出发所有discounted returns或者returns的期望，在不知道环境模型的情况下进行采样
$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] \approx \frac{1}{N} \sum_{i=1}^{N} G_t^{(i)}$$
当采样足够多就具备从每个state执行每个Action的Action Value，可以进行策略的迭代更新，策略更新后重复的进行蒙特卡洛采样
* 针对采样方案，普遍采用的方案从一个状态出发进行十分长的episode采样，这样会出现多个相同的<s, a>, 对出现的次数和Value进行累加，最后求均值(这个方案叫every-visit,如果只记录第一次遇见的<s, a>则为first-visit)
* 对于MC采样时间过长的问题，可以使用增量式的更新方案
$$N(s) \leftarrow N(s) + 1$$
$$V(s) \leftarrow V(s) + \frac{1}{N(s)}(G - V(s))$$
* 针对策略更新每次贪心选择Action Value最大的<s, a>的情况，做出调整，普遍使用ε-Greedy,还有softmax等其他方案 <br>
Greedy公式
$$\pi_{k+1}(a|s) = \begin{cases}
1, & a = a_k^*(s) \\
0, & a \neq a_k^*(s)
\end{cases}$$
ε-Greedy公式
$$\pi(a|s) = \begin{cases}
1 - \frac{\epsilon}{|A(s)|}(|A(s)| - 1), & \text{for the greedy action} \\
\frac{\epsilon}{|A(s)|}, & \text{for the other } |A(s)| - 1 \text{ actions}
\end{cases}$$
ε-Greedy不再贪心的在策略更新时选择Action Value最大的Action，而是给了其他Action一定选择的概率，ε范围是[0, 1],当ε越大，策略收敛的越慢，ε平衡exploitation(利用)和exploration(探索) <br>
在对策略进行数据采样中尤其注重exploitation和exploration，利用是指已经知道一个action会带来较高reward时会尽可能的选择，探索是指如果只选择固定的几个action就不会知道除此之外其他action会不会有更高的reward，需要多尝试选择其他action，增加选择性
* 针对MC采样出来的数据是无偏的，但随着MDP的增长会做每一步都会叠加上方差，使得数据方差越来越大，解决方案是采样不走那么多步的数据，也就不是MC采样，使用时序差分算法或者多步时序差分算法

### 时序差分算法(temporal difference, TD)
时序差分算法本质上是蒙特卡洛的一种形式，上面增量式的MC算法中:
$$N(s) \leftarrow N(s) + 1$$
$$V(s) \leftarrow V(s) + \frac{1}{N(s)}(G - V(s))$$
对V(s)更换一种表达方式:
$$V(s_t) \leftarrow V(s_t) + \alpha[G_t - V(s_t)]$$
用α替换掉1/N(s)来做为一个超参数，G是一个discounted returns，其形式为:
$$G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k}$$
每次计算G需要计算很多步保证期望值无偏，但造成数据方差偏离严重，其实可以走1步，具体来说，时序差分算法用当前获得的奖励加上下一个状态的价值估计来作为在当前状态会获得的状态
$$V(s_t) \leftarrow V(s_t) + \alpha[r_t + \gamma V(s_{t+1}) - V(s_t)]$$
$r_t + \gamma V(s_{t+1}) - V(s_t)$为时序差分误差
之所以叫时序差分算法的原因就是使用t+1步的value来估计t步的value <br>
TD算法的详细公式
$$\underbrace{v_{t+1}(s_t)}_{\text{new estimate}} = \underbrace{v_t(s_t)}_{\text{current estimate}} - \alpha_t(s_t) \underbrace{[ v_t(s_t) - \underbrace{(r_{t+1} + \gamma v_t(s_{t+1}))}_{\text{TD target } \tilde{v}_t} ]}_{\text{TD error } \delta_t}$$
TD error反应了两个时间步之间的差异，迭代的目标是使得TD error为0 <br>
不管是MC算法还是TD算法，其作用都是用来estimate价值函数，包括状态价值(state value)和动作价值(action value),动作价值的估计实际上已经完成了策略迭代中的策略评估部分

#### Sarsa算法
$$\begin{align}
q_{t+1}(s_t, a_t) &= q_t(s_t, a_t) \\
&\quad - \alpha_t(s_t, a_t)[q_t(s_t, a_t) - (r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1}))], \\
\\
q_{t+1}(s, a) &= q_t(s, a), \quad \text{for all } (s, a) \neq (s_t, a_t).
\end{align}$$

原始的TD算法是预估State Value，根据当前的State Value可以使用值迭代或者计算出Action Value来进行策略更新，Sarsa则直接预估Action Value完成策略迭代中的策略评估阶段，再进行策略更新，MC既可以预估State Value也可以预估Action Value，以上都是针对贝尔曼公式开展迭代的

#### Q-learning算法
$$\begin{align}
q_{t+1}(s_t, a_t) &= q_t(s_t, a_t) \\
&\quad - \alpha_t(s_t, a_t)[q_t(s_t, a_t) - (r_{t+1} + \gamma \max_{a \in A(s_{t+1})} q_t(s_{t+1}, a))], \\
\\
q_{t+1}(s, a) &= q_t(s, a), \quad \text{for all } (s, a) \neq (s_t, a_t).
\end{align}$$

### on-policy and off-policy
上面介绍的所有model-free算法中，不管是蒙特卡洛还是时序差分中的Sarsa、Q-learning，其都是使用策略迭代方案，首先要进行策略更新，在策略评估阶段首先需要在当前策略下去生成数据，然后根据生成的数据来进行策略评估，最后进行策略更新，其流程主要有两个策略部分: 1.一个策略生成数据 2.根据生成的数据来进行策略评估，生成数据的策略称为行为策略，需要更新的策略称为目标策略。这两个策略如果是同一个则属于on-policy,这两个策略如果不是同一个则属于off-policy

# 基于价值/策略的深度强化学习算法
深度神经网络最大的优势是拟合数据分布，并对数据进行预测，可以使用神经网络来预测State Value和Action Value
### Value function
$$J(w) = \mathbb{E}[(v_\pi(S) - \hat{v}(S, w))^2]$$
Sarsa的深度神经网络形式
DQN-Deep-Q-learning算法则是直接在原始Q-Learning算法上进行优化，没有使用Deep Sarsa的目标函数方法来构建

### Policy Gradient
策略π的本质也是一个多分类函数，输出的是每个action在当前策略下的概率，LLM输出同样的是一个多分类函数，输出是next token,本质上来说LLM就是一种策略函数，可以看到当我们预训练或者监督微调时每个token作为动作的奖励都是1，这意味着我们无条件信任给定的专家数据来进行LLM的梯度更新，也可以说是策略更新。从这个观点来看，LLM监督微调或无监督预训练本质上都属于强化学习的一种，只不过奖励函数被隐藏掉了，实际在强化学习中这种基于专家数据进行offline且off policy的方案称为模仿学习或者行为克隆，一般可以快速提升策略能力，降低强化学习整体的训练时间



# Actor-Critic/TRPO/PPO/DPO
### Actor-Critic
$\begin{align}
\nabla_\theta J(\theta) &\propto \sum_{s \in S} \nu^\pi(s) \sum_{a \in A} Q^\pi(s, a) \nabla_\theta \pi(a|s) \\
&= \sum_{s \in S} \nu^\pi(s) \sum_{a \in A} \pi_\theta(a|s) Q^\pi(s, a) \frac{\nabla_\theta \pi(a|s)}{\pi_\theta(a|s)} \\
&= \mathbb{E}_\pi[Q^\pi(s, a) \nabla_\theta \log \pi_\theta(a|s)]
\end{align}$
Actor-Critic架构其实就是存在两个模型，Actor模型作为策略生成数据，Critic模型就作为策略评估的模型，策略评估的方法只能是Value，要么是Action Value，要么是State Value. 主观上理解Actor产生一系列<s, a>数据，Critic来评估Actor生成这些数据质量如何，也就是将Actor本身看做策略，Critic来评估Actor那些地方做得好，那些地方做的不好. <br>
另一个视角来看Actor-Critic架构，其实隐含这对抗博弈的思想，Actor模型产生数据，Critic模型评估数据，评估的本身是找到策略提升点，或者说找Actor的不足，有一些对抗的因素存在，Critic找到策略提升点会提升Actor的能力，同时会提升自身能力来继续寻找新策略下Actor的策略提升点，直到找不到策略提升点，那么Actor就已经到达最优策略. <br>
Actor模型的梯度:
$$\begin{align}
\theta_{t+1} &= \theta_t + \alpha \mathbb{E}[\nabla_\theta \ln \pi(A|S, \theta_t)|q_\pi(S, A) - v_\pi(S))] \\
&= \theta_t + \alpha \mathbb{E}[\nabla_\theta \ln \pi(A|S, \theta_t) \delta_\pi(S, A)] \\
\\
\delta_\pi(S, A) &= q_\pi(S, A) - v_\pi(S)
\end{align}$$
Critic模型的目标函数
$$\mathcal{L}(\omega) = \frac{1}{2}(r + \gamma V_\omega(s_{t+1}) - V_\omega(s_t))^2$$
其梯度为:
$$\nabla_\omega \mathcal{L}(\omega) = -(r + \gamma V_\omega(s_{t+1}) - V_\omega(s_t)) \nabla_\omega V_\omega(s_t)$$

### TRPO(Trust Region Policy Optimization) - 信任区域策略优化
在策略梯度和AC算法中是存在一个目标函数的
$$J(\theta) = \mathbb{E}_{s_0}[V^{\pi_\theta}(s_0)] = \mathbb{E}_{\pi_\theta}[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)]$$
目标是更新目标函数的参数梯度，从而找到最优的参数使得策略到达最优策略
$$\theta^* = \arg\max_\theta J(\theta)$$
采用梯度更新的算法，不断沿着目标函数方向去更新梯度，但分析可以知道梯度方向很容易走偏，一个重要原因就是前面提到的Critic模型预估不准确问题，很可能拟合到错误的<s, a>数据上从而偏离最优解，或者梯度增长过快很容易踏空 <br>
直观的理解，加入和LLM比较，LLM训练时如果学习率过大，很容易错误最优解，所以梯度增长过大也是同样道理，但不同的是LLM训练目标的解空间基本是可以认为不变或者变化很小，因为给定的数据都是训练正确的，但强化学习目标函数的解空间在每一次梯度更新时都会发生变化，且变化很大，原因在于每个策略下面的最优解都不相同，解空间一直在变化 <br>
TRPO的想法是考虑在每次梯度更新时有一个信任区域，在这个范围内的更新策略时可以保证梯度更新的安全性，因此叫做信任区域策略优化

### PPO(Proximal Policy Optimization Algorithms) - 近端策略优化
PPO将约束项KL散度与主目标函数进行合并，原因在于求解约束优化问题十分复杂，PPO还做了两项策略优化: 近端策略优化惩罚和近端策略优化裁剪，加上KL散度使得两个策略非常近<br>
#### 近端策略惩罚:
* Using several epochs of minibatch SGD, optimize the KL-penalized objective
$$L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)} \hat{A}_t - \beta \text{KL}[\pi_{\theta_{old}}(\cdot | s_t), \pi_\theta(\cdot | s_t)] \right]$$
* Compute $d = \hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_{old}}(\cdot | s_t), \pi_\theta(\cdot | s_t)]]$
    - If $d < d_{\text{targ}}/1.5$, $\beta \leftarrow \beta/2$
    - If $d > d_{\text{targ}} \times 1.5$, $\beta \leftarrow \beta \times 2$ <br>

当KL散度小于设置值的1.5倍时，KL约束因子缩减一倍，即新旧策略距离过近时，目标函数更注重策略提升点 <br>
当KL散度大于设置值的1.5倍时，KL约束因子增加一倍，即新旧策略距离过远时，增加KL散度约束，减少策略提升点带来的影响

#### 近端策略裁剪
TRPO主函数:
$$L^{CPI}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t \right] = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t \right]$$
策略裁剪后目标函数:
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$
裁剪的目的在于之前提到的重要性采样带来方差的问题，为了减弱重要性采样使得方差过大因此针对此项进行裁剪
裁剪后的目标函数展开后为:
$$\arg \max_{\pi_\theta} \mathbb{E}_{s \sim d^{\pi_k}, a \sim \pi_k(\cdot|s)} \left[ \min \left( \frac{\pi_\theta(a|s)}{\pi_k(a|s)} A^{\pi_k}(s, a), \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_k(a|s)}, 1-\epsilon, 1+\epsilon \right) A^{\pi_k}(s, a) \right) \right]$$

#### GAE(Generalized Advantage Estimation)-广义优势估计
TD算法时提到，一步TD会带来比较小的方差，但是数据本身的信息量很少，所以用于更新梯度的数据量就很少，甚至无法组成一个minibatch，同时虽然一步TD方差小，但由于当前Critic能力较差，会给出错误的Value，从而造成数据有偏，即偏差大，如何衡量偏差与方差，需要在一步TD与MC之间取值，也就是多步的TD算法


# DPO(Direct Preference Optimization)-直接偏好微调
DPO出发点是直接绕过奖励模型，最大程度简化ppo中的目标函数

