# TSP solution based on Genetic Algorithm
原项目地址：https://github.com/zifeiyu0531/ga-tsp  
## TSP （Traveling Salesman Problem）
假设有一个旅行商人要拜访n个城市，他必须选择所要走的路径，路径的限制是每个城市只能拜访一次，而且最后要回到原来出发的城市。路径的选择目标是要求得的路径路程为所有路径之中的最小值。

## 代码结构
```
参数配置：config.py
遗传算法实现：ga.py
程序入口：main.py
```

## 环境
* 语言：`python3.7`
* 数据处理：`numpy`
* 数据可视化：`matplotlib`  

## 使用
1. clone到本地
2. 准备环境
3. 运行main.py

## 结果展示
**输入**：城市坐标矩阵  
![avatar](https://github.com/chenjunwen0123/ga-tsp/blob/main/img/city_input.png)  
**输出**：路线图 和 适应度曲线（适应度：路线总路程）  
![avatar](https://github.com/chenjunwen0123/ga-tsp/blob/main/img/demo1_route.png)  
![avatar](https://github.com/chenjunwen0123/ga-tsp/blob/main/img/demo1_fitness.png)

# 代码解释
## config.py
**主要参数 ：城市数量，坐标纬度，个体数量，迭代轮数，变异概率 **   
迭代轮数和城市数量（输入规模）是相关的，城市数量越多，迭代轮数相应也应该增大。但是也不能说迭代轮数是越多越好，从适应度曲线来看，在迭代了一定轮数后就会逐渐收敛。
``` python
data_arg = add_argument_group('Data')
data_arg.add_argument('--city_num', type=int, default=15, help='city num')                      # 城市数量
data_arg.add_argument('--pos_dimension', type=int, default=2, help='city num')                  # 坐标维度
data_arg.add_argument('--individual_num', type=int, default=60, help='individual num')          # 个体数
data_arg.add_argument('--gen_num', type=int, default=400, help='generation num')                # 迭代轮数
data_arg.add_argument('--mutate_prob', type=float, default=0.25, help='probability of mutate')  # 变异概率
```

## main.py
### 数据输入
首先输入城市坐标（或者随机生成），计算城市距离矩阵
```python
#随机生成城市的坐标
city_pos_list = np.random.rand(config.city_num, config.pos_dimension)
#指定城市的坐标
city_pos_list = np.array([[0.48039628 ,0.23975031],
 [0.37209854 ,0.07665969],
 [0.31869436 ,0.52966435],
 [0.46518153 ,0.85472255],
 [0.05875489 ,0.33020598],
 [0.60158229 ,0.48950949],
 [0.77726785 ,0.343438  ],
 [0.89189258 ,0.3528553 ],
 [0.27161261 ,0.19631259],
 [0.5035966  ,0.30218683],
 [0.56980736 ,0.32381325],
 [0.4029206  ,0.62429481],
 [0.54439459 ,0.510264  ],
 [0.5827445  ,0.05827028],
 [0.78226764 ,0.42318059]])
```
并通过调用自定义的函数`build_dist_mat`生成城市的距离矩阵（n*n矩阵，n是城市数量）
元素`[i ,j]  = [j ,i] `表示 i 城市到 j 城市的距离
```python
def build_dist_mat(input_list):
    n = config.city_num
    dist_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1, n):
            d = input_list[i, :] - input_list[j, :]
            # 计算点积
            dist_mat[i, j] = np.dot(d, d)
            dist_mat[j, i] = dist_mat[i, j]
    return dist_mat
```

### 遗传算法运行
`result_list` 和 `fitness_list` 对应 最终路线 和适应度数据
将城市距离矩阵传入种群对象`ga`中，调用种群类`Ga`中的`train()`函数进行遗传算法计算。并记录最后一代的结果。
```python
ga = Ga(city_dist_mat)
result_list, fitness_list = ga.train()
result = result_list[-1]  #最后一代的元素（最终结果）
result_pos_list = city_pos_list[result, :]
```
### 绘图
调用matplotlib库的函数来绘制折线图   
```python
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

#fig = plt.figure()
plt.figure()
plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r')
plt.title(u"路线")
plt.legend()
plt.show()

plt.figure()
plt.plot(fitness_list)
plt.title(u"适应度曲线")
plt.legend()
plt.show()
```

## ga.py
主要有两个类 `Individual` 和 `Ga`，分别对应个体和种群。

### Individual 类
一个个体对象 代表一条路线。
个体适应度(fitness)：该路线的路程长度
```python
class Individual:
    def __init__(self, genes=None):
        # 输入gene生成，gene是None则随机生成序列
        if genes is None:
            genes = [i for i in range(gene_len)]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        # 计算个体适应度
        fitness = 0.0
        for i in range(gene_len - 1):
            # 起始城市和目标城市
            from_idx = self.genes[i]
            to_idx = self.genes[i + 1]
            fitness += city_dist_mat[from_idx, to_idx]
        # 连接首尾
        fitness += city_dist_mat[self.genes[-1], self.genes[0]]
        return fitness
```
为什么要首尾相连?因为TSP问题规定最后要回到最初的起点。所以最后还得加上首尾城市相连的路程长度。

### Ga 类
遗传算法的过程定义在种群类 Ga 中：
```python
class Ga:
    def __init__(self, input_):     #input_ :城市距离矩阵
        global city_dist_mat
        city_dist_mat = input_
        self.best = None            # 每一代的最佳个体
        self.individual_list = []   # 每一代的个体列表
        self.result_list = []       # 每一代对应的解
        self.fitness_list = []      # 每一代对应的适应度
    def cross(self)                 #交叉
    def mutate(self,new_gen)        #变异
    def select(self)                 #选择
```

#### train()
已知在`config.py`中配置的个体数量`Individual_num`，迭代轮数`gen_num`  
1. 初始化种群（随机生成个体对象）  
2. 迭代开始  
  （1）对种群调用自定义的迭代函数 `next_gen()`  
  （2）将迭代后的最优个体选出来，将其基因序列首尾相连  
genes = [1,2,...,n]   →  genes = [1,2,...,n,1]  
  （3）将当前子代的最优个体加入(append)到种群的`result_list`和`fitness_list`中等待数据分析。  
  
```python
def train(self):
        # 初代种群
        self.individual_list = [Individual() for _ in range(individual_num)]  #初始化种群
        self.best = self.individual_list[0]
        # 迭代
        for i in range(gen_num):
            self.next_gen()   #子一代
            # 连接首尾
            result = self.best.genes   #将最优个体的基因序列 放到 result
            result.append(result[0])
            
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
        return self.result_list, self.fitness_list
```

#### next_gen()
1. **交叉(cross) → 变异(mutate) → 选择(select) **  
(遗传算法中心思想：**优胜劣汰**）  
2. **记录此代最优个体**
```python
def next_gen(self):
    # 交叉
    new_gen = self.cross()
    # 变异
    self.mutate(new_gen)
    # 选择
    self.select()
    # 获得这一代的结果
    for individual in self.individual_list:
        if individual.fitness < self.best.fitness:
            self.best = individual
```

#### 交叉 cross
1. 打乱种群个体，以步长为2遍历抽样（保证随机性）  
2. 利用深拷贝取样，genes1和gene2分别拷贝父样本和母样本的基因序列（路线）  
3. 选取基因片段[index1,index2]，并将初始基因片段中基因和其对应的位置以value:id 键值对形式分别存入字典pos1_recorder和 pos2_recorder  
4. 将这两段基因进行交换  
genes1:[1,**2,3,4**,5]  
genes2:[1,**4,5,3**,2]  

**交换方式: **   
找到一个待交换的基因，如`genes1[1] `和 `genes2[1]`  
若genes1在所选取的基因片段（红色部分）有`genes2[1]`的话，（可以看到`genes1[3]` = `genes2[1]` = 4），直接在`genes1`的基因片段中将这两个基因交换。  
genes1:[1,**4,3,2**,5]
  
将交叉后得到的两个新个体对象存入`new_gen`数组中  
```python
    def cross(self):   #交叉
        new_gen = []
        random.shuffle(self.individual_list)  #打乱每一代的种群个体
        for i in range(0, individual_num - 1, 2):   #步长为2 抽样
            # 父代基因
            genes1 = copy_list(self.individual_list[i].genes)      #父亲
            genes2 = copy_list(self.individual_list[i + 1].genes)  #母亲
            index1 = random.randint(0, gene_len - 2)            #基因片段
            index2 = random.randint(index1, gene_len - 1)       #保证 index2 > index1
            pos1_recorder = {value: idx for idx, value in enumerate(genes1)}
            pos2_recorder = {value: idx for idx, value in enumerate(genes2)}
            # 交叉
            for j in range(index1, index2):
                value1, value2 = genes1[j], genes2[j]
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                genes1[j], genes1[pos1] = genes1[pos1], genes1[j]
                genes2[j], genes2[pos2] = genes2[pos2], genes2[j]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2
            new_gen.append(Individual(genes1))  #新个体1
            new_gen.append(Individual(genes2))  
        return new_gen
```
交叉后的新一代进行变异

#### 变异 mutate
1. 像交叉一样，先选取基因片段 `genes[index1...index2]`
2. 将所选取的基因片段进行翻转(调用数组翻转 `reverse`)
**genes:[1,2,3,4,5]   →  genes:[1,4,3,2,5]**
3. 将变异后的个体添加到种群中
```python
def mutate(self, new_gen):   #变异
        for individual in new_gen:
            if random.random() < mutate_prob:   #config的 变异概率
                # 翻转切片
                old_genes = copy_list(individual.genes)
                index1 = random.randint(0, gene_len - 2)      #起始位置
                index2 = random.randint(index1, gene_len - 1) #终止位置
                genes_mutate = old_genes[index1:index2]
                genes_mutate.reverse()      #切片翻转
               individual.genes = old_genes[:index1] + genes_mutate + old_genes[index2:]
        # 两代合并
        self.individual_list += new_gen
```

#### 选择 select
在交叉变异后合并的两代个体中，选取`config.py`中配置好的个体数量`individual_num`个个体  
选择有很多种算法可以选择，如轮盘赌算法，锦标赛算法。  
在这里选择用锦标赛算法，即将所有个体分组，规定每个小组的出线个数。  
用锦标赛算法，一定程度上保留了相对来说较差的个体，保持了染色体多样性。（进化规划）
```python
def select(self): 
        # 锦标赛算法  防止局部最优  保证物种多样性
        group_num = 10  # 小组数
        group_size = 10  # 每小组人数
        group_winner = individual_num // group_num  # 每小组获胜人数
        winners = []  # 锦标赛结果
        for i in range(group_num):
            group = []
            for j in range(group_size):
                # 随机组成小组
                player = random.choice(self.individual_list)
                player = Individual(player.genes)
                group.append(player)
            group = Ga.rank(group)
            # 取出获胜者
            winners += group[:group_winner]
        self.individual_list = winners
```
