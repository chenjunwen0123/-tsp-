import numpy as np                 #数据分析
import config as conf              #数据
from ga import Ga                  #遗传算法
import matplotlib.pyplot as plt    #可视化

config = conf.get_config()


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


# 城市坐标
#city_pos_list = np.random.rand(config.city_num, config.pos_dimension)   随机生成
city_pos_list = np.array(            #自定义
[[0.48039628 ,0.23975031],
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
# 城市距离矩阵
city_dist_mat = build_dist_mat(city_pos_list)

print(city_pos_list)
print(city_dist_mat)

# 遗传算法运行
ga = Ga(city_dist_mat)
result_list, fitness_list = ga.train()
result = result_list[-1]  #最后一代的元素（最终结果）
result_pos_list = city_pos_list[result, :]

# 绘图
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
