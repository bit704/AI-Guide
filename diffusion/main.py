"""
【54、Probabilistic Diffusion Model概率扩散模型理论与完整PyTorch代码详细解读】
 https://www.bilibili.com/video/BV1b541197HX/
 本代码根据视频编写并做部分修改，学习用
 公式可见 https://zhuanlan.zhihu.com/p/590840909
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve
import torch
import torch.nn as nn
import io
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1 准备数据集

s_curve, _ = make_s_curve(10 ** 4, noise=.1)
s_curve = s_curve[:, [0, 2]] / 10.
# print(s_curve.shape)
# (10000, 2)
dataset = torch.tensor(s_curve, device=device).float()

data = s_curve.T
fig, ax = plt.subplots()
ax.scatter(*data, color="red", edgecolor="white")
ax.axis("off")
# plt.show()

# 2 确定超参数的值
# 加噪和去噪使用的超参数

num_steps = 100

betas = torch.linspace(-6, 6, num_steps, device=device)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1], device=device).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == alphas_bar_sqrt.shape == one_minus_alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape


# print("shape: ", betas.shape)
# (100,)

# 3 确定扩散过程任意时刻的采样值
# 从原图x_0得到t时刻加噪后的图x_t
def q_x(x_0, t):
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alpha_1_m_t = one_minus_alphas_bar_sqrt[t]
    return alphas_t * x_0 + alpha_1_m_t * noise


# 4 演示原始数据加噪100步后的效果

num_shows = 20
fig, axs = plt.subplots(2, 10, figsize=(28, 3))
plt.rc("text", color="blue")

for i in range(num_shows):
    j = i // 10
    k = i % 10
    q_i = q_x(dataset, torch.tensor([i * num_steps // num_shows])).cpu()
    axs[j, k].scatter(q_i[:, 0], q_i[:, 1], color="red", edgecolor="white")
    axs[j, k].set_axis_off()
    axs[j, k].set_title("$q(\mathbf{x}_{" + str(i * num_steps // num_shows) + "})$")


# plt.show()


# 5 编写拟合逆扩散过程高斯分布的模型

class MLPDiffusion(nn.Module):

    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2),
            ]
        )

        # 对时刻t编码
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x_0, t):
        x = x_0
        for idx, embeddings_layer in enumerate(self.step_embeddings):
            t_embedding = embeddings_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x


# 6 编写训练的误差函数

def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0.shape[0]

    # 为每个样本选取一个随机时刻
    t = torch.randint(0, n_steps, size=(batch_size // 2,), device=device)
    t = torch.cat([t, n_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1)

    # x是x_0加噪后的结果，和q_x过程相同
    a = alphas_bar_sqrt[t]
    aml = one_minus_alphas_bar_sqrt[t]
    e = torch.randn_like(x_0)
    x = x_0 * a + e * aml

    output = model(x, t.squeeze(-1))

    # 计算预测噪声和实际噪声的MSE
    return (e - output).square().mean()


# 7 编写逆扩散采样函数（inference过程）

def p_sample_loop(model, shape, n_steps, betas, one_minus_alpha_bar_sqrt):
    cur_x = torch.randn(shape, device=device)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alpha_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    t = torch.tensor([t], device=device)

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t)
    mean = 1 / (1 - betas[t]).sqrt() * (x - (coeff * eps_theta))
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * torch.randn_like(x)

    return sample


# 8 训练

seed = 1234

print("Training...")

batch_size = 128
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_epoch = 4000
plt.rc("text", color="blue")

model = MLPDiffusion(num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for t in range(num_epoch):
    for idx, batch_x in enumerate(dataloader):
        loss = diffusion_loss_fn(model,
                                 batch_x,
                                 alphas_bar_sqrt,
                                 one_minus_alphas_bar_sqrt,
                                 num_steps)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    if t % 100 == 0:
        print(loss)
        x_seq = p_sample_loop(model,
                              dataset.shape,
                              num_steps,
                              betas,
                              one_minus_alphas_bar_sqrt)
        fig, axs = plt.subplots(1, 10, figsize=(28, 3))
        for i in range(1, 11):
            cur_x = x_seq[i * 10].detach().cpu()
            axs[i - 1].scatter(cur_x[:, 0], cur_x[:, 1], color="red", edgecolor="white")
            axs[i - 1].set_axis_off()
            axs[i - 1].set_title("$q(\mathbf{x}_{" + str(i * 10) + "})$")
        plt.savefig(f"./visual/png/{t}.png")
        print(f"output ./visual/png/{t}.png")
        plt.close()

# 9 动画演示扩散过程和逆扩散过程

imgs = []
for i in range(100):
    q_i = q_x(dataset, torch.tensor([i])).cpu()
    plt.figure(figsize=(10, 10))
    plt.scatter(q_i[:, 0], q_i[:, 1], color="red", edgecolors="white", s=5)
    plt.axis("off")

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close()
    img = Image.open(img_buf)
    imgs.append(img)

reverse = []
for i in range(100):
    cur_x = x_seq[i].detach().cpu()
    plt.figure(figsize=(10, 10))
    plt.scatter(cur_x[:, 0], cur_x[:, 1], color="red", edgecolors="white", s=5)
    plt.axis("off")

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close()
    img = Image.open(img_buf)
    imgs.append(img)

imgs += reverse
imgs[0].save("./visual/animation.gif", format="GIF", append_images=imgs, save_all=True, duration=20, loop=0)
print(f"output ./visual/animation.gif")
