# 服务器连接信息

## 主力 GPU 服务器（wangchong）

| 字段   | 值             |
| ------ | -------------- |
| 主机   | `10.96.27.109` |
| 端口   | `2638`         |
| 用户名 | `wangchong`    |
| 密码   | `123456`       |

### 快速连接命令

```bash
ssh -p 2638 wangchong@10.96.27.109
```

### 常用路径

| 用途             | 路径                                                       |
| ---------------- | ---------------------------------------------------------- |
| 数据集根目录     | `/home/wangchong/data/fwz/brlp-data/`                      |
| 训练结果目录     | `/home/wangchong/data/fwz/brlp-train/`                     |
| 最新评估输出     | `/home/wangchong/data/fwz/brlp-train/eval_masked_20260403` |
| 代码目录（备用） | `/home/wangchong/fwz/brlp-code/`                           |

---

## 备用服务器（test 账号）

| 字段   | 值            |
| ------ | ------------- |
| 主机   | `10.123.2.11` |
| 端口   | `22`（默认）  |
| 用户名 | `test`        |
| 密码   | `Jsx@2010`    |

### 快速连接命令

```bash
ssh test@10.123.2.11
```

### 常用路径

| 用途        | 路径                                       |
| ----------- | ------------------------------------------ |
| 代码目录    | `/home/test/fwz/brlp-code/`                |
| 训练目录    | `/home/test/fwz/brlp-train/`               |
| Python 环境 | `/home/test/anaconda3/envs/fwz/bin/python` |

---

*最后更新：2026-04-03*
