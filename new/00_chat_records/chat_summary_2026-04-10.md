# 聊天纪要（2026-04-10）

## 说明
本文件用于将本阶段与 Copilot 的沟通结果沉淀到仓库，便于后续复现与协作。

## 关键结论
1. 已确认 `yuxunlian` 目录中的文件是 BrLP 作者提供的可用预训练资产：
   - `autoencoder.pth`：自编码器权重
   - `latentdiffusion.pth`：扩散 UNet 权重
   - `controlnet.pth`：ControlNet 权重
   - `dcm-aux/*.json`：Leaspy 辅助进展模型（CN/MCI/AD）

2. 这些资产可直接用于 BrLP 推理流程（CLI 配置中分别对应 autoencoder/unet/controlnet/aux）。

3. 关于“修改模型后是否需要重训”的结论：
   - 只改推理参数：通常不需要重训。
   - 只改辅助轨迹模型：只需重训 auxiliary（dcm-aux）。
   - 改 ControlNet 相关逻辑：通常重训 ControlNet。
   - 改 UNet 结构/条件维度：通常重训 UNet + ControlNet。
   - 改 Autoencoder：通常需要从 AE 开始，连同后续 latent 提取、UNet、ControlNet 一并重跑。

## 研究组织产出（先前会话）
1. 在 `new/` 下已形成多份研究文档（基线评估、文献综述、analysis 评审、路线图等）。
2. 在 `参考/` 下按 5 个创新点建立了文献与代码索引目录。

## 下一步建议
1. 先确定本轮优先创新点（建议从风险最低的损失函数/区域加权类改动入手）。
2. 根据改动类型应用最小重训策略，节省训练成本。
3. 每次实验固定一个变量，记录配置、权重版本与指标，便于回溯。
