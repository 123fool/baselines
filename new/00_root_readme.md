# baselines/new 文件归档结构

本目录用于放置非仓库内的本地辅助文件（例如检查脚本、上传脚本、启动脚本）。

## 当前结构

- 01_local_tools/
  - python*helpers/：本地 Python 辅助脚本（如 \_check*_.py、*upload*_.py）
  - powershell_tools/：本地 PowerShell 脚本（如 \*.ps1）

## 后续放置规则

1. 本地临时脚本、排障脚本、上传脚本，统一放在本目录。
2. 新增 Python 辅助脚本放到 01_local_tools/python_helpers。
3. 新增 PowerShell 脚本放到 01_local_tools/powershell_tools。
4. 与 BrLP-main 项目内容直接相关的文档或代码，不放这里，放到 BrLP-main/new 对应主题目录。
