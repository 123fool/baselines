# Innovation 5: 一键部署脚本 (从 Windows 上传到服务器)
# 在本地 PowerShell 中运行此脚本

$SERVER = "wangchong@10.96.27.109"
$PORT = "2638"
$LOCAL_DIR = "C:\Users\PC\Desktop\baselines\BrLP-main\new\06_innovation_5"
$REMOTE_DIR = "/home/wangchong/data/fwz/code/innovation_5"

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Innovation 5 - 一键部署到服务器" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: 上传代码
Write-Host "[1/3] 创建服务器目录..." -ForegroundColor Yellow
ssh -p $PORT $SERVER "mkdir -p $REMOTE_DIR/{src,scripts,configs,dashboard}"

Write-Host "[2/3] 上传代码文件..." -ForegroundColor Yellow
scp -P $PORT -r "$LOCAL_DIR\src\*" "${SERVER}:$REMOTE_DIR/src/"
scp -P $PORT -r "$LOCAL_DIR\scripts\*" "${SERVER}:$REMOTE_DIR/scripts/"
scp -P $PORT -r "$LOCAL_DIR\configs\*" "${SERVER}:$REMOTE_DIR/configs/"
scp -P $PORT -r "$LOCAL_DIR\dashboard\*" "${SERVER}:$REMOTE_DIR/dashboard/"
scp -P $PORT "$LOCAL_DIR\run.sh" "${SERVER}:$REMOTE_DIR/run.sh"
scp -P $PORT "$LOCAL_DIR\README.md" "${SERVER}:$REMOTE_DIR/README.md"
scp -P $PORT "$LOCAL_DIR\changelog.json" "${SERVER}:$REMOTE_DIR/changelog.json"

Write-Host "[3/3] 设置执行权限..." -ForegroundColor Yellow
ssh -p $PORT $SERVER "chmod +x $REMOTE_DIR/run.sh"

Write-Host ""
Write-Host "====================================" -ForegroundColor Green
Write-Host "部署完成!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""
Write-Host "接下来在服务器上执行:" -ForegroundColor White
Write-Host "  ssh -p $PORT $SERVER" -ForegroundColor Gray
Write-Host "  conda activate fwz" -ForegroundColor Gray
Write-Host "  cd $REMOTE_DIR" -ForegroundColor Gray
Write-Host "  bash run.sh all" -ForegroundColor Gray
Write-Host ""
Write-Host "或分步执行:" -ForegroundColor White
Write-Host "  bash run.sh dashboard    # 仅启动监控面板" -ForegroundColor Gray
Write-Host "  bash run.sh ae           # 训练 AutoEncoder" -ForegroundColor Gray
Write-Host "  bash run.sh controlnet   # 训练 ControlNet" -ForegroundColor Gray
Write-Host "  bash run.sh evaluate     # 评估" -ForegroundColor Gray
