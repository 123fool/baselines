# Innovation 4 + 5: Deploy Combined Experiment to Server

$SERVER = "wangchong@10.96.27.109"
$PORT = 2638
$REMOTE_DIR = "/home/wangchong/data/fwz/code/combined_4_5"
$LOCAL_DIR = "C:\Users\PC\Desktop\baselines\BrLP-main\new\09_combined_4_5"

Write-Host "=== 创新点4+5联合实验: 部署到服务器 ===" -ForegroundColor Cyan

# 创建远程目录
Write-Host "创建远程目录..."
ssh -p $PORT $SERVER "mkdir -p $REMOTE_DIR"

# 上传脚本
Write-Host "上传 run.sh..."
scp -P $PORT "$LOCAL_DIR\run.sh" "${SERVER}:${REMOTE_DIR}/"

Write-Host "上传 README.md..."
scp -P $PORT "$LOCAL_DIR\README.md" "${SERVER}:${REMOTE_DIR}/"

Write-Host "=== 部署完成 ===" -ForegroundColor Green
Write-Host "执行评估: ssh -p $PORT $SERVER 'bash $REMOTE_DIR/run.sh'"
