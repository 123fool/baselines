# 部署脚本: 将 combined_retrain 代码上传到服务器

$SERVER = "wangchong@10.96.27.109"
$PORT = 2638
$REMOTE_DIR = "/home/wangchong/data/fwz/code/combined_retrain"
$LOCAL_DIR = "C:\Users\PC\Desktop\baselines\BrLP-main\new\10_combined_retrain"

Write-Host "=== 创新点 4+5 联合重训 (方案 A): 部署到服务器 ===" -ForegroundColor Cyan

# 创建远程目录
ssh -p $PORT $SERVER "mkdir -p $REMOTE_DIR"

# 上传脚本
foreach ($f in @("train.sh", "eval.sh", "test_compatibility.sh", "README.md")) {
    Write-Host "上传 $f..."
    python -c "
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('10.96.27.109', port=2638, username='wangchong', password='123456')
sftp = c.open_sftp()
sftp.put(r'$LOCAL_DIR\$f', '$REMOTE_DIR/$f')
sftp.chmod('$REMOTE_DIR/$f', 0o755)
print('  OK')
sftp.close(); c.close()
"
}

Write-Host "=== 部署完成 ===" -ForegroundColor Green
