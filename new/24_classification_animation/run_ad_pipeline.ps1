# ============================================================
# 一键运行: 查找 AD 转化患者 + 运行 pipeline + 下载结果
# ============================================================
# 用法: 
#   cd C:\Users\PC\Desktop\baselines
#   .\BrLP-main\new\24_classification_animation\run_ad_pipeline.ps1
# ============================================================

$ErrorActionPreference = "Continue"
$SERVER = "wangchong@10.96.27.109"
$PORT = 2638
$REMOTE_DIR = "/home/wangchong/data/fwz/code/brlp_src/new/24_classification_animation"
$LOCAL_DIR = "C:\Users\PC\Desktop\baselines\BrLP-main\new\24_classification_animation"
$RESULTS_DIR = "$LOCAL_DIR\results_v3"
$OUTPUT_DIR = "/home/wangchong/data/fwz/output/classification_animation"

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Step 1: 上传脚本到服务器" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

scp -P $PORT "$LOCAL_DIR\find_and_run_ad.py" "${SERVER}:${REMOTE_DIR}/"
scp -P $PORT "$LOCAL_DIR\run_pipeline.py" "${SERVER}:${REMOTE_DIR}/"

Write-Host "`n" 
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Step 2: 先列出 AD 候选人" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

ssh -p $PORT $SERVER "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && cd /home/wangchong/data/fwz/code/brlp_src && python new/24_classification_animation/find_and_run_ad.py --list-only"

Write-Host "`n"
Write-Host "=" * 60 -ForegroundColor Cyan  
Write-Host "Step 3: 运行 pipeline (GPU 1)" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

ssh -p $PORT $SERVER "source /home/wangchong/miniconda3/etc/profile.d/conda.sh && conda activate fwz && cd /home/wangchong/data/fwz/code/brlp_src && nohup python new/24_classification_animation/find_and_run_ad.py --gpu 1 > ${OUTPUT_DIR}/ad_pipeline_log.txt 2>&1 & echo 'PID:' `$!`"

Write-Host "`nPipeline 已在后台启动!" -ForegroundColor Yellow
Write-Host "检查进度: ssh -p $PORT $SERVER `"tail -30 ${OUTPUT_DIR}/ad_pipeline_log.txt`"" -ForegroundColor Gray

Write-Host "`n"
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "后续步骤 (pipeline 完成后手动执行):" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host @"

# 检查进度:
ssh -p $PORT $SERVER "tail -30 ${OUTPUT_DIR}/ad_pipeline_log.txt"

# 下载结果 (替换 SUBJECT_ID 为实际患者 ID):
scp -P $PORT ${SERVER}:${OUTPUT_DIR}/*_summary.json "$RESULTS_DIR\"
scp -P $PORT ${SERVER}:${OUTPUT_DIR}/*_progression.gif "$RESULTS_DIR\"
scp -P $PORT ${SERVER}:${OUTPUT_DIR}/*_trajectory.png "$RESULTS_DIR\"

"@ -ForegroundColor White
