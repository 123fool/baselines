# Innovation 4: Deploy to Server
# Uploads code + pretrained MedicalNet weights to server

$SERVER = "wangchong@10.96.27.109"
$PORT = 2638
$REMOTE_DIR = "/home/wangchong/data/fwz/code/innovation_4"
$LOCAL_DIR = "C:\Users\PC\Desktop\baselines\BrLP-main\new\07_innovation_4"
$MEDNET_WEIGHTS = "C:\Users\PC\Desktop\baselines\参考\创新点4_3D感知损失与频域约束\code\3D-MedDiffusion\warvito_MedicalNet-models_main\medicalnet\resnet_10_23dataset.pth"

Write-Host "=== Innovation 4: Deploying to server ==="

# Create remote directory structure
Write-Host "Creating remote directories..."
ssh -p $PORT $SERVER "mkdir -p $REMOTE_DIR/src $REMOTE_DIR/scripts $REMOTE_DIR/configs $REMOTE_DIR/dashboard $REMOTE_DIR/pretrained"

# Upload source files
Write-Host "Uploading source files..."
scp -P $PORT "$LOCAL_DIR\src\*.py" "${SERVER}:${REMOTE_DIR}/src/"
scp -P $PORT "$LOCAL_DIR\scripts\*.py" "${SERVER}:${REMOTE_DIR}/scripts/"
scp -P $PORT "$LOCAL_DIR\configs\*.yaml" "${SERVER}:${REMOTE_DIR}/configs/"
scp -P $PORT "$LOCAL_DIR\dashboard\app.py" "${SERVER}:${REMOTE_DIR}/dashboard/"
scp -P $PORT "$LOCAL_DIR\run.sh" "${SERVER}:${REMOTE_DIR}/"
scp -P $PORT "$LOCAL_DIR\README.md" "${SERVER}:${REMOTE_DIR}/"

# Upload MedicalNet pretrained weights
Write-Host "Uploading MedicalNet pretrained weights (~45MB)..."
scp -P $PORT "$MEDNET_WEIGHTS" "${SERVER}:${REMOTE_DIR}/pretrained/"

Write-Host "=== Deploy complete ==="
Write-Host "Next: SSH to server and run 'bash $REMOTE_DIR/run.sh train'"
