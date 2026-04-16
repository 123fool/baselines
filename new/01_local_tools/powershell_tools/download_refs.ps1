$ErrorActionPreference = 'Continue'

function Ensure-Dir($p) {
  if (!(Test-Path $p)) {
    New-Item -ItemType Directory -Force -Path $p | Out-Null
  }
}

function Ensure-Clone($target, $url) {
  if (Test-Path $target) {
    try {
      git -C $target rev-parse --is-inside-work-tree | Out-Null
      return 'exists'
    } catch {
      Remove-Item -Recurse -Force $target -ErrorAction SilentlyContinue
    }
  }
  try {
    git clone --depth 1 $url $target | Out-Null
    return 'cloned'
  } catch {
    return ('failed: ' + $_.Exception.Message)
  }
}

function Download-Pdf($url, $outPath) {
  try {
    Invoke-WebRequest -Uri $url -OutFile $outPath -MaximumRedirection 10 -Headers @{ 'User-Agent' = 'Mozilla/5.0' } -TimeoutSec 240
    $len = (Get-Item $outPath).Length
    if ($len -lt 10000) {
      Remove-Item -Force $outPath -ErrorAction SilentlyContinue
      return 'failed: too small'
    }
    $bytes = [System.IO.File]::ReadAllBytes($outPath)
    if ($bytes.Length -lt 4 -or $bytes[0] -ne 37 -or $bytes[1] -ne 80 -or $bytes[2] -ne 68 -or $bytes[3] -ne 70) {
      Remove-Item -Force $outPath -ErrorAction SilentlyContinue
      return 'failed: not pdf'
    }
    return 'ok'
  } catch {
    if (Test-Path $outPath) { Remove-Item -Force $outPath -ErrorAction SilentlyContinue }
    return ('failed: ' + $_.Exception.Message)
  }
}

$workspace = 'C:\Users\PC\Desktop\baselines'
$base = (Get-ChildItem -Path $workspace -Directory | Where-Object {
  (Get-ChildItem -Path $_.FullName -Directory -ErrorAction SilentlyContinue | Measure-Object).Count -ge 5 -and
  (Get-ChildItem -Path $_.FullName -Recurse -Filter 'references.md' -ErrorAction SilentlyContinue | Measure-Object).Count -ge 5
} | Select-Object -First 1).FullName

if ([string]::IsNullOrWhiteSpace($base)) {
  throw 'Reference base folder not found under workspace.'
}
$dirs = Get-ChildItem -Path $base -Directory | Sort-Object Name
if ($dirs.Count -lt 5) {
  throw 'Expected 5 innovation directories under reference base path.'
}

$d1 = $dirs[0].FullName
$d2 = $dirs[1].FullName
$d3 = $dirs[2].FullName
$d4 = $dirs[3].FullName
$d5 = $dirs[4].FullName

$results = New-Object System.Collections.Generic.List[object]

$plan = @(
  @{
    folderPath = $d1
    pdfs = @(
      @{ name='AG-LDM_2026_arXiv2601.14584.pdf'; url='https://arxiv.org/pdf/2601.14584.pdf' },
      @{ name='Linguistic_Compass_2025_arXiv2506.05428.pdf'; url='https://arxiv.org/pdf/2506.05428.pdf' },
      @{ name='BrLP_2024_arXiv2405.03328.pdf'; url='https://arxiv.org/pdf/2405.03328.pdf' }
    )
    repos = @(
      @{ name='BrLP'; url='https://github.com/LemuelPuglisi/BrLP.git' },
      @{ name='TaDiff-Net'; url='https://github.com/samleoqh/TaDiff-Net.git' }
    )
  },
  @{
    folderPath = $d2
    pdfs = @(
      @{ name='TADM_MICCAI2024_arXiv2406.12411.pdf'; url='https://arxiv.org/pdf/2406.12411.pdf' },
      @{ name='MRExtrap_2025_arXiv2508.19482.pdf'; url='https://arxiv.org/pdf/2508.19482.pdf' }
    )
    repos = @(
      @{ name='TADM'; url='https://github.com/MattiaLitrico/TADM-Temporally-Aware-Diffusion-Model-for-Neurodegenerative-Progression-on-Brain-MRI.git' },
      @{ name='SADM'; url='https://github.com/ubc-tea/SADM-Longitudinal-Medical-Image-Generation.git' },
      @{ name='BrLP'; url='https://github.com/LemuelPuglisi/BrLP.git' }
    )
  },
  @{
    folderPath = $d3
    pdfs = @(
      @{ name='DiT_ICCV2023_arXiv2212.09748.pdf'; url='https://arxiv.org/pdf/2212.09748.pdf' },
      @{ name='AG-LDM_2026_arXiv2601.14584.pdf'; url='https://arxiv.org/pdf/2601.14584.pdf' },
      @{ name='Brain_Diffusion_Transformer_2025_bioRxiv.pdf'; url='https://www.biorxiv.org/content/10.1101/2025.04.12.648506v1.full.pdf' }
    )
    repos = @(
      @{ name='3D-MedDiffusion'; url='https://github.com/ShanghaiTech-IMPACT/3D-MedDiffusion.git' },
      @{ name='DiT'; url='https://github.com/facebookresearch/DiT.git' },
      @{ name='BrLP'; url='https://github.com/LemuelPuglisi/BrLP.git' }
    )
  },
  @{
    folderPath = $d4
    pdfs = @(
      @{ name='MedicalNet_2019_arXiv1904.00625.pdf'; url='https://arxiv.org/pdf/1904.00625.pdf' },
      @{ name='AG-LDM_2026_arXiv2601.14584.pdf'; url='https://arxiv.org/pdf/2601.14584.pdf' },
      @{ name='Perceptual_Loss_2016_arXiv1603.08155.pdf'; url='https://arxiv.org/pdf/1603.08155.pdf' }
    )
    repos = @(
      @{ name='3D-MedDiffusion'; url='https://github.com/ShanghaiTech-IMPACT/3D-MedDiffusion.git' },
      @{ name='MedicalNet'; url='https://github.com/Tencent/MedicalNet.git' },
      @{ name='BrLP'; url='https://github.com/LemuelPuglisi/BrLP.git' }
    )
  },
  @{
    folderPath = $d5
    pdfs = @(
      @{ name='AG-LDM_2026_arXiv2601.14584.pdf'; url='https://arxiv.org/pdf/2601.14584.pdf' },
      @{ name='Frontiers_MCI_AD_2025_10.3389-fneur.2025.1596632.pdf'; url='https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2025.1596632/pdf' },
      @{ name='AlzResTher_2025_10.1186-s13195-025-01827-2.pdf'; url='https://alzres.biomedcentral.com/counter/pdf/10.1186/s13195-025-01827-2.pdf' },
      @{ name='USB_2025_arXiv2512.00269.pdf'; url='https://arxiv.org/pdf/2512.00269.pdf' }
    )
    repos = @(
      @{ name='BrLP'; url='https://github.com/LemuelPuglisi/BrLP.git' },
      @{ name='3D-MedDiffusion'; url='https://github.com/ShanghaiTech-IMPACT/3D-MedDiffusion.git' }
    )
  }
)

foreach ($item in $plan) {
  $root = $item.folderPath
  $folderName = Split-Path $root -Leaf
  $pdfDir = Join-Path $root 'pdf'
  $codeDir = Join-Path $root 'code'
  Ensure-Dir $pdfDir
  Ensure-Dir $codeDir

  foreach ($p in $item.pdfs) {
    $target = Join-Path $pdfDir $p.name
    $status = Download-Pdf $p.url $target
    $results.Add([pscustomobject]@{ innovation=$folderName; type='pdf'; name=$p.name; source=$p.url; status=$status })
  }

  foreach ($r in $item.repos) {
    $target = Join-Path $codeDir $r.name
    $status = Ensure-Clone $target $r.url
    $results.Add([pscustomobject]@{ innovation=$folderName; type='code'; name=$r.name; source=$r.url; status=$status })
  }
}

$csv = Join-Path $base 'download_status.csv'
$md = Join-Path $base 'download_status.md'
$results | Export-Csv -Path $csv -NoTypeInformation -Encoding UTF8

$lines = @()
$lines += '# Download Status Summary'
$lines += ''
$lines += '| Innovation | Type | Name | Status | Source |'
$lines += '|---|---|---|---|---|'
foreach ($r in $results) {
  $lines += ('| {0} | {1} | {2} | {3} | {4} |' -f $r.innovation, $r.type, $r.name, $r.status, $r.source)
}
Set-Content -Path $md -Value $lines -Encoding UTF8

Write-Output ('DONE: ' + $csv)
Write-Output ('DONE: ' + $md)
