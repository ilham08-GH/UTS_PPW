param (
    [string]$message = ""
)

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "     SCRIPT UPDATE OTOMATIS UTS PPW (STREAMLIT)     " -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Cek perubahan di direktori git
$status = git status --porcelain
$unpushed = git log '@{u}..HEAD' --oneline 2>$null

if (-not $status -and -not $unpushed) {
    Write-Host "Tidak ada perubahan file baru maupun commit yang tertunda." -ForegroundColor Yellow
    Write-Host "Semua file sudah tersinkronisasi dengan GitHub." -ForegroundColor Gray
    Write-Host ""
    exit
}

if ($status) {
    # 2. Tampilkan file yang berubah
    Write-Host "File yang mengalami perubahan:" -ForegroundColor Yellow
    git status -s
    Write-Host ""

    # 3. Pesan commit
    if ([string]::IsNullOrWhiteSpace($message)) {
        $inputMsg = Read-Host "Masukkan pesan update (Kosongkan untuk pesan default)"
        if ([string]::IsNullOrWhiteSpace($inputMsg)) {
            $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm")
            $message = "update: pembaruan aplikasi streamlit uts ($timestamp)"
        } else {
            $message = $inputMsg
        }
    }

    Write-Host ""
    Write-Host "[1/3] Menambahkan perubahan (git add .)..." -ForegroundColor Green
    git add .

    Write-Host "[2/3] Menyimpan commit (git commit)..." -ForegroundColor Green
    git commit -m "$message"
} else {
    Write-Host "Menemukan commit lokal yang belum di-push ke GitHub." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[3/3] Mengunggah ke GitHub (git push origin main)..." -ForegroundColor Green
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host "      UPDATE BERHASIL DIUNGGAH KE GITHUB!           " -ForegroundColor Green
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Repositori GitHub Anda:" -ForegroundColor White
    Write-Host "https://github.com/ilham08-GH/UTS_PPW" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Gagal melakukan git push. Silakan periksa koneksi atau autentikasi GitHub Anda." -ForegroundColor Red
    Write-Host ""
}
