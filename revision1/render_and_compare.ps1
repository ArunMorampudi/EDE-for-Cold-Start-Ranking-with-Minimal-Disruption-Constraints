param([switch]$SkipCompare)
$ErrorActionPreference='Stop'
$revisionRoot=(Get-Location).Path
$revisionOutput=Join-Path $revisionRoot 'submitted_peerj_docs'
$revisionQa=Join-Path $revisionRoot 'revision1/qa/word'
New-Item -ItemType Directory -Force -Path $revisionQa | Out-Null
$revisionWord=New-Object -ComObject Word.Application
try {
    $revisionWord.Visible=$false
    $revisionWord.DisplayAlerts=0
    if (-not $SkipCompare) {
        $revisionOriginal=$revisionWord.Documents.Open((Join-Path $revisionRoot 'revision1/original_submitted/cs-134580-v0.4.docx'),$false,$true)
        $revisionClean=$revisionWord.Documents.Open((Join-Path $revisionOutput 'cs-134580-v0.4.docx'),$false,$true)
        $revisionTracked=$revisionWord.CompareDocuments($revisionOriginal,$revisionClean,2,1,$false,$true,$true,$true,$true,$true,$true,$true,$false,$false,'Revision One',$true)
        $revisionTracked.SaveAs2((Join-Path $revisionOutput 'cs-134580-tracked-changes.docx'),16)
        Write-Output ('Tracked changes: '+$revisionTracked.Revisions.Count)
        $revisionTracked.Close(0)
        $revisionOriginal.Close(0)
        $revisionClean.Close(0)
    }
    $revisionRecords=@()
    foreach ($revisionFile in Get-ChildItem -LiteralPath $revisionOutput -Filter '*.docx') {
        $revisionDoc=$revisionWord.Documents.Open($revisionFile.FullName,$false,$true)
        try {
            $revisionDoc.Repaginate()
            $revisionPdf=Join-Path $revisionQa ($revisionFile.BaseName+'.pdf')
            $revisionItem=0
            if ($revisionFile.Name -like '*tracked-changes*') { $revisionItem=7 }
            $revisionDoc.ExportAsFixedFormat($revisionPdf,17,$false,0,0,1,1,$revisionItem,$true,$true,1,$true,$true,$false)
            $revisionRecords+=@{file=$revisionFile.Name;pages=$revisionDoc.ComputeStatistics(2);revisions=$revisionDoc.Revisions.Count}
            Write-Output ($revisionFile.Name+': '+$revisionDoc.ComputeStatistics(2)+' pages')
        } finally { $revisionDoc.Close(0) }
    }
    $revisionRecords | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $revisionQa 'render_manifest.json')
} finally {
    $revisionWord.Quit()
}
