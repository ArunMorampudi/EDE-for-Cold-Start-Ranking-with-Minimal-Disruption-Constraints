$ErrorActionPreference='Stop'
$revisionRoot=(Get-Location).Path
$revisionWord=New-Object -ComObject Word.Application
try {
    $revisionWord.Visible=$false
    $revisionWord.DisplayAlerts=0
    $revisionOriginal=$revisionWord.Documents.Open((Join-Path $revisionRoot 'revision1/original_submitted/cs-134580-v0.4.docx'),$false,$true)
    $revisionOriginalText=$revisionOriginal.Content.Text
    $revisionOriginal.Close(0)
    $revisionClean=$revisionWord.Documents.Open((Join-Path $revisionRoot 'submitted_peerj_docs/cs-134580-v0.4.docx'),$false,$true)
    $revisionCleanText=$revisionClean.Content.Text
    $revisionClean.Close(0)
    $revisionTracked=$revisionWord.Documents.Open((Join-Path $revisionRoot 'submitted_peerj_docs/cs-134580-tracked-changes.docx'),$false,$true)
    $revisionCount=$revisionTracked.Revisions.Count
    $revisionTracked.AcceptAllRevisions()
    $revisionAcceptedText=$revisionTracked.Content.Text
    $revisionTracked.SaveAs2((Join-Path $revisionRoot 'revision1/qa/accepted.docx'),16)
    $revisionTracked.Close(0)
    $revisionTracked=$revisionWord.Documents.Open((Join-Path $revisionRoot 'submitted_peerj_docs/cs-134580-tracked-changes.docx'),$false,$true)
    $revisionTracked.RejectAllRevisions()
    $revisionRejectedText=$revisionTracked.Content.Text
    $revisionTracked.SaveAs2((Join-Path $revisionRoot 'revision1/qa/rejected.docx'),16)
    $revisionTracked.Close(0)
    $revisionResult=@{revisions=$revisionCount;accept_matches_clean=($revisionAcceptedText -eq $revisionCleanText);reject_matches_original=($revisionRejectedText -eq $revisionOriginalText)}
    $revisionResult | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $revisionRoot 'revision1/qa/redline_validation.json')
    $revisionResult | ConvertTo-Json
    if (-not $revisionResult.accept_matches_clean -or -not $revisionResult.reject_matches_original) { throw 'Tracked-change round-trip text mismatch' }
} finally { $revisionWord.Quit() }
