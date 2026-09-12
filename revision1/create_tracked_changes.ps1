param(
    [Parameter(Mandatory=$true)][string]$OriginalDocx,
    [Parameter(Mandatory=$true)][string]$RevisedDocx,
    [Parameter(Mandatory=$true)][string]$OutputDocx
)

$word = $null
$original = $null
$revised = $null
$comparison = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $original = $word.Documents.Open($OriginalDocx, $false, $true)
    $revised = $word.Documents.Open($RevisedDocx, $false, $true)

    # Create a new document containing Word's native insertions and deletions.
    $comparison = $word.CompareDocuments(
        $original, $revised, 2, 1,
        $true, $true, $true, $true, $true, $true, $true, $true,
        $true, $true, "Codex", $true
    )
    $comparison.SaveAs2($OutputDocx, 16)
}
finally {
    if ($comparison -ne $null) {
        $comparison.Close($false)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($comparison)
    }
    if ($revised -ne $null) {
        $revised.Close($false)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($revised)
    }
    if ($original -ne $null) {
        $original.Close($false)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($original)
    }
    if ($word -ne $null) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
