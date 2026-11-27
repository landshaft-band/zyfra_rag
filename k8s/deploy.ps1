param(
    [string]$Context = 'kind-rag',
    [ValidateSet('All', 'Infra', 'App')]
    [string]$Stage = 'All',
    [string]$ModelEnvFile,
    [string]$CorpusDirectory = (Join-Path $PSScriptRoot '../data/processed')
)

$ErrorActionPreference = 'Stop'

function Invoke-RagKubectl {
    & kubectl --context $Context @args
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl $($args[0]) failed with exit code $LASTEXITCODE"
    }
}

function Set-RagSecret([string]$Name, [hashtable]$Values) {
    $encoded = @{}
    foreach ($entry in $Values.GetEnumerator()) {
        $encoded[$entry.Key] = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($entry.Value))
    }
    @{
        apiVersion = 'v1'
        kind = 'Secret'
        metadata = @{name = $Name; namespace = 'rag'}
        type = 'Opaque'
        data = $encoded
    } | ConvertTo-Json -Depth 5 -Compress |
        & kubectl --context $Context apply --server-side -f -
    if ($LASTEXITCODE -ne 0) { throw "Could not apply secret $Name" }
}

function Set-RagModel([string]$Path) {
    $values = @{}
    foreach ($line in (Get-Content -LiteralPath $Path -Encoding utf8)) {
        if ($line -notmatch '^\s*(?:export\s+)?((?:RAG_LLM_|RAG_RERANK_|LANGFUSE_)[A-Z0-9_]+)\s*=\s*(.*)$') {
            continue
        }
        $key = $Matches[1]
        $value = $Matches[2].Trim()
        if ($value -match '^([''"])(.*)\1$') { $value = $Matches[2] }
        $values[$key] = $value
    }
    if (-not $values.Count) { throw 'No model or observability variables found in the env file' }
    Set-RagSecret 'rag-model' $values
}

function Wait-RagJob([string]$Name, [int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        $job = Invoke-RagKubectl -n rag get job $Name -o json | ConvertFrom-Json
        $conditions = @($job.status.conditions | Where-Object { $_.status -eq 'True' })
        if ('Failed' -in $conditions.type) {
            throw "Job $Name failed. See kubectl --context $Context -n rag logs job/$Name."
        }
        if ('Complete' -in $conditions.type) {
            "job/$Name completed"
            return
        }
        Start-Sleep -Seconds 2
    }
    throw "Timed out waiting for job/$Name. See kubectl --context $Context -n rag logs job/$Name."
}

if ($Stage -ne 'App') {
    $ticketSources = Get-Item -LiteralPath (Join-Path $CorpusDirectory 'ticket_sources.jsonl')
    $documentSources = Get-Item -LiteralPath (Join-Path $CorpusDirectory 'document_sources.jsonl')
    if (($ticketSources.Length + $documentSources.Length) -gt 1MB) {
        throw 'The corpus exceeds the Kubernetes ConfigMap limit of 1 MiB'
    }
    Invoke-RagKubectl apply -f (Join-Path $PSScriptRoot 'infra/namespace.yaml')
    if (-not (Invoke-RagKubectl -n rag get secret rag-postgres --ignore-not-found -o name)) {
        $random = New-Object byte[] 24
        $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
        try { $rng.GetBytes($random) } finally { $rng.Dispose() }
        $password = [Convert]::ToBase64String($random)
        Set-RagSecret 'rag-postgres' @{password = $password}
    }
    $corpus = Invoke-RagKubectl -n rag create configmap rag-corpus --dry-run=client -o json `
        "--from-file=ticket_sources.jsonl=$($ticketSources.FullName)" `
        "--from-file=document_sources.jsonl=$($documentSources.FullName)"
    $corpus | & kubectl --context $Context apply --server-side -f -
    if ($LASTEXITCODE -ne 0) { throw 'Could not apply corpus ConfigMap' }
    Invoke-RagKubectl apply -k (Join-Path $PSScriptRoot 'infra')
    foreach ($service in @('postgres', 'redis', 'kafka', 'qdrant')) {
        Invoke-RagKubectl -n rag rollout status "statefulset/$service" --timeout=240s
    }
}

if ($ModelEnvFile) { Set-RagModel $ModelEnvFile }
if ($Stage -eq 'Infra') { return }

$existing = Invoke-RagKubectl -n rag get deployment --ignore-not-found -o name
foreach ($job in @('migrate', 'queue-init', 'index-init')) {
    $active = Invoke-RagKubectl -n rag get job $job --ignore-not-found -o 'jsonpath={.status.active}'
    if ($active -and [int]$active -gt 0) {
        throw "Bootstrap job $job is already active; wait for it before redeploying"
    }
}
Invoke-RagKubectl -n rag delete job migrate queue-init index-init --ignore-not-found
Invoke-RagKubectl apply -k (Join-Path $PSScriptRoot 'bootstrap')
Wait-RagJob 'migrate' 240
Wait-RagJob 'queue-init' 240
Wait-RagJob 'index-init' 1800
Invoke-RagKubectl apply -k (Join-Path $PSScriptRoot 'app')
if ($existing) {
    Invoke-RagKubectl -n rag rollout restart deployment/api deployment/worker deployment/publisher deployment/dispatcher
}
foreach ($component in @('api', 'worker', 'publisher', 'dispatcher')) {
    Invoke-RagKubectl -n rag rollout status "deployment/$component" --timeout=180s
}
