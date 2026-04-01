param(
    [int]$N = 1
)

for ($i = 1; $i -le $N; $i++) {
    docker run -d nn-worker python worker.py `
        --server-host host.docker.internal `
        --server-port 9999
}