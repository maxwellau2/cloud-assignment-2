# PageRank - Build & Run

## Prerequisites

- Go 1.21+

## Build

```bash
cd qn6
go mod tidy
go build -o pagerank .
```

## Run

```bash
# Default: uses ../web-Google_10k.txt with p=0.15
./pagerank

# Custom graph file
./pagerank -file ../web-Google.txt

# Custom teleport probability
./pagerank -p 0.3

# Skip closed-form (for large graphs where N^2 matrix won't fit in memory)
./pagerank -file ../web-Google.txt -no-closed

# Combine flags
./pagerank -file ../web-Google_10k.txt -p 0.2 -no-closed
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-file` | `../web-Google_10k.txt` | Path to graph edge list |
| `-p` | `0.15` | Teleport probability |
| `-no-closed` | `false` | Skip closed-form computation |

## Output

The program prints:
1. Graph stats (nodes, edges)
2. Iterative PageRank convergence info
3. Top 10 pages by rank (iterative)
4. Top 10 pages by rank (closed-form)
5. Max absolute difference between the two methods
