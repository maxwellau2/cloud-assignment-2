# PageRank Implementation Pseudocode

## Overview

We implement PageRank two ways and compare their outputs:
1. **Iterative (power method)** - the standard PageRank algorithm
2. **Closed-form (matrix inversion)** - direct solve for verification

## Step 1: Parse the Graph

```
function parseGraph(filename):
    edges = []
    nodeSet = {}

    for each line in file:
        skip lines starting with '#'
        parse (fromNode, toNode)
        edges.append((fromNode, toNode))
        nodeSet.add(fromNode)
        nodeSet.add(toNode)

    // Remap node IDs to contiguous indices 0..N-1
    // (original IDs are sparse, e.g., 0, 11342, 824020, ...)
    sortedNodes = sort(nodeSet)
    idMap = {}
    for i, node in sortedNodes:
        idMap[node] = i

    N = len(sortedNodes)
    return edges, idMap, N
```

**Rationale:** Node IDs in the dataset are sparse (e.g., 0, 11342, 824020). We remap
them to contiguous indices so we can use dense arrays and matrices without
allocating for millions of empty slots.

## Step 2: Build the Row-Stochastic Transition Matrix H

```
function buildTransitionMatrix(edges, idMap, N):
    // Count out-degree for each node
    outDegree = array of size N, initialized to 0
    adjList = map from node -> list of targets

    for (from, to) in edges:
        i = idMap[from]
        j = idMap[to]
        outDegree[i] += 1
        adjList[i].append(j)

    // Build H as sparse structure
    // H[i][j] = 1 / outDegree[i] if i -> j
    H = sparse matrix of size N x N
    for i in 0..N-1:
        if outDegree[i] > 0:
            for j in adjList[i]:
                H[i][j] = 1.0 / outDegree[i]
        else:
            // Dangling node: no outlinks
            // Treat as if it links to all pages (uniform row)
            for j in 0..N-1:
                H[i][j] = 1.0 / N

    return H, outDegree
```

**Rationale:** Row-stochastic means each row sums to 1. Entry H[i][j] represents
the probability of moving from page i to page j by following a link.

Dangling nodes (pages with no outlinks) are a special case. Without handling,
their row is all zeros, breaking the stochastic property. The standard fix is to
treat them as teleporting uniformly to all pages.

## Step 3: Iterative PageRank (Power Method)

```
function pageRankIterative(H, N, p, epsilon=1e-10, maxIter=200):
    // Initialize uniform distribution
    r = array of size N, all values = 1.0 / N

    for iter in 1..maxIter:
        r_new = array of size N, all values = 0

        // Multiply: r_new = r * H (row vector times matrix)
        // But we do it sparsely using adjacency list
        for i in 0..N-1:
            if outDegree[i] > 0:
                contribution = r[i] / outDegree[i]
                for j in adjList[i]:
                    r_new[j] += contribution
            else:
                // Dangling node distributes evenly
                for j in 0..N-1:
                    r_new[j] += r[i] / N

        // Apply teleportation
        // r_final = (1-p) * r_new + (p/N) * 1
        for i in 0..N-1:
            r_new[i] = (1 - p) * r_new[i] + p / N

        // Check convergence (L1 norm)
        diff = sum(|r_new[i] - r[i]| for i in 0..N-1)
        r = r_new

        if diff < epsilon:
            print("Converged at iteration", iter)
            break

    return r
```

**Rationale:** The power method repeatedly applies the transition:

    rᵀ ← (1-p) · rᵀH + (p/N) · 1ᵀ

This is equivalent to simulating the random surfer. At each step, with probability
(1-p) the surfer follows a link (matrix multiply by H), and with probability p
the surfer teleports to a random page (add p/N to every entry).

We use the sparse adjacency list rather than a dense matrix multiply. Each
iteration is O(edges), not O(N^2). Convergence is typically reached in 50-100
iterations.

The L1 norm measures total absolute change. When it drops below epsilon,
the distribution has stabilized.

## Step 4: Closed-Form PageRank (Matrix Inversion)

```
function pageRankClosedForm(H, N, p):
    // Compute: rᵀ = (p/N) · 1ᵀ · [I - (1-p)H]⁻¹

    // Build dense matrix A = I - (1-p)*H
    A = identity matrix of size N x N
    for i in 0..N-1:
        for j in adjList[i]:
            A[i][j] -= (1 - p) * H[i][j]
        if outDegree[i] == 0:
            for j in 0..N-1:
                A[i][j] -= (1 - p) / N

    // Invert A
    A_inv = invert(A)

    // Multiply: rᵀ = (p/N) · 1ᵀ · A_inv
    // 1ᵀ · A_inv = column sums of A_inv
    r = array of size N
    for j in 0..N-1:
        colSum = sum(A_inv[i][j] for i in 0..N-1)
        r[j] = (p / N) * colSum

    return r
```

**Rationale:** This directly solves the linear system derived in the report.
For N=10,000, the dense matrix is 10000x10000 = 100M entries (about 800MB
as float64). This is feasible but expensive. For the full web-Google.txt
(~875k nodes), this approach is impractical — only the iterative method scales.

Note: Instead of explicitly inverting the matrix (which is numerically less stable),
an alternative is to solve the linear system:

    rᵀ · [I - (1-p)H] = (p/N) · 1ᵀ

transposed as:

    [I - (1-p)H]ᵀ · r = (p/N) · 1

using LU decomposition or similar. This is faster and more stable than full
inversion.

## Step 5: Compare Results

```
function compare(r_iterative, r_closed, N):
    maxDiff = 0
    for i in 0..N-1:
        diff = |r_iterative[i] - r_closed[i]|
        maxDiff = max(maxDiff, diff)

    print("Max absolute difference:", maxDiff)
    // Should be very small, e.g., < 1e-8

    // Print top 10 pages by rank
    ranked = sortByValueDescending(r_iterative)
    for i in 0..10:
        nodeIdx, score = ranked[i]
        originalId = reverseIdMap[nodeIdx]
        print("Rank", i+1, ": Node", originalId, "Score", score)
```

**Rationale:** Comparing the two methods validates correctness. The iterative
method is an approximation (stopped at epsilon), so a small difference is
expected. A large difference would indicate a bug.

## Summary

| Method | Time Complexity | Space Complexity | Scalability |
|--------|----------------|------------------|-------------|
| Iterative | O(iterations * edges) | O(N + edges) | Scales to millions of nodes |
| Closed-form | O(N^3) | O(N^2) | Feasible up to ~10k nodes |

Both methods compute the same result. The iterative method is the practical
algorithm; the closed form is the mathematical verification.
