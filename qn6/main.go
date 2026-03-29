package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Graph struct {
	N         int
	AdjList   [][]int // adjList[i] = list of nodes i links to
	OutDegree []int
	IDMap     map[int]int // original ID -> contiguous index
	ReverseID []int       // contiguous index -> original ID
}

func parseGraph(filename string) (*Graph, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	type edge struct{ from, to int }
	var edges []edge
	nodeSet := make(map[int]bool)

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		from, err1 := strconv.Atoi(parts[0])
		to, err2 := strconv.Atoi(parts[1])
		if err1 != nil || err2 != nil {
			continue
		}
		edges = append(edges, edge{from, to})
		nodeSet[from] = true
		nodeSet[to] = true
	}

	// Sort nodes for deterministic mapping
	sortedNodes := make([]int, 0, len(nodeSet))
	for n := range nodeSet {
		sortedNodes = append(sortedNodes, n)
	}
	sort.Ints(sortedNodes)

	idMap := make(map[int]int, len(sortedNodes))
	reverseID := make([]int, len(sortedNodes))
	for i, n := range sortedNodes {
		idMap[n] = i
		reverseID[i] = n
	}

	N := len(sortedNodes)
	adjList := make([][]int, N)
	outDegree := make([]int, N)

	for _, e := range edges {
		i := idMap[e.from]
		j := idMap[e.to]
		adjList[i] = append(adjList[i], j)
		outDegree[i]++
	}

	fmt.Printf("Parsed graph: %d nodes, %d edges\n", N, len(edges))
	return &Graph{N, adjList, outDegree, idMap, reverseID}, nil
}

func pageRankIterative(g *Graph, p float64, epsilon float64, maxIter int) []float64 {
	N := g.N
	r := make([]float64, N)
	for i := range r {
		r[i] = 1.0 / float64(N)
	}

	for iter := 1; iter <= maxIter; iter++ {
		rNew := make([]float64, N)

		// Sum of rank from dangling nodes (distributed uniformly)
		danglingSum := 0.0
		for i := 0; i < N; i++ {
			if g.OutDegree[i] == 0 {
				danglingSum += r[i]
			}
		}

		// Link contributions
		for i := 0; i < N; i++ {
			if g.OutDegree[i] > 0 {
				contrib := r[i] / float64(g.OutDegree[i])
				for _, j := range g.AdjList[i] {
					rNew[j] += contrib
				}
			}
		}

		// Apply teleportation + dangling distribution
		teleport := p / float64(N)
		danglingContrib := danglingSum / float64(N)
		diff := 0.0
		for i := 0; i < N; i++ {
			rNew[i] = (1-p)*(rNew[i]+danglingContrib) + teleport
			diff += math.Abs(rNew[i] - r[i])
		}

		r = rNew
		if diff < epsilon {
			fmt.Printf("Iterative: converged at iteration %d (diff=%.2e)\n", iter, diff)
			return r
		}
	}

	fmt.Printf("Iterative: did not converge after %d iterations\n", maxIter)
	return r
}

func pageRankClosedForm(g *Graph, p float64) []float64 {
	N := g.N
	fmt.Printf("Closed-form: building %dx%d dense matrix...\n", N, N)

	// Build A = I - (1-p)*H, then solve A^T * r = (p/N) * 1
	// We build A^T directly: A^T[j][i] = I[j][i] - (1-p)*H[i][j]
	data := make([]float64, N*N)

	// Identity on diagonal
	for i := 0; i < N; i++ {
		data[i*N+i] = 1.0
	}

	// Subtract (1-p)*H^T
	for i := 0; i < N; i++ {
		if g.OutDegree[i] > 0 {
			val := (1 - p) / float64(g.OutDegree[i])
			for _, j := range g.AdjList[i] {
				// H[i][j] = 1/outDegree[i], so H^T[j][i] = 1/outDegree[i]
				// A^T[j][i] -= (1-p)*H[i][j]
				data[j*N+i] -= val
			}
		} else {
			// Dangling node: H[i][j] = 1/N for all j
			val := (1 - p) / float64(N)
			for j := 0; j < N; j++ {
				data[j*N+i] -= val
			}
		}
	}

	AT := mat.NewDense(N, N, data)

	// RHS: (p/N) * 1
	bData := make([]float64, N)
	for i := range bData {
		bData[i] = p / float64(N)
	}
	b := mat.NewVecDense(N, bData)

	// Solve A^T * r = b
	var r mat.VecDense
	err := r.SolveVec(AT, b)
	if err != nil {
		fmt.Printf("Closed-form: solve failed: %v\n", err)
		return nil
	}

	result := make([]float64, N)
	for i := 0; i < N; i++ {
		result[i] = r.AtVec(i)
	}

	fmt.Println("Closed-form: solved successfully")
	return result
}

type ranked struct {
	idx   int
	score float64
}

func printTopK(r []float64, reverseID []int, k int, label string) {
	ranks := make([]ranked, len(r))
	for i, v := range r {
		ranks[i] = ranked{i, v}
	}
	sort.Slice(ranks, func(i, j int) bool {
		return ranks[i].score > ranks[j].score
	})

	fmt.Printf("\nTop %d pages (%s):\n", k, label)
	fmt.Printf("%-6s %-12s %s\n", "Rank", "Node ID", "Score")
	fmt.Println(strings.Repeat("-", 35))
	for i := 0; i < k && i < len(ranks); i++ {
		fmt.Printf("%-6d %-12d %.10f\n", i+1, reverseID[ranks[i].idx], ranks[i].score)
	}
}

// crawlPriority computes a crawl priority score for each page.
// It combines PageRank (authority) with content diversity (out-degree to unique pages).
// Pages that block crawling are filtered out.
// Heuristic: score = alpha * normalized_pagerank + (1-alpha) * normalized_out_degree
// This favors authoritative pages that also link broadly, making them good
// entry points for discovering more high-quality content.
func crawlPriority(g *Graph, pagerank []float64, allowsCrawling map[int]bool, k int, alpha float64, nodeNames map[int]string) {
	type candidate struct {
		nodeID    int
		pagerank  float64
		outDegree int
		score     float64
	}

	// Find max pagerank and max out-degree for normalization
	maxPR := 0.0
	maxOD := 0
	for i := 0; i < g.N; i++ {
		origID := g.ReverseID[i]
		if !allowsCrawling[origID] {
			continue
		}
		if pagerank[i] > maxPR {
			maxPR = pagerank[i]
		}
		if g.OutDegree[i] > maxOD {
			maxOD = g.OutDegree[i]
		}
	}

	if maxPR == 0 || maxOD == 0 {
		fmt.Println("No crawlable pages found.")
		return
	}

	var candidates []candidate
	for i := 0; i < g.N; i++ {
		origID := g.ReverseID[i]
		if !allowsCrawling[origID] {
			continue
		}
		normPR := pagerank[i] / maxPR
		normOD := float64(g.OutDegree[i]) / float64(maxOD)
		score := alpha*normPR + (1-alpha)*normOD
		candidates = append(candidates, candidate{origID, pagerank[i], g.OutDegree[i], score})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	fmt.Printf("\nTop %d URLs to crawl (alpha=%.2f):\n", k, alpha)
	fmt.Printf("%-6s %-20s %-14s %-10s %s\n", "Rank", "Page", "PageRank", "OutDegree", "Score")
	fmt.Println(strings.Repeat("-", 68))
	for i := 0; i < k && i < len(candidates); i++ {
		c := candidates[i]
		name := fmt.Sprintf("%d", c.nodeID)
		if n, ok := nodeNames[c.nodeID]; ok {
			name = n
		}
		fmt.Printf("%-6d %-20s %-14.10f %-10d %.6f\n", i+1, name, c.pagerank, c.outDegree, c.score)
	}

	fmt.Println("\n--- Why high-PageRank pages yield better training data ---")
	fmt.Println("High-PageRank pages are linked to by many other pages, indicating")
	fmt.Println("they are authoritative and widely trusted. This correlates with")
	fmt.Println("higher content quality, factual accuracy, and topical breadth —")
	fmt.Println("all desirable properties for generative AI training data.")
	fmt.Println("")
	fmt.Println("--- Heuristic: Authority-Diversity Score ---")
	fmt.Println("score = alpha * normalized_pagerank + (1-alpha) * normalized_out_degree")
	fmt.Println("")
	fmt.Println("Pages with high out-degree link to many other pages, making them")
	fmt.Println("valuable entry points for discovering additional crawlable content.")
	fmt.Println("Combining authority (PageRank) with diversity (out-degree) ensures")
	fmt.Println("we prioritize pages that are both high-quality and useful for")
	fmt.Println("expanding the crawl frontier. Pages that disallow crawling via")
	fmt.Println("robots.txt are excluded before ranking.")
}

// demoCrawl runs the crawl prioritization on a small example graph
func demoCrawl() {
	fmt.Println("=== AI Web Crawl Prioritization Demo ===")
	fmt.Println()

	// Small directed web graph
	urls := []string{
		"example.com",        // 0
		"news.org",           // 1
		"wiki.edu",           // 2
		"blog.io",            // 3
		"shop.com",           // 4
		"research.gov",       // 5
		"social.net",         // 6
		"archive.org",        // 7
	}

	// Edges: from -> to
	edges := []struct{ from, to int }{
		{0, 1}, {0, 2}, {0, 5},       // example.com links to news, wiki, research
		{1, 0}, {1, 2}, {1, 3},       // news.org links to example, wiki, blog
		{2, 0}, {2, 5}, {2, 7},       // wiki.edu links to example, research, archive
		{3, 0}, {3, 1},               // blog.io links to example, news
		{4, 0}, {4, 3},               // shop.com links to example, blog
		{5, 2}, {5, 7},               // research.gov links to wiki, archive
		{6, 0}, {6, 1}, {6, 3}, {6, 4}, // social.net links broadly
		{7, 2}, {7, 5},               // archive.org links to wiki, research
	}

	N := len(urls)
	adjList := make([][]int, N)
	outDegree := make([]int, N)
	reverseID := make([]int, N)

	for i := 0; i < N; i++ {
		reverseID[i] = i
	}
	for _, e := range edges {
		adjList[e.from] = append(adjList[e.from], e.to)
		outDegree[e.from]++
	}

	g := &Graph{
		N:         N,
		AdjList:   adjList,
		OutDegree: outDegree,
		IDMap:     make(map[int]int),
		ReverseID: reverseID,
	}
	for i := 0; i < N; i++ {
		g.IDMap[i] = i
	}

	// Print the graph
	fmt.Println("Graph:")
	for i, url := range urls {
		links := []string{}
		for _, j := range adjList[i] {
			links = append(links, urls[j])
		}
		fmt.Printf("  %s -> %s\n", url, strings.Join(links, ", "))
	}

	// robots.txt simulation
	allowsCrawling := map[int]bool{
		0: true,  // example.com — allows
		1: true,  // news.org — allows
		2: true,  // wiki.edu — allows
		3: true,  // blog.io — allows
		4: false, // shop.com — blocks GPTBot
		5: true,  // research.gov — allows
		6: false, // social.net — blocks GPTBot
		7: true,  // archive.org — allows
	}

	fmt.Println("\nrobots.txt permissions:")
	for i, url := range urls {
		status := "ALLOW"
		if !allowsCrawling[i] {
			status = "BLOCK"
		}
		fmt.Printf("  %-20s %s\n", url, status)
	}

	// Compute PageRank
	fmt.Println()
	pr := pageRankIterative(g, 0.15, 1e-10, 200)

	fmt.Println("\nPageRank scores:")
	for i, url := range urls {
		fmt.Printf("  %-20s %.6f\n", url, pr[i])
	}

	// Run crawl prioritization
	nodeNames := make(map[int]string)
	for i, url := range urls {
		nodeNames[i] = url
	}
	crawlPriority(g, pr, allowsCrawling, 5, 0.7, nodeNames)
}

func main() {
	filePath := flag.String("file", "../web-Google_10k.txt", "path to graph file")
	p := flag.Float64("p", 0.15, "teleport probability")
	noClosedForm := flag.Bool("no-closed", false, "skip closed-form (for large graphs)")
	crawlDemo := flag.Bool("crawl", false, "run AI crawl prioritization demo")
	crawlK := flag.Int("crawl-k", 10, "top-k pages to return for crawl prioritization")
	crawlAlpha := flag.Float64("crawl-alpha", 0.7, "weight for PageRank vs out-degree (0-1)")
	flag.Parse()

	if *crawlDemo {
		demoCrawl()
		return
	}

	g, err := parseGraph(*filePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Iterative PageRank
	start := time.Now()
	rIter := pageRankIterative(g, *p, 1e-10, 200)
	iterTime := time.Since(start)
	printTopK(rIter, g.ReverseID, 10, "iterative")
	fmt.Printf("Iterative time: %v\n", iterTime)

	if *noClosedForm {
		// Run crawl prioritization on the real graph
		// Assume all pages allow crawling (no robots.txt data available)
		allowsAll := make(map[int]bool, g.N)
		for i := 0; i < g.N; i++ {
			allowsAll[g.ReverseID[i]] = true
		}
		crawlPriority(g, rIter, allowsAll, *crawlK, *crawlAlpha, nil)
		return
	}

	// Closed-form PageRank
	start = time.Now()
	rClosed := pageRankClosedForm(g, *p)
	closedTime := time.Since(start)
	if rClosed == nil {
		return
	}
	printTopK(rClosed, g.ReverseID, 10, "closed-form")
	fmt.Printf("Closed-form time: %v\n", closedTime)

	// Compare
	maxDiff := 0.0
	for i := 0; i < g.N; i++ {
		d := math.Abs(rIter[i] - rClosed[i])
		if d > maxDiff {
			maxDiff = d
		}
	}
	fmt.Printf("\nMax absolute difference between methods: %.2e\n", maxDiff)
	fmt.Printf("Speedup: closed-form is %.1fx %s than iterative\n",
		math.Max(float64(iterTime), float64(closedTime))/math.Max(1, math.Min(float64(iterTime), float64(closedTime))),
		map[bool]string{true: "slower", false: "faster"}[closedTime > iterTime])
}
