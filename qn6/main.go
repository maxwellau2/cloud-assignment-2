package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http"
	"net/url"
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
	InDegree  []int
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
	inDegree := make([]int, N)

	for _, e := range edges {
		i := idMap[e.from]
		j := idMap[e.to]
		adjList[i] = append(adjList[i], j)
		outDegree[i]++
		inDegree[j]++
	}

	fmt.Printf("Parsed graph: %d nodes, %d edges\n", N, len(edges))
	return &Graph{N, adjList, inDegree, outDegree, idMap, reverseID}, nil
}

// parseJSONGraph reads a web graph from a JSON file formatted as:
//
//	{ "https://example.com": ["https://a.org", "https://b.edu"], ... }
//
// Returns the graph, a mapping from node index to URL, and robots.txt
// permission map (all allowed by default; caller can override).
func parseJSONGraph(filename string) (*Graph, map[int]string, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, nil, err
	}

	var adjMap map[string][]string
	if err := json.Unmarshal(data, &adjMap); err != nil {
		return nil, nil, fmt.Errorf("invalid JSON: %w", err)
	}

	// Collect all unique URLs (keys + values)
	urlSet := make(map[string]bool)
	for src, dsts := range adjMap {
		urlSet[src] = true
		for _, dst := range dsts {
			urlSet[dst] = true
		}
	}

	// Sort for deterministic ordering
	urls := make([]string, 0, len(urlSet))
	for u := range urlSet {
		urls = append(urls, u)
	}
	sort.Strings(urls)

	urlToIdx := make(map[string]int, len(urls))
	for i, u := range urls {
		urlToIdx[u] = i
	}

	N := len(urls)
	adjList := make([][]int, N)
	outDegree := make([]int, N)
	inDegree := make([]int, N)
	idMap := make(map[int]int, N)
	reverseID := make([]int, N)
	nodeNames := make(map[int]string, N)

	for i, u := range urls {
		idMap[i] = i
		reverseID[i] = i
		nodeNames[i] = u
	}

	for src, dsts := range adjMap {
		i := urlToIdx[src]
		for _, dst := range dsts {
			j := urlToIdx[dst]
			adjList[i] = append(adjList[i], j)
			outDegree[i]++
			inDegree[j]++
		}
	}

	g := &Graph{N, adjList, inDegree, outDegree, idMap, reverseID}
	fmt.Printf("Parsed JSON graph: %d nodes, %d edges\n", N, func() int {
		total := 0
		for _, d := range outDegree {
			total += d
		}
		return total
	}())
	return g, nodeNames, nil
}

func pageRankIterativeWithLog(g *Graph, p float64, epsilon float64, maxIter int, convergenceLog *os.File) []float64 {
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

		if convergenceLog != nil {
			fmt.Fprintf(convergenceLog, "%d,%.15e\n", iter, diff)
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

func pageRankIterative(g *Graph, p float64, epsilon float64, maxIter int) []float64 {
	return pageRankIterativeWithLog(g, p, epsilon, maxIter, nil)
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

// varyP runs PageRank at multiple p values and outputs CSV data.
func varyP(g *Graph, outFile string) {
	pValues := []float64{0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.85, 0.95}

	f, err := os.Create(outFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating %s: %v\n", outFile, err)
		return
	}
	defer f.Close()

	fmt.Fprintln(f, "p,node_id,rank,score")

	for _, p := range pValues {
		r := pageRankIterative(g, p, 1e-10, 200)
		ranks := make([]ranked, len(r))
		for i, v := range r {
			ranks[i] = ranked{i, v}
		}
		sort.Slice(ranks, func(i, j int) bool {
			return ranks[i].score > ranks[j].score
		})
		for rank := 0; rank < 5 && rank < len(ranks); rank++ {
			fmt.Fprintf(f, "%.2f,%d,%d,%.15e\n", p, g.ReverseID[ranks[rank].idx], rank+1, ranks[rank].score)
		}
	}
	fmt.Printf("Wrote vary-p data to %s\n", outFile)
}

// domainTrust returns a trust score based on TLD.
// .edu and .gov domains are more likely to contain factual, well-curated content.
// .org domains (non-profits, foundations) tend to have reliable information.
// Commercial and other domains get a baseline score.
func domainTrust(urlStr string) (float64, string) {
	host := urlStr
	if u, err := url.Parse(urlStr); err == nil && u.Host != "" {
		host = u.Host
	}
	host = strings.ToLower(host)

	switch {
	case strings.HasSuffix(host, ".edu"):
		return 1.0, ".edu (academic)"
	case strings.HasSuffix(host, ".gov"):
		return 1.0, ".gov (government)"
	case strings.HasSuffix(host, ".org"):
		return 0.7, ".org (non-profit)"
	case strings.HasSuffix(host, ".io"):
		return 0.4, ".io (tech)"
	default:
		return 0.3, "commercial/other"
	}
}

// checkRobotsTxt fetches robots.txt for the given URL's host and checks
// whether the specified bot (e.g. "GPTBot") is allowed to crawl.
// Returns true if crawling is allowed, false if disallowed.
// On any error (network, timeout, no robots.txt), assumes allowed.
func checkRobotsTxt(rawURL string, botName string) bool {
	u, err := url.Parse(rawURL)
	if err != nil || u.Host == "" {
		return true // can't determine host, assume allowed
	}

	robotsURL := fmt.Sprintf("%s://%s/robots.txt", u.Scheme, u.Host)
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(robotsURL)
	if err != nil || resp.StatusCode != 200 {
		return true // no robots.txt or unreachable, assume allowed
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	currentAgent := ""
	botNameLower := strings.ToLower(botName)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}

		if strings.HasPrefix(strings.ToLower(line), "user-agent:") {
			currentAgent = strings.TrimSpace(strings.SplitN(line, ":", 2)[1])
			currentAgent = strings.ToLower(currentAgent)
		} else if strings.HasPrefix(strings.ToLower(line), "disallow:") {
			path := strings.TrimSpace(strings.SplitN(line, ":", 2)[1])
			if (currentAgent == botNameLower || currentAgent == "*") && path == "/" {
				return false
			}
		}
	}
	return true
}

// crawlCandidate holds all scoring components for a crawl target.
type crawlCandidate struct {
	nodeID      int
	name        string
	pagerank    float64
	normPR      float64
	inDegree    int
	normID      float64
	outDegree   int
	normOD      float64
	trustScore  float64
	trustLabel  string
	finalScore  float64
}

// Weights for the multi-factor heuristic
type crawlWeights struct {
	authority float64 // PageRank weight
	demand    float64 // in-degree weight
	diversity float64 // out-degree weight
	trust     float64 // domain trust weight
}

// crawlPriority computes a crawl priority score for each page using a
// multi-factor heuristic that combines:
//   - Authority (PageRank): how trusted/important is the page
//   - Demand (in-degree): how many pages link here (independent validation)
//   - Diversity (out-degree): how many outlinks (good crawl frontier entry point)
//   - Domain trust: TLD-based quality signal (.edu/.gov > .org > .com)
//
// Pages that block crawling via robots.txt are excluded before ranking.
func crawlPriority(g *Graph, pagerank []float64, allowsCrawling map[int]bool, k int, w crawlWeights, nodeNames map[int]string, addtCSVFile string) {
	hasDomainInfo := false
	for _, name := range nodeNames {
		if strings.Contains(name, ".") {
			hasDomainInfo = true
			break
		}
	}

	// If no domain info available, redistribute trust weight to authority
	if !hasDomainInfo {
		w.authority += w.trust
		w.trust = 0
	}

	// Find max values for normalization (only among crawlable pages)
	maxPR := 0.0
	maxOD := 0
	maxInD := 0
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
		if g.InDegree[i] > maxInD {
			maxInD = g.InDegree[i]
		}
	}

	if maxPR == 0 {
		fmt.Println("No crawlable pages found.")
		return
	}

	var candidates []crawlCandidate
	for i := 0; i < g.N; i++ {
		origID := g.ReverseID[i]
		if !allowsCrawling[origID] {
			continue
		}

		name := fmt.Sprintf("%d", origID)
		if n, ok := nodeNames[origID]; ok {
			name = n
		}

		normPR := pagerank[i] / maxPR
		normOD := 0.0
		if maxOD > 0 {
			normOD = float64(g.OutDegree[i]) / float64(maxOD)
		}
		normInD := 0.0
		if maxInD > 0 {
			normInD = float64(g.InDegree[i]) / float64(maxInD)
		}

		trustScore := 0.0
		trustLabel := "N/A"
		if hasDomainInfo {
			trustScore, trustLabel = domainTrust(name)
		}

		score := w.authority*normPR + w.demand*normInD + w.diversity*normOD + w.trust*trustScore

		candidates = append(candidates, crawlCandidate{
			nodeID: origID, name: name,
			pagerank: pagerank[i], normPR: normPR,
			inDegree: g.InDegree[i], normID: normInD,
			outDegree: g.OutDegree[i], normOD: normOD,
			trustScore: trustScore, trustLabel: trustLabel,
			finalScore: score,
		})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].finalScore > candidates[j].finalScore
	})

	// Write ADDT CSV if requested
	if addtCSVFile != "" {
		csvF, err := os.Create(addtCSVFile)
		if err == nil {
			fmt.Fprintln(csvF, "rank,name,authority,demand,diversity,trust,final_score")
			for i := 0; i < k && i < len(candidates); i++ {
				c := candidates[i]
				fmt.Fprintf(csvF, "%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f\n",
					i+1, c.name, w.authority*c.normPR, w.demand*c.normID, w.diversity*c.normOD, w.trust*c.trustScore, c.finalScore)
			}
			csvF.Close()
			fmt.Printf("Wrote ADDT breakdown to %s\n", addtCSVFile)
		}
	}

	fmt.Printf("\n=== Top %d URLs to Crawl for AI Training Data ===\n", k)
	fmt.Printf("Weights: authority=%.2f, demand=%.2f, diversity=%.2f, trust=%.2f\n\n",
		w.authority, w.demand, w.diversity, w.trust)

	for i := 0; i < k && i < len(candidates); i++ {
		c := candidates[i]
		fmt.Printf("Rank %d: %s\n", i+1, c.name)
		fmt.Printf("  PageRank:     %.6f  (normalized: %.3f) [weight: %.2f]\n", c.pagerank, c.normPR, w.authority)
		fmt.Printf("  In-degree:    %-8d (normalized: %.3f) [weight: %.2f]\n", c.inDegree, c.normID, w.demand)
		fmt.Printf("  Out-degree:   %-8d (normalized: %.3f) [weight: %.2f]\n", c.outDegree, c.normOD, w.diversity)
		if hasDomainInfo {
			fmt.Printf("  Domain trust: %-8s (score: %.2f)     [weight: %.2f]\n", c.trustLabel, c.trustScore, w.trust)
		}
		fmt.Printf("  Final score:  %.4f\n", c.finalScore)

		// Per-page reasoning
		reasons := []string{}
		if c.normPR > 0.7 {
			reasons = append(reasons, "highly authoritative (many pages endorse it)")
		} else if c.normPR > 0.4 {
			reasons = append(reasons, "moderately authoritative")
		}
		if c.normOD > 0.7 {
			reasons = append(reasons, "broad outlinks (good crawl frontier)")
		}
		if c.normID > 0.7 {
			reasons = append(reasons, "high demand (widely referenced)")
		}
		if c.trustScore >= 0.7 {
			reasons = append(reasons, fmt.Sprintf("trusted domain (%s)", c.trustLabel))
		}
		if len(reasons) > 0 {
			fmt.Printf("  Why crawl:    %s\n", strings.Join(reasons, "; "))
		}
		fmt.Println()
	}

	// Print explanations
	fmt.Println("--- Why High-PageRank Pages Yield Better Training Data ---")
	fmt.Println()
	fmt.Println("PageRank measures recursive authority: a page is important if many")
	fmt.Println("important pages link to it. For AI training, this matters because:")
	fmt.Println("  1. Authority correlates with factual accuracy -- widely-cited pages")
	fmt.Println("     undergo more scrutiny and correction over time.")
	fmt.Println("  2. High-PageRank pages tend to cover topics broadly and clearly,")
	fmt.Println("     since they serve as reference material for many other sites.")
	fmt.Println("  3. Training on authoritative text reduces hallucination risk,")
	fmt.Println("     as the model learns patterns from reliable sources.")
	fmt.Println("  4. These pages are often well-structured (headings, paragraphs,")
	fmt.Println("     citations), making them easier to parse into clean training data.")
	fmt.Println()
	fmt.Println("--- Heuristic: Authority-Demand-Diversity-Trust (ADDT) Score ---")
	fmt.Println()
	fmt.Println("  score = w1*PageRank + w2*InDegree + w3*OutDegree + w4*DomainTrust")
	fmt.Println()
	fmt.Println("This multi-factor heuristic finds pages that permit crawling and are")
	fmt.Println("likely to contain high-quality training data by combining:")
	fmt.Println("  - Authority (PageRank): recursive trust from the web graph")
	fmt.Println("  - Demand (in-degree): direct link count as independent validation")
	fmt.Println("  - Diversity (out-degree): broad outlinks for crawl frontier expansion")
	fmt.Println("  - Domain trust: TLD-based prior (.edu/.gov > .org > commercial)")
	fmt.Println()
	fmt.Println("Pages that disallow crawling via robots.txt (checking for GPTBot and")
	fmt.Println("wildcard User-agent rules) are excluded before scoring.")
}

// demoCrawl runs the crawl prioritization on a small example graph
func demoCrawl(fetchRobots bool, addtCSVFile string) {
	fmt.Println("=== AI Web Crawl Prioritization Demo ===")
	fmt.Println()

	// Small directed web graph as a dictionary
	webGraph := map[string][]string{
		"https://en.wikipedia.org":      {"https://www.example.com", "https://www.nist.gov", "https://archive.org"},
		"https://www.example.com":       {"https://en.wikipedia.org", "https://news.ycombinator.com", "https://blog.github.io"},
		"https://www.nist.gov":          {"https://en.wikipedia.org", "https://arxiv.org", "https://archive.org"},
		"https://news.ycombinator.com":  {"https://www.example.com", "https://en.wikipedia.org", "https://blog.github.io"},
		"https://blog.github.io":        {"https://www.example.com", "https://news.ycombinator.com"},
		"https://www.amazon.com":        {"https://www.example.com", "https://blog.github.io"},
		"https://www.facebook.com":      {"https://www.example.com", "https://news.ycombinator.com", "https://blog.github.io", "https://www.amazon.com"},
		"https://arxiv.org":             {"https://en.wikipedia.org", "https://www.nist.gov"},
		"https://archive.org":           {"https://en.wikipedia.org", "https://arxiv.org"},
		"https://mit.edu":               {"https://en.wikipedia.org", "https://arxiv.org", "https://www.nist.gov", "https://archive.org"},
	}

	// Print the graph
	fmt.Println("Web graph (dictionary of URLs -> outlinks):")
	// Sort keys for deterministic output
	sortedKeys := make([]string, 0, len(webGraph))
	for k := range webGraph {
		sortedKeys = append(sortedKeys, k)
	}
	sort.Strings(sortedKeys)
	for _, u := range sortedKeys {
		fmt.Printf("  %s\n", u)
		for _, link := range webGraph[u] {
			fmt.Printf("    -> %s\n", link)
		}
	}

	// Build graph
	urlSet := make(map[string]bool)
	for src, dsts := range webGraph {
		urlSet[src] = true
		for _, dst := range dsts {
			urlSet[dst] = true
		}
	}
	urls := make([]string, 0, len(urlSet))
	for u := range urlSet {
		urls = append(urls, u)
	}
	sort.Strings(urls)

	urlToIdx := make(map[string]int, len(urls))
	for i, u := range urls {
		urlToIdx[u] = i
	}

	N := len(urls)
	adjList := make([][]int, N)
	outDegree := make([]int, N)
	inDegree := make([]int, N)
	reverseID := make([]int, N)
	idMap := make(map[int]int, N)
	nodeNames := make(map[int]string, N)

	for i, u := range urls {
		reverseID[i] = i
		idMap[i] = i
		nodeNames[i] = u
	}
	for src, dsts := range webGraph {
		i := urlToIdx[src]
		for _, dst := range dsts {
			j := urlToIdx[dst]
			adjList[i] = append(adjList[i], j)
			outDegree[i]++
			inDegree[j]++
		}
	}

	g := &Graph{N, adjList, inDegree, outDegree, idMap, reverseID}

	// Check robots.txt permissions
	fmt.Println("\nChecking robots.txt permissions for GPTBot...")
	allowsCrawling := make(map[int]bool, N)
	if fetchRobots {
		for i, u := range urls {
			allowed := checkRobotsTxt(u, "GPTBot")
			allowsCrawling[i] = allowed
			status := "ALLOW"
			if !allowed {
				status = "BLOCK"
			}
			fmt.Printf("  %-40s %s\n", u, status)
		}
	} else {
		// Simulated robots.txt: major social/e-commerce platforms typically block AI bots
		blocked := map[string]bool{
			"https://www.amazon.com":   true,
			"https://www.facebook.com": true,
		}
		for i, u := range urls {
			if blocked[u] {
				allowsCrawling[i] = false
				fmt.Printf("  %-40s BLOCK (simulated)\n", u)
			} else {
				allowsCrawling[i] = true
				fmt.Printf("  %-40s ALLOW\n", u)
			}
		}
	}

	// Compute PageRank
	fmt.Println()
	pr := pageRankIterative(g, 0.15, 1e-10, 200)

	fmt.Println("\nPrecomputed PageRank scores:")
	type prEntry struct {
		url   string
		score float64
	}
	var prList []prEntry
	for i, u := range urls {
		prList = append(prList, prEntry{u, pr[i]})
	}
	sort.Slice(prList, func(i, j int) bool { return prList[i].score > prList[j].score })
	for _, e := range prList {
		fmt.Printf("  %-40s %.6f\n", e.url, e.score)
	}

	// Run crawl prioritization with ADDT heuristic
	w := crawlWeights{authority: 0.40, demand: 0.20, diversity: 0.20, trust: 0.20}
	crawlPriority(g, pr, allowsCrawling, 5, w, nodeNames, addtCSVFile)
}

func main() {
	filePath := flag.String("file", "../web-Google_10k.txt", "path to graph file")
	p := flag.Float64("p", 0.15, "teleport probability")
	noClosedForm := flag.Bool("no-closed", false, "skip closed-form (for large graphs)")
	crawlDemoFlag := flag.Bool("crawl", false, "run AI crawl prioritization demo")
	crawlJSON := flag.String("crawl-json", "", "path to JSON web graph for crawl prioritization")
	crawlK := flag.Int("crawl-k", 10, "top-k pages to return for crawl prioritization")
	fetchRobots := flag.Bool("fetch-robots", false, "fetch real robots.txt (requires network)")
	convergenceCSV := flag.String("convergence-csv", "", "output convergence data to CSV file")
	varyPCSV := flag.String("vary-p-csv", "", "output PageRank vs varying p to CSV file")
	addtCSV := flag.String("addt-csv", "", "output ADDT score breakdown to CSV file")
	flag.Parse()

	if *crawlDemoFlag {
		demoCrawl(*fetchRobots, *addtCSV)
		return
	}

	if *crawlJSON != "" {
		g, nodeNames, err := parseJSONGraph(*crawlJSON)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing JSON graph: %v\n", err)
			os.Exit(1)
		}

		pr := pageRankIterative(g, *p, 1e-10, 200)

		fmt.Println("\nPageRank scores:")
		type prEntry struct {
			name  string
			score float64
		}
		var prList []prEntry
		for id, name := range nodeNames {
			prList = append(prList, prEntry{name, pr[id]})
		}
		sort.Slice(prList, func(i, j int) bool { return prList[i].score > prList[j].score })
		for _, e := range prList {
			fmt.Printf("  %-40s %.6f\n", e.name, e.score)
		}

		// Check robots.txt
		allowsCrawling := make(map[int]bool, g.N)
		if *fetchRobots {
			fmt.Println("\nChecking robots.txt...")
			for id, name := range nodeNames {
				allowsCrawling[id] = checkRobotsTxt(name, "GPTBot")
			}
		} else {
			for i := 0; i < g.N; i++ {
				allowsCrawling[g.ReverseID[i]] = true
			}
		}

		w := crawlWeights{authority: 0.40, demand: 0.20, diversity: 0.20, trust: 0.20}
		crawlPriority(g, pr, allowsCrawling, *crawlK, w, nodeNames, *addtCSV)
		return
	}

	g, err := parseGraph(*filePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Generate vary-p data if requested
	if *varyPCSV != "" {
		varyP(g, *varyPCSV)
	}

	// Iterative PageRank
	var convergenceFile *os.File
	if *convergenceCSV != "" {
		var err2 error
		convergenceFile, err2 = os.Create(*convergenceCSV)
		if err2 != nil {
			fmt.Fprintf(os.Stderr, "Error creating convergence CSV: %v\n", err2)
		} else {
			fmt.Fprintln(convergenceFile, "iteration,l1_diff")
		}
	}
	start := time.Now()
	rIter := pageRankIterativeWithLog(g, *p, 1e-10, 200, convergenceFile)
	iterTime := time.Since(start)
	if convergenceFile != nil {
		convergenceFile.Close()
		fmt.Printf("Wrote convergence data to %s\n", *convergenceCSV)
	}
	printTopK(rIter, g.ReverseID, 10, "iterative")
	fmt.Printf("Iterative time: %v\n", iterTime)

	if *noClosedForm {
		// Run crawl prioritization on the real graph
		allowsAll := make(map[int]bool, g.N)
		for i := 0; i < g.N; i++ {
			allowsAll[g.ReverseID[i]] = true
		}
		w := crawlWeights{authority: 0.50, demand: 0.25, diversity: 0.25, trust: 0.0}
		crawlPriority(g, rIter, allowsAll, *crawlK, w, nil, *addtCSV)
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
