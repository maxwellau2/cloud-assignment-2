.PHONY: build clean demo demo-live json pagerank-10k pagerank-full crawl-10k crawl-full all

build:
	go build -o pagerank .

clean:
	rm -f pagerank

# Built-in demo with simulated robots.txt
demo: build
	./pagerank -crawl

# Built-in demo with live robots.txt fetching
demo-live: build
	./pagerank -crawl -fetch-robots

# Crawl prioritization from JSON graph
json: build
	./pagerank -crawl-json sample_graph.json -crawl-k 5

# PageRank on 10k dataset (iterative + closed-form)
pagerank-10k: build
	./pagerank -file ../web-Google_10k.txt

# PageRank on full dataset (iterative only)
pagerank-full: build
	./pagerank -file ../web-Google.txt -no-closed

# Crawl prioritization on 10k dataset
crawl-10k: build
	./pagerank -file ../web-Google_10k.txt -no-closed -crawl-k 10

# Crawl prioritization on full dataset
crawl-full: build
	./pagerank -file ../web-Google.txt -no-closed -crawl-k 10

# Run everything
all: build demo json pagerank-10k crawl-10k
