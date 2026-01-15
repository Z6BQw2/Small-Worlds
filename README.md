# Small-Worlds
Results:
- Verified the small world hypothesis.
- Verified our theory that the formation of Small World clusters is irregular and non-trivial.
- Showed a pathway to simulating small worlds
- Found results hinting at the idea that the sparsity of small worlds mostly come from a recursive shape (clusters themselves organized as Small Worlds) -> Which I used afterwards to develop a better gradient checkpointing strategy than PyTorch's base method (same time complexity, 15% lower memory usage, by usingthis theory to find -more- optimal checkpoints)
