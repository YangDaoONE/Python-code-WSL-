def count_remaining_trees(L, M, regions):
    trees = [True] * (L + 1)
    for start, end in regions:
        for i in range(start, end + 1):
            trees[i] = False
    remaining_trees = sum(trees)
    return remaining_trees
L = int(input("马路长度："))
M = int(input("区域数目："))
regions = []
for _ in range(M):
    start = int(input("区域开始位置："))
    end = int(input("区域结束位置："))
    regions.append((start, end))
remaining_trees = count_remaining_trees(L, M, regions)
print(remaining_trees)




