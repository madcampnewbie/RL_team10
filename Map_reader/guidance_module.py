import heapq
import matplotlib.pyplot as plt

# 상, 하, 좌, 우 (dy, dx)

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
action_map = {
    (-1, 0): 0,
    (1, 0): 1,
    (0, -1): 2,
    (0, 1): 3,
}

def compute_policy_field(grid, goal):
    H, W = len(grid), len(grid[0])
    gy, gx = goal
    dist = [[float('inf')] * W for _ in range(H)]
    dist[gy][gx] = 0
    queue = [(0, gy, gx)]

    while queue:
        cost, y, x = heapq.heappop(queue)
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] == 0:
                if dist[ny][nx] > cost + 1:
                    dist[ny][nx] = cost + 1
                    heapq.heappush(queue, (dist[ny][nx], ny, nx))

    # 각 셀에서 가장 가까운 방향으로의 액션 계산
    policy = [['#'] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            if grid[y][x] == 1:
                continue
            if (y, x) == goal:
                policy[y][x] = '*'
                continue
            if dist[y][x] == float('inf'):
                policy[y][x] = 'X'
                continue

            min_cost = dist[y][x]
            best_action = (-1, 0)
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and dist[ny][nx] < min_cost:
                    min_cost = dist[ny][nx]
                    best_action = (dy, dx)

            policy[y][x] = action_map[best_action]

    return policy

def visualize_policy(policy):
    H, W = len(policy), len(policy[0])
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.invert_yaxis()
    ax.set_aspect('equal')
    for y in range(H):
        for x in range(W):
            c = policy[y][x]
            if c == '#':
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black'))
            else:
                ax.text(x, y, c, ha='center', va='center', fontsize=14)
    plt.grid(True)
    plt.show()