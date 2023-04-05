'''
### 1758 알바생 강호
import sys
input = sys.stdin.readline

n = int(input())
cash = []
for _ in range(n):
    pay = int(input())
    cash.append(pay)
answer = 0
cash.sort(reverse = True)
for i in range(len(cash)):
    if cash[i] > i:
        answer += (cash[i]-i)
print(answer)
'''

'''
### 19941 햄버거 분배
import sys
input = sys.stdin.readline

n, k = map(int, input().split())
table = list(input().rstrip())
person = 0
for i in range(n):
    if table[i] == 'P':
        for j in range(i-k, i+k+1):
            if 0<=j<n and table[j] == 'H':
                person += 1
                table[j] = 'False'
                break
print(person)
'''

'''
### 10026 적록색약
import sys
input = sys.stdin.readline
import copy
sys.setrecursionlimit(10**6)

n = int(input())
graph = [list(input().rstrip()) for _ in range(n)]
graph_answer = 0
dx = [-1, 1, 0 ,0]
dy = [0, 0, -1, 1]
disease = copy.deepcopy(graph)
for i in range(n):
    for j in range(n):
        if disease[i][j] == 'G':
            disease[i][j] = 'R'
disease_answer = 0
visited = [[0]*n for _ in range(n)]

def dfs(x, y, matrix):
    visited[x][y] = 1
    now_color = matrix[x][y]
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if 0<=nx<n and 0<=ny<n and not visited[nx][ny] and matrix[nx][ny] == now_color:
            dfs(nx, ny, matrix)

for i in range(n):
    for j in range(n):
        if not visited[i][j]:
            dfs(i, j, graph)
            graph_answer += 1

visited = [[0]*n for _ in range(n)]
for i in range(n):
    for j in range(n):
        if not visited[i][j]:
            dfs(i, j, disease)
            disease_answer += 1
print(graph_answer, disease_answer)
'''

'''
### 2206 벽 부수고 이동하기
import sys
input = sys.stdin.readline
from collections import deque

n, m = map(int, input().split())
graph = [list(map(int, input().rstrip())) for _ in range(n)]
# visited[x][y][z]에서 z=0 : 통과 가능 , z=1 : 통과 X
visited = [[[0]*2 for _ in range(m)] for _ in range(n)]
visited[0][0][0] = 1
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

def bfs(a, b, c):
    queue = deque()
    queue.append((a, b, c))
    while queue:
        x, y, z = queue.popleft()
        if x == n-1 and y == m-1:
            return visited[x][y][z]
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0<=nx<n and 0<=ny<m:
                # 다음 위치 벽이고, 통과 가능한 경우
                if graph[nx][ny] == 1 and z == 0:
                    visited[nx][ny][1] = visited[x][y][0] + 1
                    queue.append((nx, ny, 1))
                # 다음 위치가 벽이 아닌 이동할 수 있는 칸의 경우
                elif graph[nx][ny] == 0 and not visited[nx][ny][z]:
                    visited[nx][ny][z] = visited[x][y][z] + 1
                    queue.append((nx, ny, z))
    return -1
print(bfs(0, 0, 0))
'''

'''
### 2644 촌수계산
# dfs
import sys
input = sys.stdin.readline

n = int(input())
a, b = map(int, input().split())
graph = [[] for _ in range(n+1)]
visited = [0] * (n+1)
m = int(input())
for _ in range(m):
    x, y = map(int, input().split())
    graph[x].append(y)
    graph[y].append(x)
def dfs(node):
    for i in graph[node]:
        if not visited[i]:
            visited[i] = visited[node] + 1
            dfs(i)
dfs(a)
if visited[b] > 0:
    print(visited[b])
else:
    print(-1)

# bfs
from collections import deque
import sys
input = sys.stdin.readline

n = int(input())
a, b = map(int, input().split())
m = int(input())
graph = [[] for _ in range(n+1)]
visited = [0] * (n+1)
for _ in range(m):
    x, y = map(int, input().split())
    graph[x].append(y)
    graph[y].append(x)
def bfs(node):
    queue = deque()
    queue.append(node)
    while queue:
        node = queue.popleft()
        for i in graph[node]:
            if not visited[i]:
                visited[i] = visited[node] + 1
                queue.append(i)
bfs(a)
if visited[b] > 0:
    print(visited[b])
else:
    print(-1)
'''

'''
### 7562 나이트의 이동
import sys
input = sys.stdin.readline
from collections import deque

dx = [-2, -1, 1, 2, 2, 1, -1, -2]
dy = [1, 2, 2, 1, -1, -2, -2, -1]
def bfs():
    queue = deque()
    queue.append((sx, sy))
    while queue:
        x, y = queue.popleft()
        if x == ex and y == ey:
            return graph[x][y]
        for i in range(8):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0 <= nx < l and 0 <= ny < l and not graph[nx][ny]:
                queue.append((nx, ny))
                graph[nx][ny] = graph[x][y] + 1
t = int(input())
for _ in range(t):
    l = int(input())
    graph = [[0] * l for _ in range(l)]
    visited = [[0] * l for _ in range(l)]
    sx, sy = map(int, input().split())
    ex, ey = map(int, input().split())
    print(bfs())
'''

'''
### 1389 케빈 베이커의 6단계 법칙
import sys
input = sys.stdin.readline
from collections import deque

n, m = map(int, input().split())
graph = [[] for _ in range(n+1)]
for  _ in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)
def bfs(start):
    number = [0] * (n+1)
    queue = deque()
    queue.append(start)
    visited[start] = 1
    while queue:
        x = queue.popleft()
        for i in graph[x]:
            if not visited[i]:
                queue.append(i)
                visited[i] = 1
                number[i] = number[x] + 1
    return sum(number)
answer = []
for i in range(1, n+1):
    visited = [0] * (n+1)
    answer.append(bfs(i))
print(answer.index(min(answer)) + 1)
'''


'''
### 1107 리모컨
import sys
input = sys.stdin.readline

n = int(input())
m = int(input())
broken = list(map(int, input().split()))
# 초기값 +, - 2가지 경우로 이동만 하는 경우
answer = abs(100-n)

for number in range(10**6 + 1):
    number = str(number)
    for j in range(len(number)):
        if int(number[j]) in broken:
            break
        elif j == len(number) - 1:
            answer = min(answer, abs(int(number) - n) + len(number))
print(answer)
'''

# 4/4
'''
### 2559 수열
import sys
input = sys.stdin.readline

n, k = map(int, input().split())
temperature = list(map(int, input().split()))

value = sum(temperature[:k])
answer = value
for i in range(k, n):
    value += temperature[i] - temperature[i-k]
    answer = max(answer, value)
print(answer)
'''

'''
### 11728 배열 합치기
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

answer = a + b
answer.sort()
print(*answer)
'''

'''
### 2470 두 용액
import sys
input = sys.stdin.readline

n = int(input())
li = sorted(list(map(int, input().split())))

left = 0
right = n - 1

value = 2e9 + 1
answer = []

while left < right:
    left_val =  li[left]
    right_val = li[right]
    total = left_val + right_val
    if abs(total) < value:
        value = abs(total)
        answer = [left_val, right_val]
    if total < 0:
        left += 1
    else:
        right -= 1
print(*answer)
'''

'''
# 4/5
### 1238 파티
import sys
input = sys.stdin.readline
import heapq

n, m, x = map(int, input().split())
graph = [[] for _ in range(n+1)]

for _ in range(m):
    start, end, time = map(int, input().split())
    graph[start].append((end, time))

def dijkstra(start):
    total_time = [int(1e9)] * (n + 1)
    heap = []
    heapq.heappush(heap, (start, 0))
    total_time[start] = 0
    while heap:
        node, time = heapq.heappop(heap)
        if total_time[node] < time:
            continue
        for next_node, next_time in graph[node]:
            if time + next_time < total_time[next_node]:
                total_time[next_node] = time + next_time
                heapq.heappush(heap, (next_node, time + next_time))
    return total_time

dist = [[]]
for i in range(1, n+1):
    dist.append(dijkstra(i))
print(dist)
answer = []
for i in range(1, n+1):
    answer.append(dist[i][x] + dist[x][i])
print(max(answer))
'''

'''
### 1504 특정한 최단 경로
import sys
input = sys.stdin.readline
import heapq

n, e = map(int, input().split())
graph = [[] for _ in range(n+1)]
for _ in range(e):
    a, b, c = map(int, input().split())
    graph[a].append((b, c))
    graph[b].append((a, c))

def dijkstra(start, end):
    distance = [int(1e9)] * (n+1)
    distance[start] = 0
    heap = []
    heapq.heappush(heap, (start, 0))
    while heap:
        node, dist = heapq.heappop(heap)
        if distance[node] < dist:
            continue
        for next_node, next_dist in graph[node]:
            if dist + next_dist < distance[next_node]:
                distance[next_node] = dist + next_dist
                heapq.heappush(heap, (next_node, dist + next_dist))
    return distance[end]
v1, v2 = map(int, input().split())

path1 = dijkstra(1, v1) + dijkstra(v1, v2) + dijkstra(v2, n)
path2 = dijkstra(1, v2) + dijkstra(v2, v1) + dijkstra(v1, n)

if path1 >= int(1e9) and path2 >= int(1e9):
    print(-1)
else:
    print(min(path1, path2))
'''


# 4/6
'''
### 13164 행복 유치원
import sys
input = sys.stdin.readline

n, k = map(int, input().split())
height = list(map(int, input().split()))

answer = []
for i in range(n-1):
    answer.append(height[i+1] - height[i])
answer.sort()
print(sum(answer[:n-k]))
'''

'''
### 19939 박 터뜨리기
import sys
input = sys.stdin.readline

n, k = map(int, input().split())
ball = [i for i in range(1, n+1)]

min_ball = k * (k+1) / 2
if min_ball > n:
    print(-1)
elif (n-min_ball) % k == 0:
    print(k-1)
else:
    print(k)
    
'''

'''
### 21758 꿀 따기
import sys
input = sys.stdin.readline

n = int(input())
honey = list(map(int, input().split()))
result = []
result.append(honey[0])
for i in range(1, n):
    result.append(result[i-1] + honey[i])
print(result)
answer = 0
# case1 꿀통 오른쪽 끝, 왼쪽 끝 벌, i번째 벌 위치
for i in range(1, n-1):
    answer = max(answer, result[n-1] - honey[0] - honey[i] + result[n-1] - result[i])
# case2 꿀통 왼쪽끝, i번째 벌, 오른쪽 끝 벌 위치
for i in range(1, n-1):
    answer = max(answer, result[n-2] - honey[i] + result[i-1])
# case3 꿀통 가운데, 왼쪽 끝, 오른쪽 끝 벌 위치
for i in range(1, n-1):
    answer = max(answer, result[n-2] - honey[0] + honey[i])
print(answer)
'''

###