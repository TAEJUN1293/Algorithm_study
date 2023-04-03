'''
for i in range(10):
    print(i, )
else:
    print('Finish')


decimal = 10
result = ''
while decimal > 0:
    remainder = decimal % 2
    decimal = decimal // 2
    result = str(remainder) + result
print(result)


def f():
    global s
    s = 'I love London'
    print(s)

s = 'I love Paris'
f()
print(s)


from collections import Counter
text = """A press release is the quickest and easiest way to get free publicity. If 
well written, a press release can result in multiple published articles about your 
firm and its products. And that can mean new prospects contacting you asking you to 
sell to them. ….""".lower().split()
print(Counter(text))
print(Counter(text)["a"])


cities = ['Seoul', 'Busan', 'Jeju']

iter_obj = iter(cities)

print(next(iter_obj))
print(next(iter_obj))
print(next(iter_obj))

def generator_list(value):
    result = []
    for i in range(value):
        yield i

def asterisk_test(a, *args):
    print(a, args)
    print(type(args))
asterisk_test(1, (2,3,4,5,6))
'''
'''


### OOP 예제

class Note(object):
    def __init__(self, content):
        self.content = content
    def write_content(self, content):
        self.content = content
    def remove_all(self):
        self.content = ''
    def __add__(self, other):
        return self.content + other.content
    def __str__(self):
        return self.content


class NoteBook(object):
    def __init__(self, title):
        self.title = title
        self.page_number = 1
        self.notes = {}

    def add_note(self, note, page = 0):
        if self.page_number < 300:
            if page == 0:
                self.notes[self.page_number] = note
                self.page_number += 1
            else:
                self.notes = {page : note}
                self.page_number += 1
        else:
            print('Page가 모두 채워졌습니다.')

    def remove_note(self, page_number):
        if page_number in self.notes.keys():
            return self.notes.pop(page_number)
        else:
            print('해당 페이지는 존재하지 않습니다.')
    def get_number_of_pages(self):
        return len(self.notes.keys())



### closure function 간단하게
def star(func):
    def inner(*args, **kwargs):
        print('*'*30)
        func(*args, **kwargs)
        print('*' * 30)
    return inner
@star
def printer(msg):
    print(msg)
printer('Hello')


def generate_power(exponent):
    def wrapper(f):
        def inner(*args):
            result = f(*args)
            return exponent ** result
        return inner
    return wrapper

@generate_power(2)
def raise_two(n):
    return n**2

print(raise_two(7))
'''

'''
## try ~ except ~ else
for i in range(10):
    try:
        result = 10 / i
    except ZeroDivisionError :
        print('Not divided by 0')
    else:
        print(10/i)

### try ~ except ~ finally
try:
    for i in range(1, 10):
        result = 10 // i
        print(result)
except ZeroDivisionError:
    print('Not divided by 0')
finally:
    print('종료되었습니다')
'''
'''
while True:
    value = input('Input number to convert')
    for digit in value:
        if digit not in '0123456789':
            raise ValueError('Do not input the number')
    print('Converted integer Number -', int(value)) 
'''
'''
def get_binary_number(decimal_number):
    assert isinstance(decimal_number, int)
    return bin(decimal_number)
print(get_binary_number(10))
'''
'''
li = []
for i in range(5):
    li.append(i)
print(li.pop())
if li:
    print(li)
else:
    print(1)
    '''
'''
# BOJ 2563 색종이
import sys
input = sys.stdin.readline

whiteboard = [[0 for _ in range(100)] for _ in range(100)]
n = int(input())
for _ in range(n):
    x, y = map(int, input().split())
    for i in range(x, x + 10):
        for j in range(y, y + 10):
            whiteboard[i][j] = 1
answer = 0
for i in range(100):
    for j in range(100):
        if whiteboard[i][j]:
            answer += 1
print(answer)
'''
'''
# 2752 세수정렬
num = sorted(list(map(int, input().split())))
print(num[0], num[1], num[2])
'''

'''
#2752 
n = input()
print(int(n, 16))
'''
'''
### 2167 2차원 배열의 합 #1
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
matrix = []
for i in range(n):
    matrix.append(list(map(int, input().split())))

k = int(input())
for _ in range(k):
    i, j, x, y = map(int, input().split())
    answer = 0
    for w in range(i-1, x):
        for z in range(j-1, y):
            answer += matrix[w][z]
    print(answer)
    
'''
### 2167 2차원 배열의 합 #2 dp
'''
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(n)]
dp = [[0 for _ in range(m+1)] for _ in range(n+1)]
for i in range(1, n+1):
    for j in range(1, m+1):
        dp[i][j] = matrix[i-1][j-1] + dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1]

k = int(input())
for _ in range(k):
    i,j,x,y = map(int, input().split())
    answer = 0
    answer += dp[x][y] - dp[i-1][y] - dp[x][j-1] + dp[i-1][j-1]
    print(answer)
'''

### 2573 빙산
'''
from collections import deque
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
iceberg = [list(map(int, input().split())) for _ in range(n)]
iceland = []
for i in range(n):
    for j in range(m):
        if iceberg[i][j]:
            iceland.append((i, j))
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

def bfs(x, y):
    queue = deque([(x, y)])
    visited[x][y] = 1
    sea_list = []
    while queue:
        x, y = queue.popleft()
        sea = 0
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0<=nx<n and 0<=ny<m:
                if not iceberg[nx][ny]:
                    sea += 1
                elif iceberg[nx][ny] and not visited[nx][ny]:
                    queue.append((nx, ny))
                    visited[nx][ny] = 1
        if sea > 0:
            sea_list.append((x, y, sea))
    for x, y, sea in sea_list:
        iceberg[x][y] = max(iceberg[x][y] - sea, 0)
    return 1

year = 0
while iceland:
    visited = [[0]*m for _ in range(n)]
    melting = []
    cnt = 0
    for i,j in iceland:
        if iceberg[i][j] and not visited[i][j]:
            cnt += bfs(i, j)
        if iceberg[i][j] == 0:
            melting.append((i, j))
    if cnt > 1:
        print(year)
        break
    iceland = sorted(list(set(iceland) - set(melting)))
    year += 1
if cnt < 2:
    print(0)
'''


## 25683 감시
'''
import sys
input = sys.stdin.readline
import copy

n, m = map(int, input().split())
# 북동남서
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]
cctv_direction = [
    [],
    [[0], [1], [2], [3]],
    [[0, 2], [1,3]],
    [[0,1], [1,2], [2,3], [3,0]],
    [[0,1,2], [1,2,3], [2,3,0], [3,0,1]],
    [[0,1,2,3]],]

cctv_loc = []
startlink = []

for i in range(n):
    startlink.append(list(map(int, input().split())))
    for j in range(m):
        if startlink[i][j] in [1, 2, 3, 4, 5]:
            cctv_loc.append([i, j, startlink[i][j]])
def watching(x, y, startlink, cctv_direction):
    for i in cctv_direction:
        nx = x
        ny = y
        while True:
            nx += dx[i]
            ny += dy[i]
            if nx<0 or nx>=n or ny<0 or ny>=m:
                break
            if startlink[nx][ny] == 6:
                break
            elif startlink[nx][ny] == 0:
                startlink[nx][ny] = '#'
answer = 1e9
def dfs(idx, startlink):
    global answer
    if idx == len(cctv_loc):
        count = 0
        for i in range(n):
            count += startlink[i].count(0)
        answer = min(answer, count)
        return
    tmp = copy.deepcopy(startlink)
    x, y, num = cctv_loc[idx]
    for i in cctv_direction[num]:
        watching(x, y, tmp, i)
        dfs(idx + 1, tmp)
        tmp = copy.deepcopy(startlink)
dfs(0, startlink)
print(answer)
'''


### 1244 스위치 켜고 끄기
'''
import sys
input = sys.stdin.readline

n = int(input())
switch = list(map(int, input().split()))
students = int(input())
for _ in range(students):
    gender, num = map(int, input().split())
    if gender == 1:
        i = num
        while num < n+1:
            if switch[num-1] == 0:
                switch[num-1] = 1
            else:
                switch[num-1] = 0
            num += i
    else:
        if switch[num-1] == 0:
            switch[num-1] = 1
        else:
            switch[num-1] = 0
        i = num-2
        j = num
        while i >= 0 and j < n:
            if switch[i] == switch[j]:
                if switch[i] == 1:
                    switch[i], switch[j] = 0, 0
                else:
                    switch[i], switch[j] = 1, 1
            else:
                break
            i -= 1
            j += 1
for i in range(0, n, 20):
    print(*switch[i:i+20])

li = []
for i in range(1, 101):
    li.append(i)
for i in range(0, 101, 20):
    print(*li[i:i+20])
'''

'''
### 1051 숫자 정사각형
n, m = map(int, input().split())
rectangular = []
for _ in range(n):
    rectangular.append(list(input()))
v = min(n, m)
answer = 0
for i in range(n):
    for j in range(m):
        for k in range(v):
            if i+k < n and j+k < m:
                if rectangular[i][j] == rectangular[i+k][j] == rectangular[i][j+k] == rectangular[i+k][j+k]:
                    answer = max(answer, (k+1)**2)
print(answer)
'''
'''
### 2503 숫자야구
from itertools import permutations

n = int(input())
number = list(permutations(['1','2','3','4','5','6','7','8','9'], 3))
for _ in range(n):
    n, s, b = map(int, input().split())
    n = list(str(n))
    k = 0
    for i in range(len(number)):
        strike, ball = 0, 0
        i -= k
        for j in range(3):
            if number[i][j] == n[j]:
                strike += 1
            elif n[j] in number[i]:
                ball += 1
        if s != strike or b != ball:
            number.remove(number[i])
            k += 1
print(len(number))
'''
'''
### 10973 이전 순열
import sys
input = sys.stdin.readline

n = int(input())
num = list(map(int, input().split()))
for i in range(n-1, 0, -1):
    if num[i-1] > num[i]:
        for j in range(n-1, 0, -1):
            if num[i-1] > num[j]:
                num[i-1], num[j] = num[j], num[i-1]
                num = num[:i] + sorted(num[i:], reverse = True)
                print(*num)
                exit()
print(-1)
'''

'''
### 1913 달팽이
import sys
input = sys.stdin.readline

# input
n = int(input())
k = int(input())
graph = [[0 for _ in range(n)] for _ in range(n)]

# direction 상하좌우
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
# 초기값 location, value
x, y = n//2, n//2
graph[x][y] = 1
# 다음 값, graph크기
number = 2
graph_size = 3

while x!=0 or y!=0:
    while number <= graph_size**2:
        if x == n//2 and y == n//2:
            up, down, left, right = graph_size, graph_size-1, graph_size-1, graph_size-2
            x += dx[0]
            y += dy[0]
            up -= 1
        elif right > 0:
            x += dx[3]
            y += dy[3]
            right -= 1
        elif down > 0:
            x += dx[1]
            y += dy[1]
            down -= 1
        elif left > 0:
            x += dx[2]
            y += dy[2]
            left -= 1
        elif up > 0:
            x += dx[0]
            y += dy[0]
        graph[x][y] = number
        number += 1
    graph_size += 2
    n -= 2

for i in range(len(graph)):
    for j in range(len(graph)):
        if graph[i][j] == k:
            find_x, find_y = i, j
        print(graph[i][j], end = ' ')
    print()
print(find_x+1, find_y+1)

'''
'''
### 1697 숨바꼭질
from collections import deque
n, k = map(int, input().split())
li = [0] * 100001

def bfs(x):
    queue = deque([x])
    while queue:
        x = queue.popleft()
        if x == k:
            return li[x]
        for i in (x-1, x+1, 2*x):
            if 0 <= i <= 100000 and not li[i]:
                queue.append(i)
                li[i] = li[x] + 1
print(bfs(n))
'''

'''
### 11724 연결 요소의 개수 #1 BFS
from collections import deque
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
graph = [[] for _ in range(n+1)]
visited = [0] * (n+1)
for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

def bfs(start):
    queue = deque([start])
    visited[start] = 1
    while queue:
        node = queue.popleft()
        for i in graph[node]:
            if not visited[i]:
                visited[i] = 1
                queue.append(i)
answer = 0
for i in range(1, n+1):
    if not visited[i]:
        if graph[i]:
            bfs(i)
            answer += 1
        else:
            answer += 1
            visited[i] = 1
print(answer)

'''
'''
### 11724 연결 요소의 개수 #1 DFS
import sys
sys.setrecursionlimit(50000)
input = sys.stdin.readline

n, m = map(int, input().split())
graph = [[] for _ in range(n+1)]
visited = [0] * (n+1)

for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

def dfs(node):
    visited[node] = 1
    for next_node in graph[node]:
        if not visited[next_node]:
            dfs(next_node)
answer = 0
for i in range(1, n+1):
    if not visited[i]:
        dfs(i)
        answer += 1
print(answer)
'''

'''
### 그래프 (방향 그래프 구현)
# 그래프 객체를 표현하는 클래스 
class dir_graph:
    # 생성자
    def __init__(self, edges, nodes):
        # 인접 목록에 대한 메모리 할당
        self.adj = [[] for _ in range(nodes)]
        # 방향 그래프에 간선 추가
        for u, v in edges:
            # 인접 목록의 노드를 u에서 v로 할당 
            self.adj[u].append(v)
            # 양방향이라면 아래 코드 추가
            self.adj[v].append(u)
# 그래프 잘 만들어졌는지 출력  
def expression(graph):
    for u in range(len(graph.adj)):
        # 현 노드와 인접한 모든 노드 출력
        for v in graph.adj[u]:
            print(f'({u} -> {v})', end = ' ')
            print()
            
if __name__ == '__main__':
    # 그래프 간선 
    edges = [(0, 1), (1, 2), (0, 2), (3, 2), (4, 5), (6, 5)]
    # 그래프 노드 (0 ~ 7)
    nodes = 7
    # 입력된 nodes, edges로 그래프 구성 
    graph = dir_graph(edges, nodes)
    # 그래프 잘 구현되었는지 출력
    expression(graph)
'''

'''
### 그래프 (가중치를 포함한 방향 그래프)
# 그래프 객체 구현
class weight_graph:
    # 생성자
    def __init__(self, edges, nodes):
        # 인접 리스트를 나타냄
        self.adj = [None] * nodes
        # 인접 리스트에 대해 메모리 할당
        for i in range(nodes):
            self.adj[i] = []
        # 간선 추가
        for u, v, w in edges:
            self.adj[u].append((v, w))
            # 양방향일 경우 아래 코드 추가
            self.adj[v].append((u, w))
# 생성한 그래프 잘 구현되었는지 확인
def expression(graph):
    for u in range(len(graph.adj)):
        # 현 노드와 인접한 모든 노드 인쇄
        for v, w in graph.adj[u]:
            print(f'({u} -> {v}, {w})', end = ' ')
            print()
if __name__ == '__main__':
    # 가중 방향 그래프의 간선 입력(현 노드, 다음 노드, 가중 ex 비용)
    edges = [(0, 1, 6), (1, 2, 7), (0, 2, 5), (3, 2, 4), (4, 5, 1), (6, 5, 2)]
    # 노드 수
    nodes = 7
    graph = weight_graph(edges, nodes)
    expression(graph)
'''

'''
### BFS
from collections import deque

# BFS 함수 정의
def bfs(graph, start, visited):
    # 큐(Queue) 구현을 위해 deque 라이브러리 사용
    queue = deque([start])
    # 현재 노드를 방문 처리
    visited[start] = True
    # 큐가 빌 때까지 반복
    while queue:
        # 큐에서 하나의 원소를 뽑아 출력
        v = queue.popleft()
        print(v, end=' ')
        # 해당 원소와 연결된, 아직 방문하지 않은 원소들을 큐에 삽입
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True

# 각 노드가 연결된 정보를 리스트 자료형으로 표현(2차원 리스트)
graph = [
  [],
  [2, 3, 8],
  [1, 7],
  [1, 4, 5],
  [3, 5],
  [3, 4],
  [7],
  [2, 6, 8],
  [1, 7]
]

# 각 노드가 방문된 정보를 리스트 자료형으로 표현(1차원 리스트)
visited = [False] * 9

# 정의된 BFS 함수 호출
bfs(graph, 1, visited)
'''
'''
### 백트래킹
def recursive_dfs(v, discovered=[]):
    discovered.append(v)
    for w in graph[v]:
        if not w in discovered:
            ``` 가능성이 있는 경우에만 진행 ```
            if 가능성:
                 discovered = recursive_dfs(w, discovered)
            # 방문 노드를 계속해서 업데이트
    return discovered

'''

'''
### 9663 N-Queen
import sys
input = sys.stdin.readline

n = int(input())
chess = [0] * n
answer = 0

def promising(rows):
    for i in range(rows):
        if chess[rows] == chess[i] or abs(chess[rows] - chess[i]) == abs(rows-i):
            return False
    return True

def Nqueen(rows):
    global answer
    if rows == n:
        answer += 1
        return
    else:
        for i in range(n):
            chess[rows] = i
            if promising(rows):
                Nqueen(rows+1)
Nqueen(0)
print(answer)
'''

'''
### 1987 알파벳
import sys
input = sys.stdin.readline

rows, cols = map(int, input().split())
graph = [list(input().rstrip()) for _ in range(rows)]
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
answer = 1

def bfs(x, y):
    global answer
    # 중복 제거를 위해 deque 대신 set 사용
    queue = set([(x, y, graph[x][y])])
    while queue:
        # 말 현재 칸 뽑기
        x, y, alpha = queue.pop()
        # 말이 지나가는 칸 초기화
        answer = max(len(alpha), answer)
        for i in range(4):
            # 상하좌우 4방향 탐색
            nx = x + dx[i]
            ny = y + dy[i]
            # 주어진 범위 내
            if 0<=nx<rows and 0<=ny<cols:
                # 다음 칸에 놓인 알파벳이 지금까지 지난 알파벳에 포함이 안되면
                if graph[nx][ny] not in alpha:
                    # set 집합에 추가
                    queue.add((nx, ny, alpha + graph[nx][ny]))
bfs(0, 0)
print(answer)
'''

'''
### 2589 보물섬
from collections import deque
import sys
input = sys.stdin.readline

l, w = map(int, input().split())
maps = [list(input().rstrip()) for _ in range(l)]
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

def bfs(i, j):
    queue = deque()
    queue.append((i, j))
    visited = [[0] * w for _ in range(l)]
    visited[i][j] = 1
    total = 0
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0>nx or l<=nx or 0>ny or w<=ny:
                continue
            elif maps[nx][ny] == 'L' and not visited[nx][ny]:
                visited[nx][ny] = visited[x][y] + 1
                total = max(total, visited[nx][ny])
                queue.append((nx, ny))
    return total-1
answer = 0
for i in range(l):
    for j in range(w):
        if maps[i][j] == 'L':
            answer = max(answer, bfs(i, j))
print(answer)
'''

'''
### 2468 안전 영역
from collections import deque
import sys
input = sys.stdin.readline

n = int(input())
# 최고 높이 찾기 위해 변수 선언
max_height = 0
graph = []
for i in range(n):
    graph.append(list(map(int, input().split())))
    for j in range(n):
        if graph[i][j] > max_height:
            max_height = graph[i][j]
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


def bfs(i, j, rain, visited):
    queue = deque()
    queue.append((i, j))
    visited[i][j] = 1
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0<=nx<n and 0<=ny<n:
                if graph[nx][ny] > rain and not visited[nx][ny]:
                    visited[nx][ny] = 1
                    queue.append((nx, ny))
answer = 0
for i in range(max_height):
    visited = [[0]*n for _ in range(n)]
    group = 0
    for j in range(n):
        for k in range(n):
            if graph[j][k] > i and not visited[j][k]:
                bfs(j, k, i, visited)
                group += 1
    answer = max(answer, group)
print(answer)
'''

'''
### 2583 안전영역
from collections import deque
import sys
input = sys.stdin.readline

m, n, k = map(int, input().split())
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
visited = [[0]*n for _ in range(m)]
for _ in range(k):
    x1, y1, x2, y2 = map(int, input().split())
    for j in range(y1, y2):
        for k in range(x1, x2):
            visited[j][k] = 1

def bfs(i, j):
    queue = deque()
    queue.append((i, j))
    visited[i][j] = 1
    size = 1
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0<=nx<m and 0<=ny<n and not visited[nx][ny]:
                visited[nx][ny] = 1
                queue.append((nx, ny))
                size += 1
    return size

answer = []
for i in range(m):
    for j in range(n):
        if visited[i][j] == 0:
            answer.append(bfs(i, j))
answer.sort()
print(len(answer))
print(*answer, end = ' ')
'''

'''
### 2636 치즈
import sys
input = sys.stdin.readline
from collections import deque

n, m = map(int, input().split())
graph = []
for _ in range(n):
    graph.append(list(map(int, input().split())))

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
answer = []

def bfs():
    queue = deque()
    queue.append([0, 0])
    visited[0][0] = 1
    cheese = 0
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny]:
                if graph[nx][ny] == 0:
                    visited[nx][ny] = 1
                    queue.append((nx, ny))
                elif graph[nx][ny] == 1:
                    visited[nx][ny] = 1
                    graph[nx][ny] = 0
                    cheese += 1
    answer.append(cheese)
    return cheese
time = 0
while True:
    time += 1
    visited = [[0]*m for _ in range(n)]
    cheese = bfs()
    if cheese == 0:
        break
print(time - 1)
print(answer[-2])
'''

'''
### 7576 토마토
from collections import deque
import sys
input = sys.stdin.readline

m, n = map(int, input().split())
graph = [list(map(int, input().split())) for _ in range(n)]
queue = deque()
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

for i in range(n):
    for j in range(m):
        if graph[i][j] == 1:
            queue.append((i, j))
def bfs():
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0<=nx<n and 0<=ny<m:
                if graph[nx][ny] == 0:
                    graph[nx][ny] = graph[x][y] + 1
                    queue.append((nx, ny))
day = 0
bfs()
for i in graph:
    for j in i:
        if j == 0:
            print(-1)
            exit(0)
    day = max(day, max(i))
print(day-1)
'''

'''
### 7569 토마토
from collections import deque
import sys
input = sys.stdin.readline

m, n, h = map(int, input().split())
tomato = [[list(map(int, input().rstrip().split())) for _ in range(n)] for _ in range(h)]
visited = [[[False]*m for _ in range(n)] for _ in range(h)]
queue = deque()
# 앞뒤 상하 좌우
dx = [-1, 1, 0, 0, 0, 0]
dy = [0, 0, -1, 1, 0, 0]
dz = [0, 0, 0, 0, -1, 1]
day = 0

def bfs():
    while queue:
        x, y, z = queue.popleft()
        for i in range(6):
            nx = x+dx[i]
            ny = y+dy[i]
            nz = z+dz[i]
            if 0<=nx<h and 0<=ny<n and 0<=nz<m and visited[nx][ny][nz] == False:
                if tomato[nx][ny][nz] == 0:
                    queue.append((nx, ny, nz))
                    tomato[nx][ny][nz] = tomato[x][y][z] + 1
                    visited[nx][ny][nz] = True

for i in range(h):
    for j in range(n):
        for r in range(m):
            if tomato[i][j][r] == 1 and visited[i][j][r] == False:
                queue.append((i,j,r))
                visited[i][j][r] = True
bfs()
for i in tomato:
    for j in i:
        for r in j:
            if r == 0:
                print(-1)
                exit(0)
        day = max(day, max(j))
print(day-1)

import sys
input = sys.stdin.readline

n = int(input())
graph = []
for _ in range(n):
    graph.append(list(map(int, input().rstrip().split())))
baby_size = 2
baby_x, baby_y = 0, 0

for i in range(n):
    for j in range(n):
        if graph[i][j] == 9:
            baby_x, baby_y = i, j
            graph[baby_x][baby_y] = 0
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

from collections import deque
def bfs():
    dist = [[-1]*n for _ in range(n)]
    queue = deque([(baby_x, baby_y)])
    dist[baby_x][baby_y] = 0
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0<=nx<n and 0<=ny<n:
                if graph[nx][ny]<=baby_size and dist[nx][ny] == -1:
                    queue.append((nx, ny))
                    dist[nx][ny] = dist[x][y] + 1
    return dist

def find(dist):
    x, y = 0, 0
    mini = 1e9
    for i in range(n):
        for j in range(n):
            if dist[i][j] != -1 and 1 <= graph[i][j] < baby_size:
                if dist[i][j] < mini :
                    x, y = i, j
                    mini = dist[i][j]
    if mini == 1e9:
        return None
    else:
        return x, y, mini

result = 0
ate = 0
while True:
    value = find(bfs())
    if value == None:
        print(result)
        break
    else:
        baby_x, baby_y = value[0], value[1]
        result += value[2]
        graph[baby_x][baby_y] = 0
        ate += 1
    if ate >= baby_size:
        baby_size += 1
        ate = 0
'''

'''
### 16236 아기 상어
from collections import deque
import sys
input = sys.stdin.readline

n = int(input())
graph = []
for i in range(n):
    row = list(map(int, input().split()))
    graph.append(row)
    for j in range(n):
        if row[j] == 9:
            graph[i][j] = 2
            start = [i, j]
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

def bfs(i, j):
    visited = [[0]*n for _ in range(n)]
    visited[i][j] = 1
    queue = deque()
    queue.append((i, j))
    dist = [[0]*n for _ in range(n)]
    eat = []
    while queue:
        x, y = queue.popleft()
        for d in range(4):
            nx = x + dx[d]
            ny = y + dy[d]
            if 0<=nx<n and 0<=ny<n and not visited[nx][ny]:
                if not graph[nx][ny] or graph[nx][ny] <= graph[i][j]:
                    visited[nx][ny] = 1
                    dist[nx][ny] = dist[x][y] + 1
                    queue.append((nx, ny))
                if graph[nx][ny] < graph[i][j] and graph[nx][ny]:
                    eat.append((nx, ny, dist[nx][ny]))
    if not eat:
        return -1, -1, -1
    eat.sort(key = lambda x: (x[2], x[0], x[1]))
    return eat[0][0], eat[0][1], eat[0][2]

answer = 0
size = 0
while True:
    i, j = start[0], start[1]
    x, y, dist = bfs(i, j)
    if x == -1:
        break
    graph[x][y] = graph[i][j]
    graph[i][j] = 0
    start = [x, y]
    size += 1
    if size == graph[x][y]:
        size = 0
        graph[x][y] += 1
    answer += dist
print(answer)
'''

'''
### 13549 숨바꼭질 3
from collections import deque
import sys
input = sys.stdin.readline

n, k = map(int, input().split())

def bfs():
    visited = [0] * 100001
    visited[n] = 1
    queue = deque()
    queue.append(n)
    while queue:
        x = queue.popleft()
        if x == k:
            print(visited[k] - 1)
            break
        for next in (x-1, x+1, 2*x):
            if 0 <= next < 100001 and visited[next] == 0:
                if 2*x == next:
                    visited[next] = visited[x]
                    queue.appendleft(next)
                else:
                    visited[next] = visited[x] + 1
                    queue.append(next)
bfs()
'''
'''
### 11403 경로찾기 플로이드-워셜
import sys
input = sys.stdin.readline

n = int(input())
graph = [list(map(int, input().split())) for _ in range(n)]
for i in range(n):
    for j in range(n):
        for k in range(n):
            if graph[j][i] and graph[i][k]:
                graph[j][k] = 1
for i in graph:
    print(*i)
'''
'''
### 11403 BFS

from collections import deque

n = int(input())
graph = [list(map(int, input().split())) for _ in range(n)]
visited = [[0]*n for _ in range(n)]

def bfs(x):
    queue = deque()
    queue.append(x)
    check = [0 for _ in range(n)]
    while queue:
        p = queue.popleft()
        for i in range(n):
            if check[i] == 0 and graph[p][i] == 1:
                queue.append(i)
                check[i] = 1
                visited[x][1] = 1
for i in range(0, n):
    bfs(i)
for i in visited:
    print(*i)
'''


'''
### 11403 dfs
n = int(input())
graph = [list(map(int, input().split())) for _ in range(n)]
visited = [0 for _ in range(n)]

def dfs(x):
    for i in range(n):
        if graph[x][i] == 1 and not visited[i]:
            visited[i] = 1
            dfs(i)

for i in range(n):
    dfs(i)
    for j in range(n):
        if visited[j] == 1:
            print(1, end = ' ')
        else:
            print(0, end = ' ')
    print()
    visited = [0 for _ in range(n)]
'''

'''
### 4963 섬의 개수
from collections import deque

dx = [-1, 1, 0, 0, 1, -1, -1, 1]
dy = [0, 0, -1, 1, -1, -1, 1, 1]

def bfs(i, j):
    queue = deque()
    queue.append((i, j))
    while queue:
        x, y = queue.popleft()
        for k in range(8):
            nx = x + dx[k]
            ny = y + dy[k]
            if 0<=nx<h and 0<=ny<w and graph[nx][ny] == 1:
                queue.append((nx, ny))
                graph[nx][ny] = 0
while True:
    w,h = map(int,input().split())
    if w == 0 and h == 0 :
        break
    graph = []
    for _ in range(h):
        graph.append(list(map(int,input().split())))

    answer = 0
    for i in range(h):
        for j in range(w):
            if graph[i][j] == 1:
                answer += 1
                bfs(i,j)

    print(answer)
'''

'''
### 17204 죽음의 게임
import sys
input = sys.stdin.readline
from collections import deque

n, k = map(int, input().split())
graph = [int(input()) for _ in range(n)]
visited = [0] * n

def bfs():
    queue = deque()
    queue.append((0))
    while queue:
        next = queue.popleft()
        m = graph[next]
        if next != m and not visited[m]:
            queue.append(m)
            visited[m] = visited[next] + 1
            if m == k:
                return
bfs()
if visited[k]:
    print(visited[k])
else:
    print(-1)
'''


'''
### 11558 The Game of Death
def dfs(v):
    for i in game[v]:
        if not visited[i]:
            visited[i] = visited[v] + 1
            dfs(i)
t = int(input())
for _ in range(t):
    n = int(input())
    game = [[] for _ in range(n+1)]
    for i in range(1, n+1):
        num = int(input())
        game[i].append(num)
    print(game)
    visited = [0] * (n+1)
    dfs(1)
    if visited[n] > 0:
        print(visited[n])
    else:
        print(0)

'''
'''
cities = ["Jeju", "Pangyo", "Seoul", "NewYork", "LA", "Jeju", "Pangyo", "Seoul", "NewYork", "LA"]
print(cities.pop(0))

li = [True, False, True]
for i in li:
    if i:
        print(2)
    else:
        print(10)


li_x = ['Hong', 'kil', 'dong']
print(li_x)
print(li_x[:2])
print(li_x.index('kil'))

li_x.insert(1, 'Tae')
print(li_x)

li_x.remove('Tae')
print(li_x)
li_x.pop(1)
print(li_x)
'''
'''
### 연결리스트 구현
class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next
head = ListNode(0)
# 노드 추가
curr_node = head
new_node = ListNode(1)
curr_node.next = new_node
curr_node = curr_node.next

curr_node.next = ListNode(2)
curr_node = curr_node.next

curr_node.next = ListNode(3)
curr_node = curr_node.next

curr_node.next = ListNode(4)
curr_node = curr_node.next

# 전체 연결리스트 탐색
node = head
while node:
    print(node.val)
    node = node.next

# 탐색하며 삭제
node = head
while node.next:
    if node.next.val == 3:
        next_node = node.next.next
        node.next = next_node
        break
    node = node.next
node = head
while node:
    print(node.val)
    node = node.next
'''
'''
### 버블정렬
def bubble_sort(array):
	# 뒤부터 탐색시작
    for i in range(len(array)-1, 0, -1):
        # 최적화를 위해 진행
        check = False
    	for j in range(i):
        	if array[j] > array[j+1]:
            	array[j], array[j+1] = array[j+1], array[j]
				check = True
        # 교환 일어나지 않으면 정렬 완료
		if not check:
        	break

### 선택 정렬
def selection_sort(array):
    for i in range(len(array)-1):
        min_idx = i
        for j in range(i+1, len(array)):
            if array[j] < array[min_idx]:
                min_idx = j
        array[i], array[min_idx] = array[min_idx], array[i]

### 삽입정렬
def Insertion_sort(arr):
    for end in range(1, len(arr)):
        i = end
        while i > 0 and arr[i - 1] > arr[i]:
            arr[i - 1], arr[i] = arr[i], arr[i - 1]
            i -= 1
    return arr
arr = [10, 92, 5, 11]
print(Insertion_sort(arr))

### quick sort
def quick_sort(array):
    # 재귀함수 만들기
    def sort(low, high):
        if high <= low:
            return
        mid = partition(low, high)
        sort(low, mid - 1)
        sort(mid, high)

    def partition(low, high):
        # 가운데 값 피벗 지정
        pivot = array[(low + high) // 2]
        # 인덱스 교차하기 전까지 진행
        while low <= high:
            # 올바르게 정렬 시 + 1
            while array[low] < pivot:
                low += 1
            # 큰 값들 정렬
            while array[high] > pivot:
                high -= 1

            if low <= high:
                array[low], array[high] = array[high], array[low]
                low, high = low + 1, high - 1
        return low
    return array

print(quick_sort(arr))
'''

'''
### merge sort
def Merge_sort(array):
    if len(array) < 2:
        return array

    mid = len(array) // 2
    low_arr = Merge_sort(array[:mid])
    high_arr = Merge_sort(array[mid:])

    merged_arr = []
    l = h = 0
    while l < len(low_arr) and h < len(high_arr):
        if low_arr[l] < high_arr[h]:
            merged_arr.append(low_arr[l])
            l += 1
        else:
            merged_arr.append(high_arr[h])
            h += 1
    merged_arr += low_arr[l:]
    merged_arr += high_arr[h:]
    return merged_arr

arr = [10, 92, 5, 11]
print(Merge_sort(arr))
'''


# 힙 정렬
def Heap_sort(array):
    n = len(array)
    # heap 구성
    for i in range(n):
        c = i
        while c != 0:
            r = (c-1)//2
            if array[r] < array[c]:
                array[r], array[c] = array[c], array[r]
            c = r
    # 크기를 줄여가면서 heap 구성
    for j in range(n-1, -1, -1):
        array[0] , array[j] = array[j], array[0]
        r = 0
        c = 1
        while c<j:
            c = 2*r +1
            # 자식 중 더 큰 값 찾기
            if (c<j-1) and (array[c] < array[c+1]):
                c += 1
            # 루트보다 자식이 크다면 교환
            if (c<j) and (array[r] < array[c]):
                array[r], array[c] = array[c], array[r]
            r=c
    return array
array = [1, 10, 5, 8, 7, 6, 4, 3, 2, 9]
print(Heap_sort(array))

## 이진탐색
# sorted_list : 이미 정렬된 배열
# num : 찾고자 하는 값
def binary_search(sorted_list, num):
    left = 0
    right = len(sorted_list) - 1
    # 인덱스끼리 교차하기 전까지만
    while left <= right:
        # 가운데 인덱스 지정
        mid = (left+right) // 2
        # 찾는 값이 배열 가운데 값 보다 작으면, 끝 인덱스 변경
        if num < sorted_list[mid]:
            right = mid - 1
        # 반대의 경우 시작인덱스 변경
        elif sorted_list[mid] < num:
            left = mid + 1
        # 번호를 찾은 경우 mid 값 리턴
        else:
            return mid
    # 주어진 배열 내 찾고자하는 num 존재 X
    return -1