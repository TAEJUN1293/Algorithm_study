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

'''
# 4/7
### 2152 예산
import sys
input = sys.stdin.readline

n = int(input())
country = list(map(int, input().split()))
m = int(input())
start = 0
end = max(country)

while start <= end:
    mid = (start + end) // 2
    num = 0
    for i in country:
        if i >= mid:
            num += mid
        else:
            num += i
    if num <= m:
        start = mid + 1
    else:
        end = mid - 1
print(end)
'''

'''
### 1068 트리
import sys
input = sys.stdin.readline

n = int(input())
tree = list(map(int, input().split()))
m = int(input())

def dfs(remove_node):
    tree[remove_node] = -2
    for i in range(n):
        if tree[i] == remove_node:
            dfs(i)
dfs(m)
answer = 0
for i in range(n):
    if tree[i] != -2 and i not in tree:
        answer += 1
print(answer)
'''

'''
### 1967 트리의 지름
import sys
input = sys.stdin.readline

n = int(input())
graph = [[] for _ in range(n+1)]
for _ in range(n-1):
    a, b, c = map(int, input().split())
    graph[a].append((b, c))
    graph[b].append((a, c))
distance = [-1] * (n+1)
distance[1] = 0

def dfs(node, dist):
    for next_node, d in graph[node]:
        if distance[next_node] == -1:
            distance[next_node] = dist + d
            dfs(next_node, dist + d)
dfs(1, 0)
start = distance.index(max(distance))
distance = [-1] * (n+1)
distance[start] = 0
dfs(start, 0)
print(max(distance))
'''

# 4/9
'''
### 2776 암기왕
import sys
input = sys.stdin.readline

t = int(input())

for _ in range(t):
    n = int(input())
    note1 = set(map(int, input().split()))
    m = int(input())
    note2 = list(map(int, input().split()))
    for i in note2:
        if i in note1:
            print(1)
        else:
            print(0)
'''

'''
### 2792 보석 상자
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
jewel = []
for _ in range(m):
    jewel.append(int(input()))
left = 1
right = max(jewel)
while left <= right:
    mid = (left + right) // 2
    person = 0
    for i in jewel:
        if i % mid == 0:
            person += i // mid
        else:
            person += (i//mid) + 1
    if person > n:
        left = mid + 1
    else:
        right = mid - 1
print(left)
'''

# 4/10
'''
li = [7, 12, 22, 34]
del(li[2]) > 리스트에서 원소 제거 
print(li) > 리턴값 출력 후 리스트에서 원소 제거
#print(li.pop(2))
'''

'''
def solution(park, routes):
    r,c,R,C = 0,0,len(park),len(park[0]) # r,c : S위치 / R,C : 보드경계
    move = {"E":(0,1),"W":(0,-1),"S":(1,0),"N":(-1,0)}
    for i,row in enumerate(park): # 시작점 찾기
        if "S" in row:
            r,c = i,row.find("S")
            break

    for route in routes:
        dr,dc = move[route[0]] # 입력받는 route의 움직임 방향
        new_r,new_c = r,c # new_r,new_c : route 적용 후 위치
        for _ in range(int(route[2])):
            # 한칸씩 움직이면서, 보드 안쪽이고 "X"가 아니라면 한칸이동
            if 0<=new_r+dr<R and 0<=new_c+dc<C and park[new_r+dr][new_c+dc] != "X":
                new_r,new_c = new_r+dr,new_c+dc
            else: # 아니라면 처음 위치로
                new_r,new_c = r,c
                break
        r,c = new_r,new_c # 위치 업데이트

    return [r,c]

def solution(L, x):
    answer = -1
    left = 0
    right = len(L)-1
    while right >= left:
        mid = (left+right)//2
        if L[mid] == x:
            return mid
        elif L[mid] > x:
            right = mid-1
        else:
            left = mid+1
    return answer
'''
'''
# Single LinkedList - Dummy node 구현
class Node:
    def __init__(self, item):
        self.data = item
        self.next = None

class LinkedList:
    # 객체 생성
    def __init__(self):
        self.nodeCount = 0
        self.head = None
        self.tail = None
    # 리스트를 print로 출력할 때
    def __repr__(self):
        if self.nodeCount == 0:
            return 'LinkedList: empty'
        s = ''
        curr = self.head
        while curr is not None:
            s += repr(curr.data)
            if curr.next is not None:
                s += ' -> '
            curr = curr.next
        return s
    # 연결리스트 내 pos 번째 노드 찾기
    def getAt(self, pos):
        # 범위 벗어나면 None 리턴
        if pos < 1 or pos > self.nodeCount:
            return None
        # 1부터 탐색 진행하여 찾기
        i = 1
        curr = self.head
        while i < pos:
            curr = curr.next
            i += 1
        return curr
    # 연결 리스트 내 데이터를 리스트로 반환
    def traverse(self):
        # 빈 리스트 초기화
        result = []
        # Head부터 시작하여
        curr = self.head
        # 다음 위치가 None이 아닐때까지 반복
        while curr:
            result.append(curr.data)
            curr = curr.next
        return result

    def insertAt(self, pos, newNode):
        # pos 범위 만족 못하면 False
        if pos < 1 or pos > self.nodeCount + 1:
            return False
        if pos == 1:
            newNode.next = self.head
            self.head = newNode
        else:
            if pos == self.nodeCount + 1:
                prev = self.tail
            else:
                prev = self.getAt(pos - 1)
            newNode.next = prev.next
            prev.next = newNode
        if pos == self.nodeCount + 1:
            self.tail = newNode
        self.nodeCount += 1

    def popAt(self, pos):
        # 주어진 범위 아니면 False
        if pos < 1 or pos > self.nodeCount:
            return False
        # 삭제할 노드가 Head면
        if pos == 1:
            remove_node = self.head
            self.head = self.head.next
            self.nodeCount -= 1
            if not self.nodeCount:
                self.tail = None
            return remove_node.data
        # Head가 아닌 경우
        else:
            prev = self.getAt(pos - 1)
            remove_node = prev.next
            prev.next = remove_node.next
            if pos == self.nodeCount:
                self.tail = prev
            self.nodeCount -= 1
            return remove_node.data
a = Node(67)
b = Node(34)
c = Node(28)
L = LinkedList()
print(L)
L.insertAt(1, a)
L.insertAt(2, b)
L.insertAt(1, c)
print(L)
def solution(x):
    return 0
'''
'''
# 4/11
# DoublyLinkedList 구현 (head, tail에 dummy 노드 구현)
class Node:
    def __init__(self, item):
        self.data = item
        self.prev = None
        self.next = None
# 양방향 빈 연결리스트 생성
class DoublyLinkedList:
    def __init__(self):
        self.nodeCount = 0
        self.head = Node(None)
        self.tail = Node(None)
        self.head.prev = None
        self.head.next = self.tail
        self.tail.prev = self.head
        self.tail.next = None

    def __repr__(self):
        if self.nodeCount == 0:
            return 'LinkedList: empty'
        s = ''
        curr = self.head
        while curr.next.next:
            curr = curr.next
            s += repr(curr.data)
            if curr.next.next is not None:
                s += ' -> '
        return s

    def getLength(self):
        return self.nodeCount
    # 리스트 순회 (빈 리스트는 진행 X)
    def traverse(self):
        result = []
        curr = self.head
        while curr.next.next:
            curr = curr.next
            result.append(curr.data)
        return result
    # 리스트 역순회 (빈 리스트는 진행 X)
    def reverse(self):
        result = []
        curr = self.tail
        while curr.prev.prev:
            curr = curr.prev
            result.append(curr.data)
        return result
    # pos번째 노드 (특정 원소 얻기)
    def getAt(self, pos):
        if pos < 0 or pos > self.nodeCount:
            return None
        # 효율성 극대화 위해 절반으로 나누어 이분탐색 진행
        # 노드 수 절반 기준 tail에 가까우므로 curr을 tail에 놓고 탐색
        if pos > self.nodeCount // 2:
            i = 0
            curr = self.tail
            while i < self.nodeCount - (pos - 1):
                curr = curr.prev
                i += 1
        # 노드 수 기준 head에 가까우므로 head에 놓고 탐색 진행
        else:
            i = 0
            curr = self.head
            while i < pos:
                curr = curr.next
                i += 1
        return curr
    # 특정 노드 이후에 원소 삽입
    def insertAfter(self, prev, newNode):
        next = prev.next
        newNode.prev = prev
        newNode.next = next
        prev.next = newNode
        next.prev = newNode
        self.nodeCount += 1
        return True
    # 특정 위치에 노드 삽입
    def insertAt(self, pos, newNode):
        if pos < 1 or pos > self.nodeCount + 1:
            return False
        prev = self.getAt(pos - 1)
        return self.insertAfter(prev, newNode)
    # 역방향 순회하며 노드 삽입
    def insertBefore(self, next, newNode):
        prev = next.prev
        newNode.prev = prev
        newNode.next = next
        prev.next = newNode
        next.prev = newNode
        self.nodeCount += 1
        return True
    # 이전 노드 다음 노드 삭제
    def popAfter(self, prev):
        if prev.next == self.tail:
            return None
        remove_node = prev.next
        next = remove_node.next
        prev.next = next
        next.prev = prev
        self.nodeCount -= 1
        return remove_node.data
    # next 이전 노드 삭제
    def popBefore(self, next):
        if next.prev == self.head:
            return None
        remove_node = next.prev
        prev = remove_node.prev
        prev.next = next
        next.prev = prev
        self.nodeCount -= 1
        return remove_node.data
    # 특정(pos번째) 위치 노드 삭제
    def popAt(self, pos):
        if pos < 1 or pos > self.nodeCount:
            return False
        prev = self.getAt(pos - 1)
        return self.popAfter(prev)
    # 양방향 리스트 병합
    def concat(self, L):
        self.tail.prev.next = L.head.next
        L.head.next.prev = self.tail.prev
        if L.tail:
            self.tail = L.tail
        self.nodeCount += L.nodeCount
'''

'''
# 스택 구현
import sys
import os
sys.path.append('C:/Users/wnrrh/PycharmProjects/Algorithm_study/DevCourse')
import doublylinkedlist
from doublylinkedlist import Node
from doublylinkedlist import DoublyLinkedList

class ArrayStack:
	# 빈 리스트 생성
	def __init__(self):
		self.data = []
	# 리스트 크기 구하기
	def size(self):
		return len(self.data)
	# 비어있나 여부 확인
	def isEmpty(self):
		return self.size() == 0
	# item을 리스트에 추가
	def push(self, item):
		self.data.append(item)
	# 리스트 내 젤 뒤에 원소 삭제
	def pop(self):
		return self.data.pop()
	# 제일 뒤에 원소 찾기
	def peek(self):
		return self.data[-1]
# 양방향 연결리스트로 스택 구현
class LinkedListStack:
	# 연결리스트 생성
	def __init__(self):
		self.data = DoublyLinkedList()
	# 길이 구하기
	def size(self):
		return self.data.getLength()
	# 비어있나 여부 확인
	def isEmpty(self):
		return self.size() == 0
	# 연결리스트에 item 원소 추가 (append 대신 기존 작성한 함수로 구현)
	def push(self, item):
		node = Node(item)
		self.data.insertAt(self.size() + 1, node)
	# op대신 기존에 작성한 함수로 구현 (제일 뒤에 원소 삭제)
	def pop(self):
		return self.data.popAt(self.size())
	# 기존 함수로 구현 (제일 뒤 원소 찾기)
	def peek(self):
		return self.data.getAt(self.size()).data
prec = {
    '*': 3, '/': 3,
    '+': 2, '-': 2,
    '(': 1
}
def solution(S):
    opStack = ArrayStack()
    answer = ''
    for i in S:
        # 알파벳이면 answer에 추가
        if i not in prec and i != ')':
            answer += i
        # 괄호로 시작하는 경우
        elif i == '(':
            opStack.push(i)
        # 괄호 닫는 경우
        elif i == ')':
            # opStack에 저장된 끝자리가 (가 올때까지 answer 에 연산 더해주기
            while opStack.peek() != '(':
                answer += opStack.pop()
            # (차례에서 조건 끝나면서 opStack에서 '(' pop해주기
            opStack.pop()
        # 남은 연산 *, /, +, - 인 경우
        else:
            # opStack에 (가 있으면서 연산 더 높은 *,/ 라면
            while not opStack.isEmpty() and prec[opStack.peek()] >= prec[i]:
                answer += opStack.pop()
            # 이외의 연산들은 opStack에 추가
            opStack.push(i)
    # opStack에 연산이 남았으면
    while not opStack.isEmpty():
        answer += opStack.pop()
    return answer
S = "(A+B)*(C+D)"
print(solution(S))
'''
# 기본 라이브러리
'''
import sys
import os
sys.path.append('C:/Users/wnrrh/PycharmProjects/Algorithm_study/DevCourse')
import doublylinkedlist
from doublylinkedlist import Node
from doublylinkedlist import DoublyLinkedList

# 스택 구현
# 빈 리스트로 스택 생성
class ArrayStack:
    def __init__(self):
        self.data = []
	# 길이 출력하는 함수
    def size(self):
        return len(self.data)
	# 비어있나 여부 확인
    def isEmpty(self):
        return self.size() == 0
	# 스택에 item 추가
    def push(self, item):
        self.data.append(item)
	# 스택에서 젤 끝 원소 제거
    def pop(self):
        return self.data.pop()
	# 스택 젤 마지막 원소 찾기
    def peek(self):
        return self.data[-1]

# 주어진 문자열 exprStr split하여 끊어서 리스트로 표현
def splitTokens(exprStr):
    tokens = []
    val = 0
    valProcessing = False
    for c in exprStr:
        if c == ' ':
            continue
        if c in '0123456789':
            val = val * 10 + int(c)
            valProcessing = True
        else:
            if valProcessing:
                tokens.append(val)
                val = 0
            valProcessing = False
			# 괄호, 연산 추가
            tokens.append(c)
    # 마지막 숫자 추가
    if valProcessing:
        tokens.append(val)
    return tokens
# 후위표현식으로 연산자, 피연산자 순서대로 늘어놓게 만들기
def infixToPostfix(tokenList):
    prec = {
        '*': 3,
        '/': 3,
        '+': 2,
        '-': 2,
        '(': 1,
    }

    opStack = ArrayStack()
    postfixList = []
    for i in tokenList:
        if type(i) is int:
            postfixList.append(i)
        elif i == '(':
            opStack.push(i)
        elif i == ')':
            while opStack.peek() != '(':
                postfixList.append(opStack.pop())
            opStack.pop()
        else:
            while not opStack.isEmpty() and prec[opStack.peek()] >= prec[i]:
                postfixList.append(opStack.pop())
            opStack.push(i)
    while not opStack.isEmpty():
        postfixList.append(opStack.pop())
    return postfixList

# 후위표기로 변환된 리스트 연산하기
def postfixEval(tokenList):
	# 빈 스택 생성
    valStack = ArrayStack()
    for i in tokenList:
		# 숫자면 valStack에 push
        if type(i) is int:
            valStack.push(i)
		# 숫자가 아니고 연산인 경우
        else:
			# pop 2번 진행해 숫자 뽑고 각 연산 진행한 이후 연산한 값 다시 push
            b = valStack.pop()
            a = valStack.pop()
            if i == '*':
                valStack.push(a*b)
            elif i == '/':
                valStack.push(a/b)
            elif i == '+':
                valStack.push(a+b)
            elif i == '-':
                valStack.push(a-b)
	# size 계산해 1이상이면 pop 아니면 None 처리
    if valStack.size() >= 1:
        return valStack.pop()
    else:
        return None

def solution(expr):
    tokens = splitTokens(expr)
    postfix = infixToPostfix(tokens)
    print(postfix)
    val = postfixEval(postfix)
    return val
'''
'''
#4/11
### 1976 여행가자
import sys
input = sys.stdin.readline

n = int(input())
m = int(input())
graph = [list(map(int, input().split())) for _ in range(n)]
route = list(map(int, input().split()))
parent = [i for i in range(n+1)]

def find(x):
    if parent[x] != x:
        return find(parent[x])
    return x
def union(a, b):
    a = find(a)
    b = find(b)
    if a > b:
        parent[a] = b
    else:
        parent[b] = a

for i in range(n):
    for j in range(n):
        if graph[i][j] == 1:
            union(i+1, j+1)

start = find(route[0])
check = True
for i in route:
    if start != find(i):
        check = False
        break
if check:
    print('YES')
else:
    print('NO')
    
'''
'''
# 4/12
### 1707 이분 그래프
import sys
input = sys.stdin.readline
from collections import deque

def bfs(start):
    queue = deque([start])
    visited[start] = 1
    while queue:
        x = queue.popleft()
        for i in graph[x]:
            if not visited[i]:
                queue.append(i)
                visited[i] = -1 * visited[x]
            elif visited[i] == visited[x]:
                return False
    return True

k = int(input())
for _ in range(k):
    v, e = map(int, input().split())
    graph = [[] for _ in range(v+1)]
    visited = [0] * (v+1)
    for _ in range(e):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
    for i in range(1, v+1):
        if not visited[i]:
            answer = bfs(i)
            if not answer:
                break
    if answer:
        print('YES')
    else:
        print('NO')
'''

'''
# 4/14

### 23971 ZOAC 4

import sys
input = sys.stdin.readline
import math

h, w, n, m = map(int, input().split())
height = math.ceil(h/(n+1))
width = math.ceil(w/(m+1))
print(height*width)
'''

'''
### 5073 삼각형과 세 변
import sys
input = sys.stdin.readline

while True:
    a, b, c = map(int, input().split())
    if a == 0 and b == 0 and c == 0:
        break
    k = max(a, b, c)
    if a == b == c:
        print('Equilateral')
    elif k >= (a+b+c - k):
        print('Invalid')
    elif a != b and b != c and c!= a:
        print('Scalene')
    elif (a == b and a!= c) or (b == c and c!= a) or (c == a and a != b):
        print('Isosceles')
'''
'''
### 2292 벌집
import sys
input = sys.stdin.readline

n = int(input())
a = 1
d = 6
answer = 1
while n > a:
    answer += 1
    a += d
    d += 6
print(answer)
'''

'''
### 1157 단어 공부
import sys
input = sys.stdin.readline

alphabet = input().upper().rstrip()
word = list(set(alphabet))
li = []
for i in word:
    cnt = alphabet.count(i)
    li.append(cnt)
if li.count(max(li)) > 1:
    print('?')
else:
    maxidx = li.index(max(li))
    print(word[maxidx])
'''

'''
### 11723 집합
import sys
input = sys.stdin.readline
m = int(input())
s = set()
for i in range(m):
    command = input().rstrip().split()
    if len(command) == 1:
        if command[0] == 'all':
            s = set([i for i in range(1, 21)])
        else:
            s = set()
    else:
        comm, num = command[0], command[1]
        num = int(num)
        if comm == 'add':
            s.add(num)
        elif comm == 'remove':
            s.discard(num)
        elif comm == 'check':
            if num in s:
                print(1)
            else:
                print(0)
        elif comm == 'toggle':
            if num in s:
                s.discard(num)
            else:
                s.add(num)
'''

'''
### 13305 주유소
import sys
input = sys.stdin.readline

n = int(input())
roads = list(map(int, input().split()))
cost = list(map(int, input().split()))

price = cost[0]
answer = 0
for i in range(n-1):
    if price > cost[i]:
        price = cost[i]
    answer += price * roads[i]
    print(answer)
'''

'''
### 20920 영단어 암기는 괴로워
import sys
input = sys.stdin.readline
n, m = map(int, input().split())
dic = {}
for _ in range(n):
    word = input().rstrip()
    if len(word) < m:
        continue
    if word in dic:
        dic[word][0] += 1
    else:
        dic[word] = [1, len(word), word]
answer = sorted(dic.items(), key = lambda x: (-x[1][0], -x[1][1], x[1][2]))
for i in answer:
    print(i[0])
'''

'''
### 2075 N번째 큰 수
import sys
input = sys.stdin.readline
import heapq

n = int(input())
heap = []
for i in range(n):
    number = list(map(int, input().split()))
    if not heap:
        for num in number:
            heapq.heappush(heap, num)

    else:
        for num in number:
            if heap[0] < num:
                heapq.heappush(heap, num)
                heapq.heappop(heap)
print(heap[0])
'''
'''
### 21921 블로그
import sys
input = sys.stdin.readline

n, x = map(int, input().split())
day = list(map(int, input().split()))

if max(day) == 0:
    print('SAD')
else:
    visitor = sum(day[:x])
    max_visit = visitor
    cnt = 1
    for i in range(x, n):
        visitor += day[i]
        visitor -= day[i-x]
        if visitor > max_visit:
            max_visit = visitor
            cnt = 1
        elif visitor == max_visit:
            cnt += 1
    print(max_visit)
    print(cnt)
'''

'''
# 4/17
### 10431 줄세우기
import sys
input = sys.stdin.readline

t = int(input())
for _ in range(t):
    line = list(map(int, input().rstrip().split()))
    cnt = 0
    for i in range(1, len(line)-1):
        for j in range(i+1, len(line)):
            if line[i] > line[j]:
                line[i], line[j] = line[j], line[i]
                cnt += 1
    print(line[0], cnt)
'''


'''
### 13460 구슬 탈출 2
import sys
input = sys.stdin.readline
from collections import deque

n, m = map(int, input().split())
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
# 빨간구슬, 파란구슬 위치 총 4번
visited = [[[[0]*m for _ in range(n)] for _ in range(m)] for _ in range(n)]
# 그래프 입력넣고 빨간구슬, 파란구슬 각 시작점 정하기
graph = [list(input().rstrip()) for _ in range(n)]
for i in range(n):
    for j in range(m):
        if graph[i][j] == 'R':
            rx, ry = i, j
            graph[i][j] = '.'
        elif graph[i][j] == 'B':
            bx, by = i, j
            graph[i][j] = '.'
# 기울이는 순간 벽의 끝으로 이동하는 함수 생성
def move(x, y, dx, dy):
    move_cnt = 0
    while graph[x+dx][y+dy] != '#' and graph[x][y] != 'O':
        x += dx
        y += dy
        move_cnt += 1
    return x, y, move_cnt

def bfs():
    queue = deque()
    queue.append((rx, ry, bx, by, 1))
    visited[rx][ry][bx][by] = 1
    while queue:
        red_x, red_y, blue_x, blue_y, cnt = queue.popleft()
        # 구슬 이동이 10번 이상이면 중단
        if cnt > 10:
            break
        # 각 구슬 별 상하좌우 탐색 진행
        for i in range(4):
            nrx, nry, rmove = move(red_x, red_y, dx[i], dy[i])
            nbx, nby, bmove = move(blue_x, blue_y, dx[i], dy[i])
            # 파란구슬이 구멍에 들어가지 않았다는 조건 하에
            if graph[nbx][nby] != 'O':
                # 빨간구슬이 구멍에 들어가면 벽 기울인 횟수 cnt 리턴
                if graph[nrx][nry] == 'O':
                    return cnt
                # 빨간구슬, 파란구슬 같은 위치인 경우
                if nrx == nbx and nry == nby:
                    # 더 많이 기울인 구슬을 골라 해당 구슬의 위치를 이전으로 돌리기
                    if rmove > bmove:
                        nrx -= dx[i]
                        nry -= dy[i]
                    else:
                        nbx -= dx[i]
                        nby -= dy[i]
                # 방문하지 않았던 곳에 한해 queue에 삽입하고 방문처리
                if not visited[nrx][nry][nbx][nby]:
                    visited[nrx][nry][nbx][nby] = 1
                    queue.append((nrx, nry, nbx, nby, cnt + 1))
    return -1
print(bfs())
'''


# 4/18

