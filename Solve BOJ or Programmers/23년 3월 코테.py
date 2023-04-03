### 프로그래머스 스택/큐
'''
print(ord('A'))
print(ord('Z'))

print([2] + [4])
maps = ["X591X","X1X5X","X231X", "1XXX1"]
print(maps[0][1])
'''

### 백준 설탕배달 2839
'''
n = int(input())
answer = 0
while n >= 0:
    if n % 5 == 0:
        answer += (n//5)
        print(answer)
        break
    n -= 3
    answer += 1
else:
    print(-1)
'''

### 2xn 타일링 11726
'''
n = int(input())
dp = [0 for _ in range(n+1)]
if n <= 3:
    print(n)
else:
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n+1):
        dp[i] = dp[i-2] + dp[i-1]
    print(dp[n] % 10007)
'''

### RGB 거리
'''
n = int(input())
cost = 0
graph = []
for _ in range(n):
    graph.append(list(map(int, input().split())))
for i in range(1, n):
    graph[i][0] = min(graph[i-1][1], graph[i-1][2]) + graph[i][0]
    graph[i][1] = min(graph[i-1][0], graph[i-1][2]) + graph[i][1]
    graph[i][2] = min(graph[i-1][0], graph[i-1][0]) + graph[i][2]
print(min(graph[n-1]))

'''

'''
### 11727 타일링 2
import sys
input = sys.stdin.readline
n = int(input())
dp = [0, 1, 3]
for i in range(3, n+1):
    dp.append((dp[i-1]) + ((dp[i-2]) * 2))
print(dp[n]%10007)
'''
'''
### 12865 배낭 - 냅색 알고리즘
import sys
input = sys.stdin.readline
n, k = map(int, input().split())
bag = [[0]*(k+1) for _ in range(n+1)]
for i in range(1, n+1):
    w, v = map(int, input().split())
    for j in range(1, k+1):
        if j < w:
            bag[i][j] = bag[i-1][j]
        else:
            bag[i][j] = max(bag[i-1][j], bag[i-1][j-w] + v)
print(bag[n][k])'''

'''
### 1010 다리놓기
import sys
input = sys.stdin.readline

t = int(input())
for _ in range(t):
    n, m = map(int, input().split())
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            if i == 1:
                dp[i][j] = j
                continue
            if i == j:
                dp[i][j] = 1
            else:
                if j > i:
                    dp[i][j] = dp[i][j-1] + dp[i-1][j-1]
    print(dp[n][m])

'''

'''
### 2193 이친수
n = int(input())
dp = [0, 1, 1]
for i in range(3, n+1):
    dp.append(dp[i-2] + dp[i-1])
print(dp[n])
'''

'''
### 14501 퇴사
n = int(input())
income = [0]*(n+1)
plan = []
for _ in range(n):
    plan.append(list(map(int, input().split())))
for i in range(n):
    for j in range(i+plan[i][0], n+1):
        if income[j] < income[i] + plan[i][1]:
            income[j] = income[i] + plan[i][1]
print(income[-1])
'''

'''
### 9251 LCS
import sys
input = sys.stdin.readline

seq1 = input().strip().upper()
seq2 = input().strip().upper()
l1, l2 = len(seq1), len(seq2)
dp = [[0] * (l2 + 1) for _ in range(l1 + 1)]
for i in range(1, l1+1):
    for j in range(1, l2+1):
        if seq1[i-1] == seq2[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
print(dp[-1][-1])
'''

'''
### 11052 카드 구매하기
import sys
input = sys.stdin.readline

n = int(input())
pay = [0] + list(map(int, input().split()))
dp = [0]*(n+1)
for i in range(1, n+1):
    for j in range(1, i+1):
        dp[i] = max(dp[i], dp[i-j] + pay[j])
print(dp[n])
'''

'''
### 2293 동전 1
import sys
input = sys.stdin.readline

n, k = map(int, input().split())
coin = [int(input()) for _ in range(n)]
dp = [0]*(k+1)
dp[0] = 1

for i in coin:
    for j in range(i, k+1):
        dp[j] += dp[j-i]
print(dp[k])
'''

'''
### 9663 돌 게임 2
import sys
input = sys.stdin.readline

n = int(input())
if n % 2 == 1:
    print('CY')
else:
    print('SK')
'''

'''
### 11057 오르막 수
import sys
input = sys.stdin.readline
n = int(input())
dp = [1] * 10
for i in range(n-1):
    for j in range(1, 10):
        dp[j] += dp[j-1]
print(sum(dp) % 10007)
'''

'''
### 11051 이항계수 2
import sys
input = sys.stdin.readline

n, k = map(int, input().split())
dp = [[1]*(k+1) for _ in range(n+1)]
for i in range(1, k+1):
    for j in range(i+1, n+1):
        dp[j][i] = dp[j-1][i] + dp[j-1][i-1]
print(dp[n][k] % 10007)
'''

'''
### 11055 가장 큰 증가하는 부분 수열
n = int(input())
seq = list(map(int, input().split()))
dp = [0]*n
dp[0] = seq[0]
for i in range(1, n):
    for j in range(i):
        if seq[j] < seq[i]:
            dp[i] = max(dp[i], dp[j] + seq[i])
        else:
            dp[i] = max(dp[i], seq[i])
print(max(dp))
'''

'''
### 1699 제곱수의 합
import sys
input = sys.stdin.readline

n = int(input())
dp = [0] * (n+1)
for i in range(1, n+1):
    dp[i] = i
    for j in range(1, i):
        if j**2 > i:
            break
        if dp[i] > dp[i-j**2] + 1:
            dp[i] = dp[i-j**2] + 1
print(dp[n])
print(dp)
'''
'''
### 11660 구간 합 구하기 5
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
graph = [list(map(int, input().split())) for _ in range(n)]
dp = [[0]*(n+1) for _ in range(n+1)]
for i in range(1, n+1):
    for j in range(1, n+1):
        dp[i][j] = dp[i-1][j] + dp[i][j-1] + graph[i-1][j-1] - dp[i-1][j-1]
for _ in range(m):
    x1, y1, x2, y2 = map(int, input().split())
    print(dp[x2][y2] - dp[x1-1][y2] - dp[x2][y1-1] + dp[x1-1][y1-1])
'''

'''
### 11722 가장 긴 감소하는 부분 수열
import sys
input = sys.stdin.readline

n = int(input())
seq = list(map(int, input().split()))
dp = [1]*n
for i in range(1, n):
    for j in range(i):
        if seq[i] < seq[j]:
            dp[i] = max(dp[i], dp[j] + 1)
print(max(dp))
'''
'''
### 2294 동전 2
import sys
input = sys.stdin.readline

n, k = map(int, input().split())
coin = [int(input()) for _ in range(n)]
dp = [10001]*(k+1)
dp[0] = 0

for i in coin:
    for j in range(i, k+1):
        if dp[j] > 0:
            dp[j] = min(dp[j], dp[j-i] + 1)
if dp[k] == 10001:
    print(-1)
else:
    print(dp[k])
'''


'''
## 프로그래머스 데브코스 #1 계좌
accounts = [
    ['1', '123-4567-89'],
    ['1', '00458230495'],
    ['2', '779-12345-523'],
    ['3', '987-6543-21'],
    ['3', '0593-52-60293']
]
k = '11111-222222'
a = '123-4567-89'
b = '00458-230495'
li = '123-4567-8900458-23049511111-222222'



dic = {1: '123-4567-8900458-230495', 2: '779-12345-523', 3: '987-6543-210593-52-60293'}
transfers = [
    ['123-4567-89', '0593-52-69283', '100000'],
    ['00458-230495', '123-4567-89', '1000000'],
    ['0593-52-60293', '987-6543-21', '50000'],
    ['0593-52-60293', '779-12345-523', '30000'],
    ['779-12345-523', '00458-230495', '500000'],
    ['779-12345-523', '123-4567-89', '90000']
    ]

def solutions(n, accounts, transfers):
    answer = {}
    dic = {}
    money = {}
    for i in range(len(accounts)):
        dic[accounts[i][1]] = int(accounts[i][0])
    for transfer in transfers:
        pay = 0
        if dic[transfer[0]] == dic[transfer[1]]:
            continue
        else:
            pay += int(transfer[2])
            if money.get(dic[transfer[0]]) == None:
                money[dic[transfer[0]]] = pay
            else:
                money[dic[transfer[0]]] += pay
    money.sorted(money.items())
    for i in money:
        answer.append(i[1])
    if len(answer) == n:
        return answer
    else:
        return answer + [0]
'''

'''
### 프로그래머스 데브코스 2번 티셔츠
def solutions(size_info, shirts_info):
    shirts_info.sort(key = lambda x: (x[0], x[1]))
    ordered = []
    for shirts in shirts_info:
        for i in range(len(size_info)):
            if size_info[i]-5 <= shirts[0] <= size_info[i]+5:
                ordered.append((size_info[i], shirts[1]))
    ordered.sort(key = lambda x: x[1])
    dic = {}
    for i in range(len(ordered)):
        if dic.get(ordered[i][0]) == None:
            dic[ordered[i][0]] = ordered[i][1]
        else:
            continue
    mini = 1e9
    for i in range(len(shirts_info)):
        if shirts_info[i][1] < mini:
            mini = shirts_info[i][1]
    answer = 0
    for i in dic.keys():
        if i in size_info:
            answer += dic[i]
    if len(dic.keys()) != len(answer):
        answer += mini * (len(size_info)-len(dic.keys()))
    return answer
'''

'''
### 9012 괄호
import sys
input = sys.stdin.readline

t = int(input())
for _ in range(t):
    vps = input()
    stack = []
    for i in vps:
        if i == '(':
            stack.append(i)
        elif i == ')':
            if stack:
                stack.pop()
            else:
                print('NO')
                break
    else:
        if stack:   
            print('NO')
        else:
            print('YES')
'''

'''
### 10828 스택
import sys
input = sys.stdin.readline
n = int(input())
stack = []

for i in range(n):
    command = input().split()
    if command[0] == 'push':
        stack.append(command[1])
    elif command[0] == 'pop':
        if not stack:
            print(-1)
        else:
            print(stack.pop())
    elif command[0] == 'size':
        print(len(stack))
    elif command[0] == 'empty':
        if not stack:
            print(1)
        else:
            print(0)
    elif command[0] == 'top':
        if not stack:
            print(-1)
        else:
            print(stack[-1])
'''


'''
### 10845 큐
import sys

n = int(sys.stdin.readline())
queue = []
for i in range(n):
    command = sys.stdin.readline().split()
    if command[0] == 'push':
        queue.append(command[1])
    elif command[0] == 'pop':
        if queue:
            print(queue.pop(0))
        else:
            print(-1)
    elif command[0] == 'size':
        print(len(queue))
    elif command[0] == 'empty':
        if queue:
            print(0)
        else:
            print(1)
    elif command[0] == 'front':
        if queue:
            print(queue[0])
        else:
            print(-1)
    elif command[0] == 'back':
        if queue:
            print(queue[-1])
        else:
            print(-1)
'''

'''
### 2164 카드 2
from collections import deque
import sys
input = sys.stdin.readline

n = int(input())
queue = deque()
for i in range(1, n+1):
    queue.append(i)
while len(queue) > 1:
    queue.popleft()
    queue.append(queue.popleft())
print(queue.pop())
'''

'''

### 1874 스택 수열
import sys
input = sys.stdin.readline

n = int(input())
cnt = 1
flag = True
stack = []
operation = []

for _ in range(n):
    num = int(input())
    while cnt <= num:
        stack.append(cnt)
        cnt += 1
        operation.append('+')
    if stack[-1] == num:
        stack.pop()
        operation.append('-')
    else:
        flag = False
if not flag:
    print('NO')
else:
    for i in operation:
        print(i)
'''


'''
### 1966 프린터 큐
import sys
from collections import deque
input = sys.stdin.readline

t = int(input())
for _ in range(t):
    n, m = map(int, input().split())
    queue = deque(list(map(int, input().split())))
    importance = 0
    while queue:
        maxi = max(queue)
        front = queue.popleft()
        m -= 1
        if maxi == front:
            importance += 1
            if m < 0:
                print(importance)
                break
        else:
            queue.append(front)
            if m < 0:
                m = len(queue) - 1
'''

'''
### 1157 요세푸스 순열
import sys
input = sys.stdin.readline

n, k = map(int, input().split())
array = [i for i in range(1, n+1)]
answer = []
idx = 0
for i in range(n):
    idx += k-1
    if idx >= len(array):
        idx %= len(array)
    answer.append(str(array.pop(idx)))
print('<', ', '.join(answer), '>', sep='')
'''

'''
### 10866 덱
from collections import deque
import sys
input = sys.stdin.readline

n = int(input())
queue = deque()

for i in range(n):
    word = input()
    command = word.split()
    if command[0] == 'push_front':
        queue.appendleft(command[1])
    elif command[0] == 'push_back':
        queue.append(command[1])
    elif command[0] == 'pop_front':
        if not queue:
            print('-1')
        else:
            print(queue[0])
            queue.popleft()
    elif command[0] == 'pop_back':
        if not queue:
            print('-1')
        else:
            print(queue[len(queue)-1])
            queue.pop()
    elif command[0] == 'size':
        print(len(queue))
    elif command[0] == 'empty':
        if not queue:
            print('1')
        else:
            print('0')
    elif command[0] == 'front':
        if not queue:
            print('-1')
        else:
            print(queue[0])
    elif command[0] == 'back':
        if not queue:
            print('-1')
        else:
            print(queue[len(queue)-1])
'''

'''
### 10799 쇠막대기
import sys
input = sys.stdin.readline

razor = list(input().strip())
answer = 0
stack = []
for i in range(len(razor)):
    if razor[i] == '(':
        stack.append(razor[i])
    else:
        if razor[i-1] == '(':
            stack.pop()
            answer += len(stack)
        else:
            stack.pop()
            answer += 1
print(answer)
'''

'''
### 11286 절대값 힙
import heapq
import sys
input = sys.stdin.readline

n = int(input())
heap = []
for _ in range(n):
    num = int(input())
    if num != 0:
        heapq.heappush(heap, (abs(num), num))
    else:
        if not heap:
            print(0)
        else:
            print(heapq.heappop(heap)[1])
'''

'''
### 1021 회전하는 큐
from collections import deque
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
idx = list(map(int, input().split()))
queue = deque([i for i in range(1, n+1)])
answer = 0
for i in idx:
    while True:
        if queue[0] == i:
            queue.popleft()
            break
        else:
            if queue.index(i) <= len(queue)//2:
                queue.rotate(-1)
                answer += 1
            else:
                queue.rotate(1)
                answer += 1
print(answer)
'''

'''
### 9095 1,2,3 더하기
import sys
input = sys.stdin.readline

def plus(n):
    if dp[n] > 0:
        return dp[n]
    if n == 1:
        return 1
    elif n == 2:
        return 2
    elif n == 3:
        return 4
    else:
        dp[n] = plus(n-3) + plus(n-2) + plus(n-1)
        return dp[n]
t = int(input())
for _ in range(t):
    n = int(input())
    dp = [0]*(n+1)
    print(plus(n))
'''

'''
### 9465 스티커
import sys
input = sys.stdin.readline

t = int(input())
for _ in range(t):
    n = int(input())
    sticker = []
    for i in range(2):
        sticker.append(list(map(int, input().split())))
    for j in range(1, n):
        if j == 1:
            sticker[0][j] += sticker[1][j-1]
            sticker[1][j] += sticker[0][j-1]
        else:
            sticker[0][j] += max(sticker[1][j-1], sticker[1][j-2])
            sticker[1][j] += max(sticker[0][j-1], sticker[0][j-2])
    print(max(sticker[0][n-1], sticker[1][n-1]))

'''


'''
### 1520 내리막길
import sys
input = sys.stdin.readline

m, n = map(int, input().split())
visited = [[-1]*n for _ in range(m)]
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
graph = []
for _ in range(m):
    graph.append(list(map(int, input().split())))

def dfs(x, y):
    if x == m-1 and y == n-1:
        return 1
    if visited[x][y] != -1:
        return visited[x][y]
    route = 0
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if 0<=nx<m and 0<=ny<n and graph[x][y] > graph[nx][ny]:
            route += dfs(nx, ny)
    visited[x][y] = route
    return visited[x][y]
print(dfs(0, 0))
'''

'''
### 4949 균형잡힌 세상
import sys
input = sys.stdin.readline
while True:
    sentence = input().rstrip()
    stack = []
    flag = True
    if sentence == '.':
        break
    for i in sentence:
        if i =='(' or i == '[':
            stack.append(i)
        elif i == ')':
            if not stack or stack[-1] == '[':
                flag = False
                break
            elif stack[-1] == '(':
                stack.pop()
        elif i == ']':
            if not stack or stack[-1] == '(':
                flag = False
                break
            elif stack[-1] == '[':
                stack.pop()
    if flag == True and not stack:
        print('yes')
    else:
        print('no')
'''

'''
### 1406 에디터
import sys
input = sys.stdin.readline

left = list(input().rstrip())
right = []
m = int(input())
for i in range(m):
    command = input().split()
    if command[0] == 'P':
        left.append(command[1])
    elif command[0] == 'L' and left:
        right.append(left.pop())
    elif command[0] == 'D' and right:
        left.append(right.pop())
    elif command[0] == 'B' and left:
        left.pop()
print(''.join(left + list(reversed(right))))
'''

'''
### 11048 이동하기
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
graph = [list(map(int, input().split())) for _ in range(n)]
dp = [[0]*(m+1) for _ in range(n+1)]
for i in range(1, n+1):
    for j in range(1, m+1):
        dp[i][j] = graph[i-1][j-1] + max(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
print(dp[n][m])
'''

'''
### 2748 피보나치 수 2
n = int(input())
fibo = []
answer = 0
for i in range(n+1):
    if i == 0:
        answer = 0
    elif i <= 2:
        answer = 1
    else:
        answer = fibo[-1] + fibo[-2]
    fibo.append(answer)
print(fibo[-1])
'''

'''
### 2133 타일 채우기
import sys
input = sys.stdin.readline
n = int(input())
dp = [0] * 31
if n % 2 == 1:
    print(0)
else:
    dp[2] = 3
    for i in range(4, 31, 2):
        dp[i] = dp[i-2]*3 + 2
        for j in range(2, i-2, 2):
            dp[i] += dp[j]*2
    print(dp[n])
'''


'''
### 2225 합분해
import sys
input = sys.stdin.readline

n, k = map(int, input().split())
dp = [[0]*201 for _ in range(201)]

for i in range(201):
    dp[1][i] = 1
    dp[2][i] = i+1
for i in range(2, 201):
    dp[i][1] = i
    for j in range(2, 201):
        dp[i][j] = (dp[i-1][j] + dp[i][j-1]) % 1e9
print(int(dp[k][n]))
'''

'''
### 2565 전깃줄
import sys
input = sys.stdin.readline

n = int(input())
wire = sorted(list(map(int, input().split())) for _ in range(n))
dp = [1] * n
for i in range(1, n):
    for j in range(i):
        if wire[j][1] < wire[i][1]:
            dp[i] = max(dp[i], dp[j] + 1)
print(n - max(dp))
'''

'''
### 18258 큐 2
from collections import deque
import sys
input = sys.stdin.readline

n = int(input())
queue = deque()
for _ in range(n):
    command = input().split()
    if command[0] == 'push':
        queue.append(command[1])
    elif command[0] == 'pop':
        if queue:
            print(queue.popleft())
        else:
            print(-1)
    elif command[0] == 'size':
        print(len(queue))
    elif command[0] == 'empty':
        if queue:
            print(0)
        else:
            print(1)
    elif command[0] == 'front':
        if queue:
            print(queue[0])
        else:
            print(-1)
    elif command[0] == 'back':
        if queue:
            print(queue[-1])
        else:
            print(-1)
'''

'''
### 17298 오큰수
import sys
input = sys.stdin.readline

n = int(input())
seq = list(map(int, input().split()))
stack = []
answer = [-1] * n

for i in range(n):
    while stack and seq[stack[-1]] < seq[i]:
        answer[stack.pop()] = seq[i]
    stack.append(i)
print(*answer)
'''

'''
### 9655 돌 게임
import sys
input = sys.stdin.readline

n = int(input())
if n % 2 == 0:
    print('CY')
else:
    print('SK')
'''

'''
### 1309 동물원
import sys
input = sys.stdin.readline

n = int(input())
dp = [0] * (n+1)
dp[0] = 1
dp[1] = 3
for i in range(2, n+1):
    dp[i] = (dp[i-2] + dp[i-1]*2) % 9901
print(dp[n])

'''

'''
### 1005 ACM Craft
import sys
input = sys.stdin.readline
from collections import deque

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    time = [0] + list(map(int, input().split()))
    graph = [[] for _ in range(n+1)]
    visited = [0 for _ in range(n+1)]
    degree = [0 for _ in range(n+1)]
    dp = [0 for _ in range(n+1)]
    for _ in range(k):
        a, b = map(int, input().split())
        graph[a].append(b)
        degree[b] += 1
    queue = deque()
    for i in range(1, n+1):
        if not degree[i]:
            queue.append(i)
            dp[i] = time[i]
    while queue:
        x = queue.popleft()
        for i in graph[x]:
            degree[i] -= 1
            dp[i] = max(dp[x] + time[i], dp[i])
            if not degree[i]:
                queue.append(i)
    num = int(input())
    print(dp[num])
'''

'''
### 1890 점프
import sys
input = sys.stdin.readline

n = int(input())
graph = [list(map(int, input().split())) for _ in range(n)]
dp = [[0]*n for _ in range(n)]
dp[0][0] = 1

for i in range(n):
    for j in range(n):
        if graph[i][j] != 0 and dp[i][j] != 0:
            if i + graph[i][j] < n:
                dp[i+graph[i][j]][j] += dp[i][j]
            if j + graph[i][j] < n:
                dp[i][j+graph[i][j]] += dp[i][j]
print(dp[-1][-1])
'''

'''
### 1753 최단 경로
import sys
import heapq
INF = sys.maxsize
input = sys.stdin.readline

V, E = map(int, input().split())
K = int(input())
graph = [[] for _ in range(V+1)]
for _ in range(E):
    start, end, cost = map(int, input().split())
    graph[start].append([end, cost])

def dijstra(start):
    distance = [INF] * (V+1)
    heap = []
    heapq.heappush(heap, [start, 0])
    distance[start] = 0
    while heap:
        now_node, now_cost = heapq.heappop(heap)
        for next_node, next_cost in graph[now_node]:
            next_cost = next_cost + now_cost
            if next_cost < distance[next_node]:
                distance[next_node] = next_cost
                heapq.heappush(heap, [next_node, next_cost])
    return distance
answer = dijstra(K)

for i in range(1, len(answer)):
    if i == K:
        print(0)
    elif answer[i] == INF:
        print("INF")
    else:
        print(answer[i])
'''

'''
### 11725 트리의 부모 찾기
import sys
input = sys.stdin.readline
from collections import deque

n = int(input())
graph = [[] for _ in range(n+1)]
for _ in range(n-1):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

answer = [0] * (n+1)
def bfs():
    queue = deque()
    queue.append(1)
    while queue:
        x = queue.popleft()
        for i in graph[x]:
            if answer[i] == 0:
                answer[i] = x
                queue.append(i)
bfs()
for i in answer[2:]:
    print(i)
'''

'''
### 1717 집합의 표현
import sys
input = sys.stdin.readline
sys.setrecursionlimit(10**6)

n, m = map(int, input().split())
parent = [i for i in range(n+1)]

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    a = find(a)
    b = find(b)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b
for i in range(m):
    value, a, b = map(int, input().split())
    if value == 0:
        union(a, b)
    else:
        if find(a) == find(b):
            print('yes')
        else:
            print('no')
'''

'''
### 2493 탑
import sys
input = sys.stdin.readline

n = int(input())
tower = list(map(int, input().split()))
stack = []
answer = []
for i in range(n):
    while stack:
        if stack[-1][1] > tower[i]:
            answer.append(stack[-1][0] + 1)
            break
        else:
            stack.pop()
    if not stack:
        answer.append(0)
    stack.append((i, tower[i]))
print(*answer)
'''

'''
### 2096 내려가기
import sys
input = sys.stdin.readline

n = int(input())
graph = list(map(int, input().split()))
max_dp = graph
min_dp = graph

for _ in range(n-1):
    a, b, c = map(int, input().split())
    max_dp = [a + max(max_dp[0], max_dp[1]), b + max(max_dp), c + max(max_dp[1], max_dp[2])]
    min_dp = [a + min(min_dp[0], min_dp[1]), b + min(min_dp), c + min(min_dp[1], min_dp[2])]
print(max(max_dp), min(min_dp))
'''

'''
### 17070 파이프 옮기기 1
import sys
input = sys.stdin.readline

n = int(input())
graph = [list(map(int, input().split())) for _ in range(n)]
answer = 0
def solution(i, j, direction):
    global answer
    if i == n-1 and j == n-1:
        answer += 1
        return answer
    if direction == 0 or direction == 2:
        if j+1 < n and graph[i][j+1] == 0:
            solution(i, j+1, 0)
    if direction == 1 or direction == 2:
        if i+1 < n and graph[i+1][j] == 0:
            solution(i+1, j, 1)
    if i+1 < n and j+1 < n:
        if graph[i+1][j] == 0 and graph[i][j+1] == 0 and graph[i+1][j+1] == 0:
            solution(i+1, j+1, 2)
if graph[n-1][n-1] == 1:
    print(0)
else:
    solution(0, 1, 0)
    print(answer)
'''

'''
### 1916 최소비용 구하기
import sys
input = sys.stdin.readline
import heapq

n = int(input())
m = int(input())
graph = [[] for _ in range(n+1)]
distance = [1e9] * (n+1)
for _ in range(m):
    a, b, c = map(int, input().split())
    graph[a].append((b, c))
def dijkstra(x):
    heap = []
    distance[x] = 0
    heapq.heappush(heap, (x, 0))
    while heap:
        node, cost = heapq.heappop(heap)
        if distance[node] < cost:
            continue
        for next_node, next_cost in graph[node]:
            total = cost + next_cost
            if distance[next_node] > total:
                distance[next_node] = total
                heapq.heappush(heap, (next_node, total))
start, end = map(int, input().split())
dijkstra(start)
print(distance[end])
'''

'''
### 11501 주식
import sys
input = sys.stdin.readline

t = int(input())
for _ in range(t):
    n = int(input())
    price = list(map(int, input().split()))
    profit = 0
    val = 0
    for i in range(len(price)-1, -1, -1):
        if price[i] > val:
            val = price[i]
        else:
            profit += val - price[i]
    print(profit)
'''

'''
### 16435 스네이크버드
import sys
input = sys.stdin.readline

n, l = map(int, input().split())
height = list(map(int, input().split()))
height.sort()
for h in height:
    if h > l:
        break
    elif h <= l:
        l += 1
print(l)
'''

'''
### 1417 국회의원 선거
import sys
input = sys.stdin.readline

n = int(input())
num_one = int(input())
vote = []
for _ in range(n-1):
    vote.append(int(input()))
vote.sort(reverse = True)
answer = 0
if n == 1:
    print(0)
else:
    while vote[0] >= num_one:
        num_one += 1
        vote[0] -= 1
        answer += 1
        vote.sort(reverse = True)
    print(answer)
'''


'''
### 1461 도서관
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
location = list(map(int, input().split()))
positive = []
negative = []
answer = []
for loc in location:
    if loc > 0:
        positive.append(loc)
    else:
        negative.append(abs(loc))
positive.sort(reverse = True)
negative.sort(reverse = True)

for i in range(len(positive)):
    if i % m == 0:
        answer.append(positive[i])
for i in range(len(negative)):
    if i % m == 0:
        answer.append(negative[i])
answer.sort()
dist = 0
for i in range(len(answer)):
    dist += 2 * answer[i]
dist -= answer[-1]
print(dist)
'''

'''
### 15988 1,2,3 더하기 3
import sys
input = sys.stdin.readline

dp = [1,2,4,7]
t = int(input())
for i in range(t):
    n = int(input())
    for j in range(len(dp), n):
        dp.append((dp[-3] + dp[-2] + dp[-1]) % 1000000009)
    print(dp[n-1])
'''

'''
### 1965 상자넣기
import sys
input = sys.stdin.readline

n = int(input())
box = list(map(int, input().split()))
dp = [1] * len(box)
for i in range(len(box)):
    for j in range(i):
        if box[i] > box[j]:
            dp[i] = max(dp[i], dp[j]+1)
print(max(dp))
'''

'''
### 14002 가장 긴 증가하는 부분 수열 4
import sys
input = sys.stdin.readline

n = int(input())
seq = list(map(int, input().split()))
length = [1] * len(seq)
answer = []
for i in range(n):
    for j in range(i):
        if seq[i] > seq[j]:
            length[i] = max(length[i], length[j] + 1)
print(max(length))

sequence = []
maxi = max(length)
for i in range(n-1, -1, -1):
    if length[i] == maxi:
        sequence.append(seq[i])
        maxi -= 1
sequence.sort()
print(*sequence)
'''
'''
### 2828 사과 담기 게임
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
j = int(input())
left = 1
right = m
answer = 0

for _ in range(j):
    apple = int(input())
    if left <= apple and right >= apple:
        continue
    elif apple > right:
        answer += (apple-right)
        left += (apple-right)
        right = apple
    else:
        answer += (left - apple)
        right -= (left - apple)
        left = apple

print(answer)
'''


'''
### 1052 물병
import sys
input = sys.stdin.readline

n, k = map(int, input().split())
answer = 0
while bin(n).count('1') > k:
    n = n+1
    answer += 1
print(answer)
'''

'''
### 2138 전구와 스위치
import sys
input = sys.stdin.readline
import copy

n = int(input())
state = list(map(int, input().rstrip()))
want = list(map(int, input().rstrip()))

def change_state(number):
    if number == 0:
        number = 1
    else:
        number = 0
    return number

def switch(state, idx):
    if idx == 1:
        state[0] = change_state(state[0])
        state[1] = change_state(state[1])
    for i in range(1, n):
        if state[i-1] != want[i-1]:
            idx += 1
            state[i-1] = change_state(state[i-1])
            state[i] = change_state(state[i])
            if i != n-1:
                state[i+1] = change_state(state[i+1])
    if state == want:
        return idx
    else:
        return -1

push_first = copy.deepcopy(state)
push_first = switch(push_first, 0)
not_push_first = copy.deepcopy(state)
not_push_first = switch(not_push_first, 1)

if not_push_first >= 0 and push_first >= 0:
    print(min(not_push_first, push_first))
elif not_push_first >= 0 and push_first < 0:
    print(not_push_first)
elif not_push_first < 0 and push_first >= 0:
    print(push_first)
else:
    print(-1)
'''

'''
### 1182 부분 수열의 합
import sys
input = sys.stdin.readline

n, s = map(int, input().split())
seq = list(map(int, input().split()))
answer = 0

def dfs(idx, total):
    global answer
    if idx >= n:
        return answer
    total += seq[idx]
    if total == s:
        answer += 1
    dfs(idx+1, total)
    dfs(idx+1, total - seq[idx])
dfs(0, 0)
print(answer)
'''

'''
### 1759 암호 만들기
import sys
input = sys.stdin.readline

l, c = map(int, input().split())
alpha = sorted(list(input().split()))
answer = []
vowel = ['a', 'e', 'i', 'o', 'u']

def dfs(cnt, idx):
    if cnt == l:
        vo, co = 0, 0
        for i in range(l):
            if answer[i] in vowel:
                vo += 1
            else:
                co += 1
        if vo >= 1 and co >= 2:
            return print(''.join(answer))

    for i in range(idx, c):
        answer.append(alpha[i])
        dfs(cnt+1, i+1)
        answer.pop()
dfs(0, 0)
'''

'''
### 1476 날짜 계산
import sys
input = sys.stdin.readline

e, s, m = list(map(int, input().split()))
year = 1

while True:
    if (year-e) % 15 == 0 and (year-s) % 28 == 0 and (year-m) % 19 == 0:
        break
    year += 1
print(year)
'''


### 2003 수들의 합 2
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
seq = list(map(int, input().split()))
left, right = 0, 1
answer = 0

while left <= right <= n:
    total = sum(seq[left:right])
    if total < m:
        right += 1
    elif total > m:
        left += 1
    else:
        answer += 1
        right += 1
print(answer)