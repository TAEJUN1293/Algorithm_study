a, b = map(int, input().split())
data = ['It','is','rather','for','us','to','be','here','dedicated','to','the','great','task','remaining','before','us','that','from','these','honored','dead','we','take','increased',
        'devotion','to','that','cause','for','which','they','gave','the','last','full','measure','of',
        'devotion','that','we','here','highly','resolve','that','these','dead','shall','not','have',
        'died','in','vain','that','this','nation','under','God','shall','have','a','new','birth',
        'of','freedom','and','that','government','of','the','people','by','the','people',
        'for','the','people','shall','not','perish','from','the','earth']
answer = []
for i in range(len(data)):
    if a <= len(data[i]) <= b:
        answer.append(data[i])
if answer:
    print(' '.join(answer))
    print(len(answer))
else:
    print(0)
