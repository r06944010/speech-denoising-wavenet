import ujson as json
import sys

log_dir = sys.argv[1]

if !log_dir:
	print('Please input log file')
	exit()

with open('spk_info.json') as f:
	spk = json.load(f)

with open(log_dir) as f:
	score = f.readlines()

g_dict = {'FF':[0, 0], 'FM':[0,0], 'MM':[0,0]} 

for i in range(3000):
	s1, s2 = score[2*i].split('_')[0:3:2]
	s1 = spk[s1[:3]]
	s2 = spk[s2[:3]]
	sdr = sum(map(float,score[2*i+1][1:-2].split()))/2

	if s1 == 'F' and s2 == 'F':
		g_dict['FF'][0] += sdr
		g_dict['FF'][1] += 1
	elif s1 == 'M' and s2 == 'M':
		g_dict['MM'][0] += sdr
		g_dict['MM'][1] += 1
	else:
		g_dict['FM'][0] += sdr
		g_dict['FM'][1] += 1

for s in ['FF', 'FM', 'MM']:
	g_dict[s][0] /= g_dict[s][1]

print(g_dict)


