import numpy as np

to_label = {'C12': 0, 'C13': 1, 'C14': 2, 'C21': 3, 'C23': 4, 'C24': 5, 'C31': 6, 'C32': 7, 'C34': 8, 'C41': 9, 'C42': 10, 'C43': 11,
					'B12': 12, 'B13': 13, 'B14': 14, 'B21': 15, 'B23': 16, 'B24': 17, 'B31': 18, 'B32': 19, 'B34': 20, 'B41': 21, 'B42': 22, 'B43': 23,
					'E12': 24, 'E13': 25, 'E14': 26,
					'P56': 27, 'P65': 28, 'P67': 29, 'P76': 30, 'P78': 31, 'P87': 32, 'P85': 33, 'P58': 34,
					'sos': 35, 'eos': 36, 'pad': 37}
label_num = [0]*35
def rm_char(string):
	while('\n' in string or '\r' in string):
		# print(string)
		string = string.rstrip('\n')
		string = string.rstrip('\r')
	return string

f = open('gt.txt', 'r')
lines = f.readlines()
num_lines = len(lines)
num_scenario = 0
scenarios = []
sce_id_list = []
sce_id = 1

for i, line in enumerate(lines):
	line = rm_char(line)
	if line.isdigit():
		sce_id = int(line)
		num_action = 0
		# actions = [35]*8
		actions = []
		actions.append(to_label['sos'])
		while(1):
			if i + num_action+1 < num_lines:
				# if not lines[i + num_action + 1].isdigit():
				new_action = lines[i+num_action+1]
				new_action = rm_char(new_action.upper())
				if new_action != '' and new_action != '\n' and new_action != '\r':
					try:
						new_action = to_label[new_action]
						label_num[new_action] += 1
					except Exception as e:
						print(line)
					# new_action = to_label[new_action]
					try:
						# actions[num_action] = new_action
						actions.append(new_action)
					except Exception as e:
						print('out of index: %s' % (line))
					num_action += 1
				else:
					break
			else:
				break
		actions.append(to_label['eos'])
		scenarios.append(actions)
		sce_id_list.append(sce_id)
	num_scenario += 1
scenarios = np.asarray(scenarios)
sce_id_list = np.asarray(sce_id_list)
np.save('gt', scenarios)
np.save('scenario_id', sce_id_list)

f.close
print(label_num)

