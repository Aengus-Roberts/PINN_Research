import itertools

permutations = list(itertools.permutations([1,2,3,4]))

print(permutations)

operations = ['+', '-', '*', '/']

rules = {}
for i in range(len(permutations)):
    rules[i] = {permutations[i][j]: operations[j] for j in range(len(operations))}

print(rules)

clues = [
    # Across
    (149445, 4),
    (4422, 2),
    (9654457, 6),
    (4502135, 3),
    (9223249, 4),
    (636724, 4),
    (1321, 2),
    (83812, 3),
    (1792156, 2),
    (164847, 5),
    (548, 2),
    (13139, 3),
    (831412, 4),
    (2272132, 2),
    (9436269, 4),
    (723345, 3),
    (23324126, 6),
    (2432, 2),
    (144444, 4),
    (214122, 4),

    # Down
    (11438281623, 4),
    (3325, 2),
    (23415, 3),
    (234411, 4),
    (15313, 2),
    (703296, 3),
    (1149812, 4),
    (13344447, 6),
    (1348499, 6),
    (164847, 5),
    (142423, 4),
    (434142, 4),
    (1437, 2),
    (44444, 4),
    (144443, 4),
    (33333, 3),
    (633388, 3),
    (9322, 2),
    (35214, 2),
    (6311, 2),
]

def get_div(rule):
    return [k for k, v in rules[rule].items() if v == '/'][0]

def get_sub(rule):
    return [k for k, v in rules[rule].items() if v == '-'][0]

def get_locations(string, char):
    return [i for i,string_char in enumerate(string) if string_char == char]

def check_rule_logic(rule):
    div_num = get_div(rule)
    sub_num = get_sub(rule)

    sound_logic = True
    correct_clue_pos = []
    for clue in clues:
        clue_str = str(clue[0])
        if str(div_num) in clue_str or str(sub_num) in clue_str:
            exists_true = False
            if str(div_num) in clue_str:
                to_check = get_locations(clue_str, div_num)
                for pos in to_check:
                    if pos == len(clue_str):
                        pass
                    else:
                        output = int(clue_str[:pos])/int(clue_str[pos+1:])
                        if len(str(output)) == clue[1]:
                            exists_true = True
                            correct_clue_pos.append(pos)
                            break
            if str(sub_num) in clue_str:
                to_check = get_locations(clue_str, sub_num)
                for pos in to_check:
                    if pos == len(clue_str):
                        pass
                    else:
                        output = int(clue_str[:pos])-int(clue_str[pos+1:])
                        if len(str(output)) == clue[1]:
                            exists_true = True
                            correct_clue_pos.append(pos)
                            break
            if exists_true == False:
                sound_logic = False
                break
    if sound_logic == True:
        return correct_clue_pos, rule
    else:
        return None

for rule in rules:
    output = check_rule_logic(rule)
    if output is not None:
        correct_positions, correct_rule = output

print(correct_rule)





