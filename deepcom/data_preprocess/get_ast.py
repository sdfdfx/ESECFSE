import javalang
import json
from tqdm import tqdm
import collections
import sys
import os
import pickle
import numpy as np

def read_json(rootpath):
    splitdata = next(os.walk(rootpath))[2]
    print(splitdata)
    for item in splitdata:
        if item == 'test.json':
            continue
        splitpath = item.replace('.json', '')
        if not os.path.exists(splitpath):
            os.mkdir(splitpath)

        tmp = []
        for line in open(rootpath + item, 'r'):
            tmp.append(json.loads(line))

        out_src_file = splitpath + '/src_' + item.replace('.json', '') + '.txt'
        out_tgt_file = splitpath + '/tgt_' + item.replace('.json', '') + '.txt'
        with open(out_src_file, 'w') as src, open(out_tgt_file, 'w') as tgt:
            for data in tqdm(tmp):
                datacode = data['code'].replace('\n', '')
                src.write(datacode + '\n')
                tgt.write(data['nl'] + '\n')


def get_name(obj):
    if (type(obj).__name__ in ['list', 'tuple']):
        a = []
        for i in obj:
            a.append(get_name(i))
        return a
    elif (type(obj).__name__ in ['dict', 'OrderedDict']):
        a = {}
        for k in obj:
            a[k] = get_name(obj[k])
        return a
    elif (type(obj).__name__ not in ['int', 'float', 'str', 'bool']):
        return type(obj).__name__
    else:
        return obj


def process_source(file_name, save_file):
    errorline = []
    with open(file_name, 'r', encoding='utf-8') as source:
        lines = source.readlines()
    with open(save_file, 'w+', encoding='utf-8') as save:
        for idx, line in enumerate(lines):
            code = line.strip()
            try:
                tokens = list(javalang.tokenizer.tokenize(code))
                tks = []
                for tk in tokens:
                    if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
                        tks.append('STR_')
                    elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
                        tks.append('NUM_')
                    elif tk.__class__.__name__ == 'Boolean':
                        tks.append('BOOL_')
                    else:
                        tks.append(tk.value)
                save.write(" ".join(tks) + '\n')
            except:
                errorline.append(idx)
    with open('errorline.pkl', 'wb') as pklf:
        pickle.dump(errorline, pklf)
    print(len(errorline))


def get_ast(file_name, w):
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(file_name.replace('source.code', 'token.nl'), 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()
    with open(w, 'w+', encoding='utf-8') as wf:
        ign_cnt = 0
        SyntaxErrorline = []
        for idx, line in tqdm(enumerate(lines)):
            code = line.strip()
            tokens = javalang.tokenizer.tokenize(code)
            token_list = list(javalang.tokenizer.tokenize(code))
            length = len(token_list)
            parser = javalang.parser.Parser(tokens)
            try:
                tree = parser.parse_member_declaration()
            except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
                print(code)
                SyntaxErrorline.append(idx)
                continue
            flatten = []
            for path, node in tree:
                flatten.append({'path': path, 'node': node})

            ign = False
            outputs = []
            stop = False
            for i, Node in enumerate(flatten):
                d = collections.OrderedDict()
                path = Node['path']
                node = Node['node']
                children = []
                for child in node.children:
                    child_path = None
                    if isinstance(child, javalang.ast.Node):
                        child_path = path + tuple((node,))
                        for j in range(i + 1, len(flatten)):
                            if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                                children.append(j)
                    if isinstance(child, list) and child:
                        child_path = path + (node, child)
                        for j in range(i + 1, len(flatten)):
                            if child_path == flatten[j]['path']:
                                children.append(j)
                d["id"] = i
                d["type"] = get_name(node)
                if children:
                    d["children"] = children
                value = None
                if hasattr(node, 'name'):
                    value = node.name
                elif hasattr(node, 'value'):
                    value = node.value
                elif hasattr(node, 'position') and node.position:
                    for i, token in enumerate(token_list):
                        if node.position == token.position:
                            pos = i + 1
                            value = str(token.value)
                            while (pos < length and token_list[pos].value == '.'):
                                value = value + '.' + token_list[pos + 1].value
                                pos += 2
                            break
                elif type(node) is javalang.tree.This \
                        or type(node) is javalang.tree.ExplicitConstructorInvocation:
                    value = 'this'
                elif type(node) is javalang.tree.BreakStatement:
                    value = 'break'
                elif type(node) is javalang.tree.ContinueStatement:
                    value = 'continue'
                elif type(node) is javalang.tree.TypeArgument:
                    value = str(node.pattern_type)
                elif type(node) is javalang.tree.SuperMethodInvocation \
                        or type(node) is javalang.tree.SuperMemberReference:
                    value = 'super.' + str(node.member)
                elif type(node) is javalang.tree.Statement \
                        or type(node) is javalang.tree.BlockStatement \
                        or type(node) is javalang.tree.ForControl \
                        or type(node) is javalang.tree.ArrayInitializer \
                        or type(node) is javalang.tree.SwitchStatementCase:
                    value = 'None'
                elif type(node) is javalang.tree.VoidClassReference:
                    value = 'void.class'
                elif type(node) is javalang.tree.SuperConstructorInvocation:
                    value = 'super'

                if value is not None and type(value) is type('str'):
                    d['value'] = value
                if not children and not value:
                    # print('Leaf has no value!')
                    print(type(node))
                    print(code)
                    ign = True
                    ign_cnt += 1
                    # break
                outputs.append(d)
            if not ign:
                wf.write(json.dumps(outputs))
                wf.write('\n')
    with open(file_name, 'w', encoding='utf-8') as s:
        for idx, line in enumerate(lines):
            if idx not in SyntaxErrorline:
                s.write(line)
    with open(file_name.replace('source.code', 'token.nl'), 'w', encoding='utf-8') as n:
        for idx, line in enumerate(lines2):
            if idx not in SyntaxErrorline:
                n.write(line)

    print(ign_cnt)

def data_split(save_path, source_code_file, random_seed, train_ratio, val_ratio, test_ratio):
    if not os.path.exists(save_path):  
        os.makedirs(save_path)
    total_num = 0
    for index, line in enumerate(open(source_code_file, 'r', encoding="utf-8")):
        total_num += 1
    print("total_num: ", total_num)
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(total_num)
    train_num = int(train_ratio * total_num)
    val_num = int(val_ratio * total_num)
    test_num = int(test_ratio * total_num)
    test_indices = shuffled_indices[:test_num]
    train_indices = shuffled_indices[test_num:test_num + train_num]
    valid_indices = shuffled_indices[test_num + train_num:]


    train = save_path + '/train'
    test = save_path + '/test'
    valid = save_path + '/valid'
    if not os.path.exists(train):
        os.makedirs(train)
    if not os.path.exists(test):
        os.makedirs(test)
    if not os.path.exists(valid):
        os.makedirs(valid)

    source_train_file = open(train + '/train.token.code', 'w', encoding='utf-8')
    source_test_file = open(test + '/test.token.code', 'w', encoding='utf-8')
    source_valid_file = open(valid + '/valid.token.code', 'w', encoding='utf-8')

    sbt_train_file = open(train + '/train.token.sbt', 'w', encoding='utf-8')
    sbt_test_file = open(test + '/test.token.sbt', 'w', encoding='utf-8')
    sbt_valid_file = open(valid + '/valid.token.sbt', 'w', encoding='utf-8')

    nl_train_file = open(train + '/train.token.nl', 'w', encoding='utf-8')
    nl_test_file = open(test + '/test.token.nl', 'w', encoding='utf-8')
    nl_valid_file = open(valid + '/valid.token.nl', 'w', encoding='utf-8')

    source_file = open(source_code_file, 'r', encoding='utf-8')
    nl_file = open(source_code_file.replace('source.code','token.nl'), 'r', encoding='utf-8')
    sbt_file = open(source_code_file.replace('source.code','token.ast'), 'r', encoding='utf-8')

    # a = sbt_file.readlines()
    # b = source_file.readlines()
    # c = nl_file.readlines()
    # print(len(a))
    # print(len(b))
    # print(len(c))

    source_line = source_file.readline()
    sbt_line = sbt_file.readline()
    nl_line = nl_file.readline()


    iter = 0
    while iter < total_num:
        if iter in train_indices:
            source_train_file.write(source_line)
            sbt_train_file.write(sbt_line)
            nl_train_file.write(nl_line)
        if iter in valid_indices:
            source_valid_file.write(source_line)
            sbt_valid_file.write(sbt_line)
            nl_valid_file.write(nl_line)
        if iter in test_indices:
            source_test_file.write(source_line)
            nl_test_file.write(nl_line)
            sbt_test_file.write(sbt_line)

        source_line = source_file.readline()
        sbt_line = sbt_file.readline()
        nl_line = nl_file.readline()
        iter += 1
        print('\r', iter, '%.2f' % (iter / total_num * 100), '%', end='')

    source_train_file.close()
    source_test_file.close()
    source_valid_file.close()

    sbt_train_file.close()
    sbt_test_file.close()
    sbt_valid_file.close()

    nl_train_file.close()
    nl_test_file.close()
    nl_valid_file.close()

    source_file.close()
    nl_file.close()
    sbt_file.close()




