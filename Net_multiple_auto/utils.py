import sys
import yaml
import os
import argparse

################################################################################
################################################################################
#read yml files in opencv format, does not suport levels 
def u_readYAMLFile(fileName):
    ret = {}
    skip_lines=1    # Skip the first line which says "%YAML:1.0". Or replace it with "%YAML 1.0"
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        #myRe = re.compile(r":([^ ])")   # Add space after ":", if it doesn't exist. Python yaml requirement
        #yamlFileOut = myRe.sub(r': \1', yamlFileOut)
        ret = yaml.load(yamlFileOut)
    return ret

################################################################################
################################################################################
def u_save2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    F.write(data)
    F.close()

################################################################################
################################################################################
def u_saveList2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        F.write(item + '\n')
    F.close()

################################################################################
################################################################################
def u_saveArray2File(file_name, data):
    print('Saving data in: ' + file_name)
    F = open(file_name,'w') 
    for item in data:
        F.write(item)
        F.write('\n')
    F.close()

################################################################################
################################################################################
def u_fileList2array(file_name):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        item = item.replace('\\', '/').rstrip()
        lst.append(item)
    F.close()
    return lst
################################################################################
################################################################################
def u_fileNumberList2array(file_name):
    print('Loading data from: ' + file_name)
    F = open(file_name,'r') 
    lst = []
    for item in F:
        lst.append(float(item))
    F.close()
    return lst

################################################################################
################################################################################
def u_listDirs(directory, token):
    lst = []
    for directory, dirs, files in os.walk(directory): 
        if directory.find(token) >= 0:
            lst.append(directory)
    return lst

################################################################################
################################################################################
def u_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

################################################################################
################################################################################
def u_listFileAll(directory, token):
    list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(token):
                 list.append(os.path.join(root, file))
    return list

################################################################################
################################################################################
def u_getPath(file):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('inputpath', nargs='?', 
                        help='The input path. Default = auto_conf.json')
    args = parser.parse_args()
    return args.inputpath if args.inputpath is not None else file
   

