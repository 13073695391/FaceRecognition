import os


def mkdir(path):
    is_exist = os.path.exists(path)

    if is_exist:
        print(path + " 目录已存在")
        return len([lists for lists in os.listdir(path)])
    else:
        os.makedirs(path)
        print(path + " 创建成功")
        return 0
