def str2top1(a):
    num_lst = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i in range(len(a)-4):
        if a[i:i+4] == "top1":
            top1_acc = a[i+7:i+12]
            while top1_acc[-1] not in num_lst:
                top1_acc = top1_acc[:-1]
            return top1_acc
    return 0