
firstresult = [3508, 10050, 368, 14945, 6975, 1143, 525, 924, 1059, 571, 637, 491, 308, 223, 261, 476, 292, 492, 469, 216, 59, 268, 147, 74, 308, 166, 159, 125, 178, 125, 314, 46, 21, 71, 176, 227, 133, 119, 49, 205, 57, 318, 195, 89, 123, 26, 19, 16, 3892, 71, 56, 26, 59, 149, 114]
n= 5



firstavg = 0
secondavg = 0

firstavglist = []
secondavglist = []

track55 = 1
track5 = 1
while track55 <= 55:
    firstavg += firstresult.pop(0)
    print(firstavg)
    if track5 == 5:
        firstavglist.append(firstavg/5)
        firstavg = 0
    track55+=1

print(firstavglist)