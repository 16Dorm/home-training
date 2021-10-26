pre_label = -1
cur_label = -1
semi_count = 0
count = 0
prediction = 3
counting_idx = 0

table = [3,2,1,2]

while 1:
    cur_label = int(input())
    # cur_label = BKs_func()
    if pre_label != cur_label:
        if cur_label == prediction:
            if semi_count == 4:
                count += 1

            semi_count = (semi_count+1) % 5
            counting_idx += 1
            prediction = table[(counting_idx) % 4]

        else:
            semi_count = 0
            counting_idx = 0
            prediction = 3

    pre_label = cur_label

    print(semi_count, count, prediction)