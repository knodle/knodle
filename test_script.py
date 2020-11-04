def calculate_average(input_integers):
    sum = 0
    for curr_int in input_integers:
        sum += curr_int
    return sum / len(input_integers)


if __name__ == '__main__':
    calculate_average()

