#!/usr/bin/python3

def convert_input_line(values):
    values = values[:-1]
    return ":".join("%d,%s" % (i,j) for (i, j) in enumerate(values))


def convert_output_line(values, indices):
    name = values[-1]
    try:
        index = indices.index(name)
        return str(index)
    except ValueError:
        indices.append(name)
        return str(len(indices) - 1)

def generate_files(file):
    i = 0
    output_indices = []
    with open("input.dsstne", "w") as input:
        with open("output.dsstne", "w") as output:
            while True:
                line = file.readline()
                if len(line) < 2:
                    return
                values = line[:-1].split(",")
                input_line = convert_input_line(values)
                output_line = convert_output_line(values, output_indices)
                input.write("%d\t%s\n" % (i, input_line))
                output.write("%d\t%s\n" % (i, output_line))
                i += 1

if __name__ == "__main__":
    data = open("iris.data", "r")
    generate_files(data)
