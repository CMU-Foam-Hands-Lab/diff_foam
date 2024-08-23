import re

def parse_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # 提取数字数据部分
            match = re.search(r'Data: array\(\'d\', \[(.*?)\]\)', line)
            if match:
                numbers = list(map(float, match.group(1).split(', ')))
                data.append(numbers)
    return data

def calculate_min_max(data):
    if not data:
        return []

    num_columns = len(data[0])
    min_values = [float('inf')] * num_columns
    max_values = [float('-inf')] * num_columns

    for row in data:
        for i in range(num_columns):
            min_values[i] = min(min_values[i], row[i])
            max_values[i] = max(max_values[i], row[i])
    
    return min_values, max_values

def write_results(filename, min_values, max_values):
    with open(filename, 'w') as file:
        file.write("Column\tMin Value\tMax Value\n")
        for i, (min_val, max_val) in enumerate(zip(min_values, max_values)):
            file.write(f"{i + 1}\t{min_val:.6f}\t{max_val:.6f}\n")

def main():
    input_file = '/home/foamlab/nw/mydata/dice/pkls/cmd_xarm_sync.txt'
    output_file = '/home/foamlab/nw/mydata/dice/pkls/statistics_cmd_xarm.txt'
    
    data = parse_file(input_file)
    min_values, max_values = calculate_min_max(data)
    write_results(output_file, min_values, max_values)
    print(f"Statistics have been written to {output_file}")

if __name__ == '__main__':
    main()
