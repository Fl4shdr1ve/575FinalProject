import csv

def convert_semicolon_to_comma_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter=';')
        writer = csv.writer(outfile, delimiter=',')
        for row in reader:
            writer.writerow(row)

# Example usage:
input_file = 'rand.csv'
output_file = 'random_opponents.csv'
convert_semicolon_to_comma_csv(input_file, output_file)