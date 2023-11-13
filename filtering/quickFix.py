import glob
import os
import sys



files = glob.glob('datasets/**/*.txt', recursive=True)

for file in files:
    # Each file contains a number of lines each formatted as follows:
    # <class_index> <x_center> <y_center> <width> <height>
    # where the coordinates are relative to the image's width and height.
    # Some of the widths and heights are negative, which is not allowed, they simply need to be converted to the absolute value.

    # Read the file
    with open(file, 'r') as f:
        lines = f.readlines()

    # Fix the lines
    for i, line in enumerate(lines):
        line = line.split(' ')
        line[3] = str(min(abs(float(line[3])), 1.0))
        line[4] = str(min(abs(float(line[4])), 1.0))
        lines[i] = ' '.join(line)
    
    # Write the file
    with open(file, 'w') as f:
        f.writelines(lines)

    print('Fixed file: {}'.format(file))

print('Done.')
