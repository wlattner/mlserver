#!/usr/bin/env python3

if __name__ == "__main__":
	import sys

	var_name = sys.argv[1]

	sys.stdout.write("package main\n\nvar {} = `\n".format(var_name))
	for line in sys.stdin:
		sys.stdout.write(line)
	sys.stdout.write("\n`\n")