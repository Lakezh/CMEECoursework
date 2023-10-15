__appname__= '[application name here]'
__author__ = 'your name (e-mail adress)'
__version__ = '0.0.1'
__license__ = "Licese for this code/program"

import sys

def main(argv):
    print('This is a boiilerplate')
    return 0

if __name__ == "__main__":
    status = main(sys.argv)
    sys.exit(status)