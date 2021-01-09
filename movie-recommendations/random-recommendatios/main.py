import random

def main():
    file_name = "MovieHelper.csv"

    file = open(file_name, "r")

    for x in file:
        print(x.rstrip().replace("NULL", str(random.randint(0, 5))))


if __name__ == "__main__":
    main()