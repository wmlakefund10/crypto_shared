import utility
import data

def __main__():
    data_dir = r'/Users/facaiwang/Nutstore Files/WMLakefund/code/data/'
    pfh = data.Public_Trading_History(data_dir)
    pfh.pulling()

if __name__ == "__main__":
    __main__()